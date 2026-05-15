import statsapi
import pandas as pd
import duckdb
from datetime import datetime, timedelta

# Configuration
MARKET_DATA_DB = "kalshi_price_data.db"
STATE_DB = "mlb_game_state.db"
RESULTS_DB = "mlb_game_results.db"
REPORTING_LAG_SECONDS = 15 

# Shared Cache for Player Stats: {(player_id, date): stats_dict}
PLAYER_CACHE = {}

TEAM_MAP = {
    "AZ": "ARIZONA", "ARI": "ARIZONA",
    "ATL": "ATLANTA",
    "BAL": "BALTIMORE",
    "BOS": "BOSTON",
    "CHC": "CHICAGO CUBS", 
    "CHW": "CHICAGO WHITE SOX",
    "CWS": "CHICAGO WHITE SOX",
    "CIN": "CINCINNATI",
    "CLE": "CLEVELAND",
    "COL": "COLORADO",
    "DET": "DETROIT",
    "HOU": "HOUSTON",
    "KC": "KANSAS CITY",
    "LAA": "LOS ANGELES ANGELS", # Exact mapping as requested
    "LAD": "LOS ANGELES DODGERS",
    "MIA": "MIAMI",
    "MIL": "MILWAUKEE",
    "MIN": "MINNESOTA",
    "NYM": "NEW YORK METS",
    "NYY": "NEW YORK YANKEES",
    "ATH": "ATHLETICS",
    "PHI": "PHILADELPHIA",
    "PIT": "PITTSBURGH",
    "SD": "SAN DIEGO",
    "SEA": "SEATTLE",
    "SF": "SAN FRANCISCO",
    "STL": "ST. LOUIS",
    "TB": "TAMPA BAY",
    "TEX": "TEXAS",
    "TOR": "TORONTO",
    "WSH": "WASHINGTON", "WAS": "WASHINGTON"
}

def prefetch_roster_stats(game_pk, game_date):
    """
    Fetches stats for EVERY player in the game in bulk calls.
    Drastically reduces API latency by avoiding per-pitch calls.
    """
    box = statsapi.boxscore_data(game_pk)
    # Collect all player IDs from both rosters
    ids = list(box['home']['players'].keys()) + list(box['away']['players'].keys())
    clean_ids = [pid.replace('ID', '') for pid in ids]
    
    season = game_date.split('-')[0]
    start_date = f"{season}-03-01"
    end_date = (datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    # Batch IDs in chunks of 50 to avoid API URL length limits
    chunk_size = 50
    for i in range(0, len(clean_ids), chunk_size):
        id_batch = ",".join(clean_ids[i:i+chunk_size])
        
        # Hydrate both pitching and hitting stats
        hydrate = f"stats(group=[pitching,hitting],type=[byDateRange],startDate={start_date},endDate={end_date},season={season})"
        raw = statsapi.get('people', {'personIds': id_batch, 'hydrate': hydrate})
        
        for person in raw.get('people', []):
            pid = person['id']
            # Default baselines
            p_stats = {"pit_era": 4.50, "pit_whip": 1.35, "bat_ops": .730, "bat_avg": .245}

            print(person.get('stats', []))
            
            for stat_group in person.get('stats', []):
                group_name = stat_group['group']['displayName']
                splits = stat_group.get('splits', [])
                if splits:
                    s = splits[0]['stat']
                    if group_name == 'pitching':
                        p_stats.update({"pit_era": s.get('era'), "pit_whip": s.get('whip'),
                                         "pit_strike_percentage": s.get('strikePercentage'),
                                         'pit_win_percentage': s.get('winPercentage'),
                                         'pit_strikeoutWalkRatio': s.get('strikeoutWalkRatio'),
                                         'pit_runsScoredPer9': s.get('runsScoredPer9')})
                    elif group_name == 'hitting':
                        p_stats.update({"bat_ops": s.get('ops'), "bat_avg": s.get('avg')})
            
            PLAYER_CACHE[(pid, game_date)] = p_stats

RE_MATRIX = {
    (0, (0,0,0)): 0.52, (0, (1,0,0)): 0.91, (0, (0,1,0)): 1.15, (0, (0,0,1)): 1.35,
    (0, (1,1,0)): 1.51, (0, (1,0,1)): 1.75, (0, (0,1,1)): 2.05, (0, (1,1,1)): 2.40,
    (1, (0,0,0)): 0.28, (1, (1,0,0)): 0.55, (1, (0,1,0)): 0.72, (1, (0,0,1)): 0.98,
    (1, (1,1,0)): 1.01, (1, (1,0,1)): 1.21, (1, (0,1,1)): 1.45, (1, (1,1,1)): 1.65,
    (2, (0,0,0)): 0.11, (2, (1,0,0)): 0.24, (2, (0,1,0)): 0.35, (2, (0,0,1)): 0.42,
    (2, (1,1,0)): 0.48, (2, (1,0,1)): 0.61, (2, (0,1,1)): 0.65, (2, (1,1,1)): 0.82
}

def get_situational_metrics(outs, r1, r2, r3, inning, score_diff):
    """Calculates RE24 and an approximated Leverage Index."""
    re = RE_MATRIX.get((min(outs, 2), (r1, r2, r3)), 0.0)
    
    # Leverage Index Approximation: 
    # High if: Late Inning, Close Score, and Runners on Base
    importance = 1.0 + (inning / 9.0) # Inning multiplier
    closeness = max(0.1, 4 - abs(score_diff)) # Score multiplier
    situation = 1.0 + re # Base occupancy multiplier
    li = (importance * closeness * situation) / 5.0 
    
    return round(re, 3), round(li, 3)

def prefetch_roster_statsV2(game_pk, game_date):
    box = statsapi.boxscore_data(game_pk)
    ids = list(box['home']['players'].keys()) + list(box['away']['players'].keys())
    clean_ids = [pid.replace('ID', '') for pid in ids]
    
    season = game_date.split('-')[0]
    start_date = f"{season}-03-01"
    end_date = (datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    chunk_size = 50
    for i in range(0, len(clean_ids), chunk_size):
        id_batch = ",".join(clean_ids[i:i+chunk_size])
        # hydrate = f"person(stats(group=[pitching,hitting],type=[byDateRange],startDate={start_date},endDate={end_date},season={season}))"
        hydrate = f"stats(group=[pitching,hitting],type=[byDateRange],startDate={start_date},endDate={end_date},season={season})"
        raw = statsapi.get('people', {'personIds': id_batch, 'hydrate': hydrate})
        
        for person in raw.get('people', []):
            pid = person['id']
            # ENHANCED BASELINES
            p_stats = {
                "pit_era": 4.50, "pit_whip": 1.35, "pit_k9": 8.0, "pit_bb9": 3.2, "pit_hr9": 1.2,
                "bat_ops": .730, "bat_obp": .320, "bat_avg": .245, "bat_hand": person.get('batSide', {}).get('code'),
                "pit_hand": person.get('pitchHand', {}).get('code')
            }
            
            for stat_group in person.get('stats', []):
                group_name = stat_group['group']['displayName']
                splits = stat_group.get('splits', [])
                if splits:
                    s = splits[0]['stat']
                    
                    if group_name == 'pitching':
                        p_stats.update({
                            "pit_era": s.get('era'), "pit_whip": s.get('whip'),
                            "pit_k9": s.get('strikeoutsPer9Inn'), "pit_bb9": s.get('walksPer9Inn'),
                            "pit_hr9": s.get('homeRunsPer9'), "pit_k_bb_ratio": s.get('strikeoutWalkRatio')
                        })
                    elif group_name == 'hitting':
                        p_stats.update({
                            "bat_ops": s.get('ops'), "bat_obp": s.get('obp'), "bat_avg": s.get('avg')
                        })

            PLAYER_CACHE[(pid, game_date)] = p_stats

def find_correct_game_pk(game_date, team_block):
    found_teams = [full for abbr, full in TEAM_MAP.items() if abbr in team_block]
    sched = statsapi.schedule(date=game_date)
    for game in sched:
        home, away = game['home_name'].upper(), game['away_name'].upper()
        if any(t in home for t in found_teams) and any(t in away for t in found_teams):
            return game
    return None

def build_mlb_databases():
    event_db = "kalshi_event_market_map.db" 
    con = duckdb.connect()
    con.execute(f"ATTACH '{event_db}' AS events")

    mapping_df = con.execute("SELECT event_ticker, market_ticker FROM events.event_market_map").df()
    unique_events = mapping_df['event_ticker'].unique()

    all_pitches, all_results = [], []

    for event in unique_events:
        team_block = event.split('-')[-1]
        date_raw = event.split('-')[1][:7]
        game_date = pd.to_datetime(date_raw, format='%y%b%d').strftime('%Y-%m-%d')

        g = find_correct_game_pk(game_date, team_block)
        if not g:
            print(f"❌ No game match: {event}")
            continue

        game_pk = g['game_id']
        print(f"📡 Processing: {event} (Game ID: {game_pk})")

        # 1. Bulk Fetch Stats for this game (Cache population)
        prefetch_roster_statsV2(game_pk, game_date)

        all_results.append({"event": event, "team_1_win": 1 if g.get('winning_team') == g['home_name'] else 0})

        # ... inside build_mlb_databases loop ...


        # --- INITIALIZE GLOBAL GAME STATE ---
        current_outs = 0
        current_r1, current_r2, current_r3 = 0, 0, 0
        current_inning, current_half = None, None
        current_home_score, current_away_score = 0, 0

        pbp = statsapi.get('game_playByPlay', {'gamePk': game_pk})

        for play in pbp.get('allPlays', []):
            about = play.get('about', {})
            inning, half = about.get('inning'), about.get('halfInning')
            
            if inning != current_inning or half != current_half:
                current_outs, current_r1, current_r2, current_r3 = 0, 0, 0, 0
                current_inning, current_half = inning, half

            p_stats = PLAYER_CACHE.get((play['matchup']['pitcher']['id'], game_date), {})
            b_stats = PLAYER_CACHE.get((play['matchup']['batter']['id'], game_date), {})
            score_diff_at_start = current_home_score - current_away_score
            
            moved_manually_ids = set()
            last_balls, last_strikes, last_pitch_count = 0, 0, 0

            # 1. EVENT ITEM LOOP
            for event_item in play.get('playEvents', []):
                details = event_item.get('details', {})
                is_pitch = event_item.get('isPitch', False)
                event_type = details.get('eventType', '')
                is_in_play = details.get('isInPlay', False)
                
                # Capture the current ID for manual moves
                event_player_id = event_item.get('player', {}).get('id')

                # Logic for mid-at-bat non-terminal actions (Steals/Pickoffs)
                if event_type == 'stolen_base_2b':
                    current_r1, current_r2 = 0, 1
                    if event_player_id: moved_manually_ids.add(event_player_id)
                elif event_type == 'stolen_base_3b':
                    current_r2, current_r3 = 0, 1
                    if event_player_id: moved_manually_ids.add(event_player_id)
                elif 'pickoff' in event_type and details.get('isOut'):
                    # Only increment outs and clear base if it's a mid-at-bat pickoff
                    current_outs += 1
                    if '1b' in event_type: current_r1 = 0
                    elif '2b' in event_type: current_r2 = 0
                    if event_player_id: moved_manually_ids.add(event_player_id)

                if is_pitch:
                    c = event_item.get('count', {})
                    last_balls, last_strikes = c.get('balls', last_balls), c.get('strikes', last_strikes)
                    last_pitch_count = event_item.get('pitchNumber', last_pitch_count)

                # CHECK FOR TERMINAL EVENT: Strikeout, Walk, or Ball In Play
                is_terminal = is_in_play or last_strikes >= 3 or last_balls >= 4

                if is_terminal:
                    # A. PERFORM PLAY-LEVEL RECONCILIATION IMMEDIATELY
                    play_runners = play.get('runners', [])
                    for runner_data in play_runners:
                        r_details = runner_data.get('details', {})
                        r_id = r_details.get('runner', {}).get('id')
                        
                        # Still respect manual moves like mid-count steals
                        if r_id in moved_manually_ids:
                            continue
                            
                        move = runner_data.get('movement', {})
                        s_base, e_base = move.get('start'), move.get('end')
                        # Check for out either in movement or detail
                        is_out_on_play = move.get('isOut') or r_details.get('isOut')

                        if is_out_on_play:
                            current_outs += 1

                        # Clear starting base
                        if s_base == '1B': current_r1 = 0
                        elif s_base == '2B': current_r2 = 0
                        elif s_base == '3B': current_r3 = 0

                        # Occupy end base if safe and not scoring
                        if not is_out_on_play:
                            if e_base == '1B': current_r1 = 1
                            elif e_base == '2B': current_r2 = 1
                            elif e_base == '3B': current_r3 = 1

                    # B. RESET COUNT FOR THE "NEW STATE"
                    last_balls, last_strikes = 0, 0
                    
                    # C. UPDATE SCORE (since the play is resolved)
                    current_home_score = play['result'].get('homeScore', current_home_score)
                    current_away_score = play['result'].get('awayScore', current_away_score)
                    score_diff_at_start = current_home_score - current_away_score

                    if current_outs >= 3:
                        current_outs, current_r1, current_r2, current_r3 = 3, 0, 0, 0
                        
                        # if current_half == 'top':
                        #     current_half = 'bottom'
                        # else:
                        #     current_half = 'top'
                        #     current_inning = int(current_inning) + 1 if current_inning else 1

                # 2. RECORD DATA POINT
                # This will now record the 0-0 state with updated runners/outs if terminal
                if is_pitch or event_type:
                    # start_time = pd.to_datetime(event_item.get('startTime'))
                    # state = {
                    #     "visible_timestamp": start_time + timedelta(seconds=REPORTING_LAG_SECONDS),
                    #     "inning": current_inning, "half": current_half,
                    #     "balls": last_balls, "strikes": last_strikes, "outs": current_outs,
                    #     "r1": current_r1, "r2": current_r2, "r3": current_r3,
                    #     "pitch_count": last_pitch_count, "score_diff": score_diff_at_start,
                    #     "event_type": "terminal_reset" if is_terminal else ("pitch" if is_pitch else "action")
                    # }''
                    start_time = pd.to_datetime(event_item.get('startTime'))
                    re_val, li_val = get_situational_metrics(current_outs, current_r1, current_r2, current_r3, current_inning, score_diff_at_start)
                    
                    # Platoon Advantage logic
                    is_platoon = 0
                    if p_stats.get('pit_hand') and b_stats.get('bat_hand'):
                        is_platoon = 1 if p_stats['pit_hand'] != b_stats['bat_hand'] else 0


                    state = {
                        "event": event,
                        "visible_timestamp": start_time + timedelta(seconds=REPORTING_LAG_SECONDS),
                        "inning": current_inning, "half": current_half,
                        "balls": last_balls, "strikes": last_strikes, "outs": current_outs,
                        "r1": current_r1, "r2": current_r2, "r3": current_r3,
                        "re24_val": re_val, "leverage_index": li_val, # NEW
                        "platoon_advantage": is_platoon,             # NEW
                        "pitch_count": last_pitch_count, "score_diff": score_diff_at_start,
                        "event_type": "terminal_reset" if is_terminal else ("pitch" if is_pitch else "action"),
                        "t1_win": 1 if g.get('winning_team') == g['home_name'] else 0
                    }

                    state.update({k: v for k, v in p_stats.items() if k.startswith('pit_')})
                    state.update({k: v for k, v in b_stats.items() if k.startswith('bat_')})
                    all_pitches.append(state)

            # 3. END OF PLAY CHECK
            # Inning closure: reset if 3 outs reached after terminal reconciliation
            if current_outs >= 3:
                current_outs, current_r1, current_r2, current_r3 = 0, 0, 0, 0

    # Save Pitch Data
    df_pitches = pd.DataFrame(all_pitches)
    df_pitches.to_csv('mlb_game_data.csv', index = False)

    with duckdb.connect(STATE_DB) as c_state:
        c_state.execute("CREATE OR REPLACE TABLE pitch_data AS SELECT * FROM df_pitches")
    
    # Save Results
    df_results = pd.DataFrame(all_results)
    with duckdb.connect(RESULTS_DB) as c_res:
        c_res.execute("CREATE OR REPLACE TABLE game_results AS SELECT * FROM df_results")

    print(f"✅ Finished. Processed {len(all_pitches)} pitches.")

if __name__ == "__main__":
    build_mlb_databases()