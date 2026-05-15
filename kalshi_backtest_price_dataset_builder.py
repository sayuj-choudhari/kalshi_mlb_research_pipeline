import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_and_merge_event_markets(event_db="kalshi_event_market_map.db", 
                                  market_db="kalshi_unified_market_data.db"):
    con = duckdb.connect()
    con.execute(f"ATTACH '{event_db}' AS events")
    con.execute(f"ATTACH '{market_db}' AS unified")

    mapping_df = con.execute("SELECT event_ticker, market_ticker FROM events.event_market_map").df()
    unique_events = mapping_df['event_ticker'].unique()

    event_price_data = {}

    for event in unique_events:
        # 1. Identify Home vs Away via Ticker Suffixes
        # Event format: KXMLBGAME-25AUG26PHINYM -> Team Block is 'PHINYM'
        team_block = event.split('-')[-1]
        markets = mapping_df[mapping_df['event_ticker'] == event]['market_ticker'].tolist()
        
        if len(markets) < 2:
            print(f"⚠️ Skipping {event}: Found less than 2 markets.")
            continue
            
        home_market = None
        away_market = None
        
        # Determine Home by checking which market team code is at the END of the team_block
        for m in markets:
            m_team_code = m.split('-')[-1] # Gets 'PHI' or 'NYM'
            if team_block.endswith(m_team_code):
                home_market = m
            else:
                away_market = m

        if not home_market or not away_market:
            print(f"⚠️ Could not resolve Home/Away for {event}")
            continue

        print(f"🔄 Event: {event} | Home (T1): {home_market.split('-')[-1]} | Away (T2): {away_market.split('-')[-1]}")

        # 2. Fetch data: T1 = Home, T2 = Away
        ordered_tickers = [home_market, away_market]
        market_dfs = []

        for i, ticker in enumerate(ordered_tickers, 1):
            query = f"""
                SELECT 
                    timestamp,
                    yes_price AS yes_team_{i},
                    no_price AS no_team_{i},
                    source AS source_{i},
                    volume AS volume_{i}
                FROM unified.unified_market_data
                WHERE ticker = '{ticker}'
            """
            m_df = con.execute(query).df()
                
            

            
                     
            market_dfs.append(m_df)

        if market_dfs:
            # 3. Point-in-time merge (No Forward Fill)
            final_df = market_dfs[0]
            for next_df in market_dfs[1:]:
                final_df = pd.merge(final_df, next_df, on='timestamp', how='outer')

            final_df = final_df.sort_values('timestamp').reset_index(drop=True)

            new_df = pd.DataFrame()
            new_df['timestamp'] = final_df['timestamp']

            # 4. Coalesce Price Logic (T1 is Home)
            # max_sell: Highest price you can sell the Home team for (Bid)
            t1_bid = 1 - final_df['no_team_1']
            t2_equiv_bid = 1 - final_df['yes_team_2']
            new_df['max_sell'] = np.fmax(t1_bid, t2_equiv_bid)

            # min_buy: Lowest price you can buy the Home team for (Ask)
            t1_ask = final_df['yes_team_1']
            t2_equiv_ask = final_df['no_team_2']
            new_df['min_buy'] = np.fmin(t1_ask, t2_equiv_ask)

            # Metrics
            new_df['spread'] = new_df['min_buy'] - new_df['max_sell']
            v1 = final_df['volume_1'].fillna(0)
            v2 = final_df['volume_2'].fillna(0)
            new_df['volume'] = np.maximum(v1, v2)
            
            new_df['event'] = event
            new_df['home_team'] = home_market.split('-')[-1]

            # Drop unusable timestamps
            new_df.dropna(subset=['max_sell', 'min_buy'], how='all', inplace=True)
            event_price_data[event] = new_df


    con.close()
    return event_price_data

def save_processed_research(games_dict, output_db="kalshi_price_data.db"):
    if not games_dict:
        print("No data to save.")
        return

    master_df = pd.concat(games_dict.values(), ignore_index=True)

    con = duckdb.connect(output_db)
    con.execute("CREATE OR REPLACE TABLE market_prices AS SELECT * FROM master_df")
    con.execute("CREATE INDEX idx_event ON market_prices (event)")

    print(f"✅ Saved {len(master_df)} rows to {output_db} (Team 1 = Home)")
    con.close()

if __name__ == "__main__":
    games_dict = fetch_and_merge_event_markets()
    save_processed_research(games_dict)