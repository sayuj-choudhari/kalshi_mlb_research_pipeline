import pandas as pd
import duckdb
from tqdm import tqdm
import numpy as np

def clean_and_shift_game_state(df):
    """
    Adjusts game states so each row reflects the pitch outcome.
    Handles dtype compatibility, string-based metadata, and inning transitions.
    """
    if df.empty:
        return df
    
    # 1. Sort to ensure chronological shifting
    df = df.sort_values('visible_timestamp').reset_index(drop=True)
    
    # 2. Identify and filter metadata columns
    # We only shift numeric columns to avoid "could not convert string to float" errors
    potential_cols = [col for col in df.columns if col.startswith(('pit_', 'bat_')) or col == 'platoon_advantage']
    metadata_cols = df[potential_cols].select_dtypes(include=['number']).columns.tolist()
    
    # 3. Cast numeric metadata to float64 to safely handle NaNs introduced by the shift
    df[metadata_cols] = df[metadata_cols].astype('float64')
    
    # 4. Shift metadata back: Current row now contains the outcome of the pitch
    df.loc[:, metadata_cols] = df[metadata_cols].shift(-1)
    
    # 5. Terminal/Reset Logic
    # 3 outs signifies the end of a half-inning
    if 'outs' in df.columns:
        transition_mask = df['outs'] >= 3
        
        if 'half' in df.columns and 'inning' in df.columns:
            # Handle 'half' transition (0=Top, 1=Bottom)
            top_to_bottom = transition_mask & (df['half'] == 0)
            bottom_to_top = transition_mask & (df['half'] == 1)
            
            df.loc[top_to_bottom, 'half'] = 1
            df.loc[bottom_to_top, 'half'] = 0
            df.loc[bottom_to_top, 'inning'] += 1
        
        # Reset game-state features for the new half-inning
        df.loc[transition_mask, 'outs'] = 0
        # Optional: Reset runners if your columns follow a specific naming convention
        runner_cols = [c for c in df.columns if 'runner' in c.lower() or 'base' in c.lower()]
        if runner_cols:
            df.loc[transition_mask, runner_cols] = 0

    # 6. Fill NaNs at the end of the game to ensure data consistency
    df[metadata_cols] = df[metadata_cols].fillna(0)
    
    return df

def build_cleaned_price_anchored_data():
    """
    Main execution script to build a market-centric tape for ML training.
    """
    # 1. Database Connection
    try:
        con = duckdb.connect()
        con.execute("ATTACH 'kalshi_event_market_map.db' AS events")
        con.execute("ATTACH 'mlb_game_state.db' AS states")
        con.execute("ATTACH 'kalshi_price_data.db' AS markets")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return

    # 2. Fetch Event Map
    try:
        mapping_df = con.execute("SELECT event_ticker FROM events.event_market_map").df()
        unique_events = mapping_df['event_ticker'].unique()
    except Exception as e:
        print(f"❌ Error fetching event map: {e}")
        return

    all_merged_data = []

    print(f"🔄 Processing {len(unique_events)} events into price-anchored tape...")

    for event in tqdm(unique_events):
        try:
            # 3. Pull pitch and price data
            state_df = con.execute(f"SELECT * FROM states.pitch_data WHERE event = '{event}' ORDER BY visible_timestamp").df()
            price_df = con.execute(f"SELECT * FROM markets.market_prices WHERE event = '{event}' ORDER BY timestamp").df()

            if state_df.empty or price_df.empty:
                continue

            # 4. Apply Pitch-Outcome Shifting & Reset Logic
            state_df = clean_and_shift_game_state(state_df)

            # 5. Standardize Timestamps (Strip TZ for merge_asof)
            state_df['visible_timestamp'] = pd.to_datetime(state_df['visible_timestamp']).dt.tz_localize(None)
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp']).dt.tz_localize(None)

            # 6. Perform the Price-Anchored Merge
            # Creates a row for EVERY market update, joining the most recent game state
            merged_chunk = pd.merge_asof(
                price_df,
                state_df,
                left_on='timestamp',
                right_on='visible_timestamp',
                direction='backward'
            )

            # 7. Post-Merge Cleanup
            # Drop market activity that happened before the game actually started
            merged_chunk = merged_chunk.dropna(subset=['visible_timestamp'])
            merged_chunk['event_ticker'] = event
            
            all_merged_data.append(merged_chunk)

        except Exception as e:
            print(f"⚠️ Error processing {event}: {e}")
            continue

    # 8. Final Consolidation and Export
    if all_merged_data:
        final_df = pd.concat(all_merged_data, ignore_index=True)
        
        # Save to DuckDB for training
        output_db = "mlb_model_training_cleaned.db"
        with duckdb.connect(output_db) as out_con:
            out_con.execute("CREATE OR REPLACE TABLE training_data AS SELECT * FROM final_df")
        
        print(f"\n🚀 Success! Compiled {len(final_df)} rows into {output_db}")
    else:
        print("❌ Final merge failed: No data was successfully processed.")

if __name__ == "__main__":
    build_cleaned_price_anchored_data()