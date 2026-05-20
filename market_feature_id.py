import os
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json  # Add to your imports at the top

def clean_and_shift_game_state(df):
    """
    Vectorized cleaning: Shifts metadata up and corrects inning transitions.
    Matches the exact structural adjustments of the predictive pipeline.
    """
    df = df.sort_values('visible_timestamp').reset_index(drop=True)
    
    # 1. Force 'half' to object to avoid categorical warnings
    df['half'] = df['half'].astype(object)
    
    # 2. Shift Metadata UP: Grab pitcher/batter stats from the NEXT row
    metadata_cols = [col for col in df.columns if col.startswith(('pit_', 'bat_')) or col == 'platoon_advantage']
    shifted_metadata = df[metadata_cols].shift(-1)
    
    # Only apply the shift where the current row is a terminal_reset (end of play)
    reset_mask = df['event_type'] == 'terminal_reset'
    df.loc[reset_mask, metadata_cols] = shifted_metadata.loc[reset_mask]

    # 3. Handle Inning Transitions (Where outs >= 3)
    transition_mask = df['outs'] >= 3
    df.loc[transition_mask & (df['half'] == 'top'), 'half'] = 'bottom'
    
    bottom_to_top_mask = transition_mask & (df['half'] == 'bottom')
    df.loc[bottom_to_top_mask, 'half'] = 'top'
    df.loc[bottom_to_top_mask, 'inning'] += 1
    
    # Reset outs to 0 for these transition states
    df.loc[transition_mask, 'outs'] = 0

    # 4. Binary Encoding for Model
    df['half'] = df['half'].map({'top': 0, 'bottom': 1})
    
    return df

def run_market_reverse_engineer(db_path: str):
    """
    Connects to the pre-merged DuckDB database, processes game states game-by-game
    using the exact production cleanup logic, and extracts reverse-engineered weights.
    """
    print(f"Opening DuckDB database at: {db_path}")
    con = duckdb.connect(database=db_path, read_only=True)
    
    # 1. System Sanity Check: Print columns and a quick info summary
    print("\n=== SYSTEM SANITY CHECK: RAW COLUMNS ===")
    try:
        raw_preview = con.execute("SELECT * FROM training_data LIMIT 5").df()
        print(raw_preview.info())
    except Exception as e:
        print(f"❌ Failed to read training_data table schema: {e}")
    print("=========================================\n")

    # 2. Extract unique events to execute the game-by-game chronological vector shifts
    print("📡 Pulling complete data universe for game-by-game preprocessing...")
    try:
        events = con.execute("SELECT DISTINCT event_ticker FROM training_data").df()['event_ticker'].tolist()
    except Exception as e:
        print(f"❌ Failed to extract event list: {e}")
        return None, None
        
    cleaned_frames = []
    print(f"Processing and shifting {len(events)} discrete event strings...")
    for event in events:
        game_df = con.execute(f"SELECT * FROM training_data WHERE event_ticker = '{event}'").df()
        if not game_df.empty:
            cleaned_frames.append(clean_and_shift_game_state(game_df))
            
    if not cleaned_frames:
        print("❌ Error: No data frames survived the ingestion filter step.")
        return None, None
        
    # Consolidate all cleanly processed game pipelines
    processed_df = pd.concat(cleaned_frames, ignore_index=True)
    
    # 3. EXACT FEATURE SELECTION REQUESTED
    features = [
        # Core Structural Features
        'inning', 'balls', 'strikes', 'outs', 'r1', 'r2', 'r3', 'score_diff',
        
        # Custom Advanced Features
        're24_val', 'leverage_index', 'platoon_advantage',
        
        # Pitcher Metrics
        'pit_era', 'pit_whip', 'pit_k9', 'pit_bb9', 'pit_hr9',
        
        # Batter Metrics
        'bat_ops', 'bat_obp', 'bat_avg'
    ]
    
    # Calculate target pricing directly from inside market components
    processed_df['kalshi_mid_price'] = (processed_df['max_sell'] + processed_df['min_buy']) / 2.0
    
    print("🧹 Sterilizing all features into numeric vectors...")
    for col in features:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
    # Standardize boundary filters on the target variable
    valid_idx = (
        processed_df['kalshi_mid_price'].notna() & 
        processed_df['kalshi_mid_price'].between(0.01, 0.99) &
        processed_df['max_sell'].notna() &
        processed_df['min_buy'].notna()
    )
    
    processed_df = processed_df[valid_idx]
    
    # Impute missing values with 0 exactly like production to protect feature shapes
    X = processed_df[features].fillna(0)
    y = processed_df['kalshi_mid_price']
    
    print(f"📊 Training Matrix Shape: {X.shape[0]} states with {X.shape[1]} clean variables.")
    
    # 4. Train the Shadow Pricing Model
    print("Training XGBoost Shadow Pricing Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    r2_score = model.score(X, y)
    print(f"Shadow Model R^2 Score: {r2_score:.4f}")
    
    # 5. Generate SHAP Explanations via Background Sampling
    print("Calculating SHAP values using background matrix sample...")
    explainer = shap.TreeExplainer(model)
    
    # Fast evaluation buffer to keep processing times under 30 seconds
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    shap_values = explainer(X_sample)
    
    # Output Visualizations
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("The Market's Brain Map: Feature Importance on Kalshi Prices", fontsize=14)
    plt.tight_layout()
    plt.savefig('market_brain_summary.png', dpi=300)
    print("Saved clean brain map plot to 'market_brain_summary.png'")
    plt.close()
    
    # 6. Quantify through Structured DataFrame
    vals = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'mean_abs_shap'])
    feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    
    print("\n=== REVERSE-ENGINEERED MARKET WEIGHTS ===")
    print(feature_importance.to_string())

    vals = np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'mean_abs_shap'])
    feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    
    print("\n=== REVERSE-ENGINEERED MARKET WEIGHTS ===")
    print(feature_importance.to_string())
    
    # --- NEW: SAVE TOP 10 TO METADATA CONFIG ---
    top_10_features = feature_importance.head(19)['feature'].tolist()
    config_payload = {
        "market_model_architecture": "XGBRegressor_Shadow_Pricing",
        "selected_features": top_10_features
    }
    
    config_path = "market_selected_features.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=4)
    print(f"\n💾 Saved top 10 reverse-engineered features to: {config_path}")
    # --------------------------------------------
    
    return feature_importance, model

if __name__ == "__main__":
    DB_PATH = "mlb_model_training_cleaned.db"
    
    if os.path.exists(DB_PATH):
        feat_imp, shadow_model = run_market_reverse_engineer(DB_PATH)
    else:
        print(f"❌ DB file not found at {DB_PATH}. Check your current working directory.")