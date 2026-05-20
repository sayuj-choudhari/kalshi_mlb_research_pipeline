import os
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def clean_and_shift_game_state(df):
    """
    Vectorized cleaning: Shifts metadata up and corrects inning transitions.
    Matches the production model pipeline.
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

def run_outcome_feature_importance(db_path="mlb_game_state.db"):
    """
    Connects to the production database, runs game-by-game state cleaning,
    and uses an XGBoost Classifier to find which features most accurately predict t1_win.
    """
    print(f"Opening DuckDB database at: {db_path}")
    con = duckdb.connect(database=db_path, read_only=True)
    
    # 1. Ingestion: Filtered strictly for Summer (06, 07, 08) like production
    print("📥 Pulling Summer (06, 07, 08) data universe...")
    try:
        events = con.execute("""
            SELECT DISTINCT event FROM pitch_data 
            WHERE strftime(visible_timestamp, '%m') IN ('06', '07', '08')
        """).df()['event'].tolist()
    except Exception as e:
        print(f"❌ Failed to extract event list from pitch_data: {e}")
        return None, None
        
    cleaned_frames = []
    print(f"Processing and shifting {len(events)} discrete games...")
    for event in events:
        game_df = con.execute(f"SELECT * FROM pitch_data WHERE event = '{event}'").df()
        if not game_df.empty:
            cleaned_frames.append(clean_and_shift_game_state(game_df))
            
    if not cleaned_frames:
        print("❌ Error: No data found for the specified months.")
        return None, None
        
    # Consolidate all cleanly processed game pipelines
    train_df = pd.concat(cleaned_frames, ignore_index=True)
    
    # 2. Complete Expanded Feature Set Map
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
    
    print("🧹 Sterilizing all features into numeric vectors...")
    for col in features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        
    # Ensure our target binary outcome variable is clean and valid
    train_df = train_df[train_df['t1_win'].notna()]
    
    # Impute missing stats with 0 to preserve matrix shapes safely
    X = train_df[features].fillna(0)
    y = train_df['t1_win'].astype(int).values
    
    print(f"📊 Training Matrix Shape: {X.shape[0]} states with {X.shape[1]} clean features.")
    
    # 3. Train the Ultimate Outcome Model (XGBClassifier)
    print("Training XGBoost Production Outcome Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Calculate performance matrix benchmark
    accuracy = model.score(X, y)
    print(f"Outcome Model Training Accuracy: {accuracy:.4f}")
    
    # 4. Generate SHAP Explanations via Background Sampling
    print("Calculating SHAP values for structural outcome brain map...")
    explainer = shap.TreeExplainer(model)
    
    # Keep evaluation space bounded for lightning fast execution
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    shap_values = explainer(X_sample)
    
    # Note: For binary classification, TreeExplainer outputs a list of arrays 
    # if it's raw margin values, or single array depending on version. 
    # We grab values corresponding to the positive class (Class 1: Win).
    if isinstance(shap_values.values, list):
        shap_matrix = shap_values.values[1]
    else:
        shap_matrix = shap_values.values

    # 5. Output Visualizations
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_matrix, X_sample, show=False)
    plt.title("The Objective Game Brain Map: Feature Importance on T1 Wins", fontsize=14)
    plt.tight_layout()
    plt.savefig('ML_brain_summary.png', dpi=300)
    print("Saved global outcome brain map plot to 'ML_brain_summary.png'")
    plt.close()
    
    # 6. Quantify through Structured DataFrame
    vals = np.abs(shap_matrix).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'mean_abs_shap'])
    feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    
    print("\n=== REVERSE-ENGINEERED WIN-OUTCOME WEIGHTS ===")
    print(feature_importance.to_string())
    
    return feature_importance, model

if __name__ == "__main__":
    # Point this to your standard production database
    DB_PATH = "mlb_game_state.db"
    
    if os.path.exists(DB_PATH):
        feat_imp, outcome_model = run_outcome_feature_importance(DB_PATH)
    else:
        print(f"❌ DB file not found at {DB_PATH}. Check your current working directory.")