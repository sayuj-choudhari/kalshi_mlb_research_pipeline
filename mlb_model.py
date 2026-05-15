import pandas as pd
import numpy as np
import duckdb
import joblib
from sklearn.ensemble import (
    HistGradientBoostingClassifier, 
    RandomForestClassifier, 
    VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

def clean_and_shift_game_state(df):
    """
    Vectorized cleaning: Shifts metadata up and corrects inning transitions.
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

def train_production_model(db_path="mlb_game_state.db"):
    con = duckdb.connect(db_path)
    
    # 1. Ingestion: Filtered strictly for Summer (06, 07, 08)
    print("📥 Pulling Summer (06, 07, 08) data universe...")
    events = con.execute("""
        SELECT DISTINCT event FROM pitch_data 
        WHERE strftime(visible_timestamp, '%m') IN ('06', '07', '08')
    """).df()['event'].tolist()
    
    cleaned_frames = []
    for event in events:
        game_df = con.execute(f"SELECT * FROM pitch_data WHERE event = '{event}'").df()
        if not game_df.empty:
            cleaned_frames.append(clean_and_shift_game_state(game_df))
    
    if not cleaned_frames:
        print("❌ Error: No data found for the specified months.")
        return

    train_df = pd.concat(cleaned_frames, ignore_index=True)
    
    # 2. Feature Selection & Hardened Sterilization
    features = [
        'inning', 'half', 'outs', 'score_diff', 'r1', 'r2', 'r3', 
        're24_val', 'leverage_index', 
        'pit_era', 'pit_whip', 'pit_k9', 'pit_bb9', 'pit_hr9', 'pit_k_bb_ratio',
        'bat_ops', 'bat_obp', 'bat_avg', 'platoon_advantage'
    ]
    
    print("🧹 Sterilizing numeric features (handling '-.--' strings)...")
    for col in features:
        # errors='coerce' turns strings like '-.--' into NaN
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    X = train_df[features].fillna(0).values
    y = train_df['t1_win'].astype(int).values
    
    print(f"📊 Training on {X.shape[0]} states with {len(features)} features.")

    # 3. Model Architecture: Consensus Ensemble
    # Random Forest for structural stability
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=50, 
        n_jobs=-1,
        random_state=42
    )

    # HistGradientBoosting for probability precision (handles large data well)
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=5.0,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb)],
        voting='soft',
        weights=[1, 1.2] # Slight weight toward HGB for better alpha
    )

    # 4. Calibration Layer (The Pricing Engine)
    print("🛠 Calibrating Fair-Value Engine via TimeSeriesSplit...")
    # TimeSeriesSplit ensures we don't cheat by looking at the "future"
    tscv = TimeSeriesSplit(n_splits=5)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=ensemble,
        method='isotonic',
        cv=tscv
    )

    calibrated_model.fit(X, y)

    # 5. Serialization and Metrics
    probs = calibrated_model.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, probs)
    
    bundle = {
        'model': calibrated_model,
        'features': features,
        'metrics': {'brier_score': brier},
        'metadata': {'months': ['06', '07', '08'], 'status': 'Production-Ready'}
    }
    
    joblib.dump(bundle, 'mlb_summer_quant_bundle.joblib')
    print(f"🚀 Success! Summer Model stored. Brier Score: {brier:.6f}")
    con.close()

if __name__ == "__main__":
    train_production_model()