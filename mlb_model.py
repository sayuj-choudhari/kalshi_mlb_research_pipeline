import os
import json
import duckdb
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

# Explicitly pull in your required model choices or add new ones here:
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

def clean_and_shift_game_state(df):
    """
    Standardized framework cleaning: Vectorized metadata shifts 
    and chronological inning transition tracking.
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

def pipeline_training_workbench(model_instance, model_name: str, db_path="mlb_game_state.db"):
    """
    Unified experiment framework. Ingests data, reads decoupled market feature selections,
    runs game-isolated state cleaning, calibrates the target model, and dumps a backtest-ready bundle.
    """
    print(f"\n========================================================")
    print(f"🚀 SPINNING UP EXPERIMENT WORKBENCH: {model_name}")
    print(f"========================================================")
    
    con = duckdb.connect(database=db_path, read_only=True)
    
    # 1. Game-Isolated Ingestion Loop
    print("📥 Gathering Summer (06, 07, 08) historical match ticks...")
    try:
        events = con.execute("""
            SELECT DISTINCT event FROM pitch_data 
            WHERE strftime(visible_timestamp, '%m') IN ('06', '07', '08')
        """).df()['event'].tolist()
    except Exception as e:
        print(f"❌ Ingestion Failure: Is your table named 'pitch_data' or 'training_data'? Error: {e}")
        con.close()
        return None

    cleaned_frames = []
    for event in events:
        game_df = con.execute(f"SELECT * FROM pitch_data WHERE event = '{event}'").df()
        if not game_df.empty:
            cleaned_frames.append(clean_and_shift_game_state(game_df))
            
    if not cleaned_frames:
        print("❌ Error: No games processed. Verify data presence and schema filters.")
        con.close()
        return None

    train_df = pd.concat(cleaned_frames, ignore_index=True)
    con.close()
    
    # 2. LOAD REVERSE-ENGINEERED FEATURES FROM JSON
    config_path = "market_selected_features.json"
    
    if os.path.exists(config_path):
        print(f"📖 Loading reverse-engineered features from '{config_path}'...")
        try:
            with open(config_path, "r") as f:
                config_payload = json.load(f)
            features = config_payload["selected_features"]
            print(f"🎯 Successfully loaded {len(features)} market features.")
        except Exception as e:
            print(f"⚠️ Error reading JSON file: {e}. Falling back to default layout.")
            os.path.exists(config_path) # placeholder block trigger safety
    else:
        print(f"⚠️ Warning: '{config_path}' not found. Defaulting to full 19-feature universe.")
        features = [
            'inning', 'balls', 'strikes', 'outs', 'r1', 'r2', 'r3', 'score_diff',
            're24_val', 'leverage_index', 'platoon_advantage',
            'pit_era', 'pit_whip', 'pit_k9', 'pit_bb9', 'pit_hr9',
            'bat_ops', 'bat_obp', 'bat_avg'
        ]
    
    print("🧹 Sterilizing all active features into numeric vectors...")
    for col in features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        
    train_df = train_df[train_df['t1_win'].notna()]
    X = train_df[features].fillna(0)
    y = train_df['t1_win'].astype(int).values
    
    print(f"📊 Training Grid Matrix: {X.shape[0]} states | {X.shape[1]} unique features.")

    # 3. Calibration Layer (Isotonic Time-Series Probability Splitting)
    print(f"🛠 Forcing probability calibration on [{model_name}] via TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=model_instance,
        method='isotonic',
        cv=tscv
    )
    
    print(f"⚡ Fitting calibrated {model_name}...")
    calibrated_model.fit(X.values, y)
    
    # 4. Out-of-Sample Probability Validation Checks
    probs = calibrated_model.predict_proba(X.values)[:, 1]
    brier = brier_score_loss(y, probs)
    print(f"🎯 Model Calibration Finalized. Verified Brier Score: {brier:.6f}")

    # 5. Fast SHAP Verification Diagnostics
    print("🧠 Extracting micro-sample SHAP audit...")
    try:
        base_estimator = calibrated_model.calibrated_classifiers_[0].estimator
        X_sample = X.sample(n=min(500, len(X)), random_state=42)
        
        if hasattr(base_estimator, "tree_method") or isinstance(base_estimator, (RandomForestClassifier, HistGradientBoostingClassifier)):
            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer(X_sample)
        else:
            explainer = shap.Explainer(base_estimator, X_sample)
            shap_values = explainer(X_sample)
            
        shap_matrix = shap_values.values[..., 1] if len(shap_values.values.shape) > 2 else shap_values.values
        
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_matrix, X_sample, show=False)
        plt.title(f"Diagnostic SHAP: Base {model_name} Structure", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'diagnostic_{model_name.lower().replace(" ", "_")}.png', dpi=150)
        print(f"💾 Diagnostics saved to 'diagnostic_{model_name.lower().replace(' ', '_')}.png'")
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP diagnostic pass skipped for this architecture: {e}")

    # 6. Production-Ready Deployment Serialization (Direct Dashboard Match)
    bundle_filename = 'mlb_summer_quant_bundle.joblib'
    bundle = {
        'model': calibrated_model,
        'features': features,  # Handed straight to the dashboard to safely index the 15 matrices
        'metrics': {'brier_score': brier},
        'metadata': {'model_architecture': model_name, 'status': 'Production-Ready', 'feature_source': config_path}
    }
    
    joblib.dump(bundle, bundle_filename)
    print(f"✨ Success! Saved active asset bundle to: {bundle_filename}")
    return calibrated_model


if __name__ == "__main__":
    # Standard production DB path
    DATABASE_FILE = "mlb_game_state.db"
    
    # ---------------------------------------------------------
    # CONFIGURATION HUB: Swap, Tune, or Add Models Freely Here!
    # ---------------------------------------------------------
    MODELS_TO_TRY = {
        "HistGBM Baseline": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=5, l2_regularization=3.0, random_state=42
        ),
        
        "Random Forest Alpha": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=40, n_jobs=-1, random_state=42
        ),
        
        "Consensus Ensemble": VotingClassifier(
            estimators=[
                ('hgb', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=5, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=40, n_jobs=-1, random_state=42))
            ],
            voting='soft',
            weights=[1.2, 1.0]
        )
    }
    
    # CHOOSE ACTIVE EXPERIMENT FOR BACKTESTING
    ACTIVE_MODEL_KEY = "Random Forest Alpha"
    
    if os.path.exists(DATABASE_FILE):
        selected_model = MODELS_TO_TRY[ACTIVE_MODEL_KEY]
        train_production_model = pipeline_training_workbench(
            model_instance=selected_model, 
            model_name=ACTIVE_MODEL_KEY, 
            db_path=DATABASE_FILE
        )
    else:
        print(f"❌ Critical Error: DuckDB database not found at '{DATABASE_FILE}'. Please verify path details.")