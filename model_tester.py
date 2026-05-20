import joblib
import pandas as pd

def test_model_manually():
    # 1. Load the bundle
    bundle_path = 'mlb_summer_quant_bundle.joblib'
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    required_features = bundle['features']

    # 2. Hard-coded Feature Dictionary
    # Input your specific baseball state values here:
    input_features = {
        'inning': 1,
        'balls': 0,
        'strikes': 0,
        'outs': 0,
        'r1': 0,              # 0 or 1
        'r2': 0,              # 0 or 1
        'r3': 0,              # 0 or 1
        'score_diff': 0,
        're24_val': 0.5,      # Run Expectancy value
        'leverage_index': 1.0,
        'platoon_advantage': 0, # 0 or 1
        'pit_era': 3.50,
        'pit_whip': 1.15,
        'pit_k9': 8.5,
        'pit_bb9': 2.5,
        'pit_hr9': 1.2,
        'bat_ops': 0.750,
        'bat_obp': 0.320,
        'bat_avg': 0.250
    }

    # 3. Validation and Inference
    # Create DataFrame and force the correct column order to match training
    df = pd.DataFrame([input_features])
    
    # Ensure all features are present
    if not all(f in df.columns for f in required_features):
        missing = [f for f in required_features if f not in df.columns]
        print(f"❌ ERROR: Missing features in dictionary: {missing}")
        return

    df = df[required_features] 
    tv = model.predict_proba(df.values)[:, 1][0]

    # 4. Output
    print("-" * 40)
    print(f"Model Prediction: {tv:.4f}")
    print("-" * 40)
    for k, v in input_features.items():
        print(f"{k:<20}: {v}")
    print("-" * 40)

if __name__ == "__main__":
    test_model_manually()