import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import joblib
import numpy as np
import random

# Use a clean plotting style
plt.style.use('ggplot')

def run_anchored_game_dashboard():
    # 1. Connect to the NEW cleaned database
    db_path = 'mlb_model_training_cleaned.db'
    bundle_path = 'mlb_summer_quant_bundle.joblib'
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        bundle = joblib.load(bundle_path)
        features = bundle['features']
        model = bundle['model']
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    # 2. Pick a random game from the processed tape
    available_events = con.execute("SELECT DISTINCT event_ticker FROM training_data").df()['event_ticker'].tolist()
    event = random.choice(available_events)
    print(f"📊 Analyzing Tape for Game: {event}")

    # 3. Load the anchored tape for this game
    df = con.execute(f"SELECT * FROM training_data WHERE event_ticker = '{event}' ORDER BY timestamp ASC").df()
    
    # 4. Re-calculate TV and Edge (ensuring numeric consistency)
    # Mapping 'half' if not already done in the tape
    if df['half'].dtype == object:
        df['half'] = df['half'].map({'top': 0, 'bottom': 1}).fillna(0)
        
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0).values
    df['tv'] = model.predict_proba(X)[:, 1]
    
    # Use actual dollar prices from the tape
    df['mid'] = (df['min_buy'] + df['max_sell']) / 2
    df['edge'] = df['tv'] - df['mid']

    # --- SIMULATION PARAMETERS ---
    starting_bankroll = 100.0
    current_cash = starting_bankroll
    entry_threshold, exit_threshold = 0.08, 0.02
    kelly_fraction, max_game_risk = 0.15, 0.10
    
    active_pos_dollars = 0 
    entry_price = 0
    pending_action = None
    pending_amount = 0
    
    equity_history, exposure_history = [], []

    # --- EXECUTION LOOP ---
    for i in range(len(df)):
        row = df.iloc[i]
        edge = row['edge']
        current_mid = row['mid']
        
        # A. Next-Tick Execution (Simulating Latency/Order Fill)
        if pending_action == 'entry_long':
            active_pos_dollars = pending_amount
            entry_price = current_mid
            pending_action = None
        elif pending_action == 'entry_short':
            active_pos_dollars = -pending_amount
            entry_price = current_mid
            pending_action = None
        elif pending_action == 'exit':
            side = 1 if active_pos_dollars > 0 else -1
            ret = (current_mid / entry_price) - 1 if side == 1 else 1 - (current_mid / entry_price)
            current_cash += (abs(active_pos_dollars) + (abs(active_pos_dollars) * ret))
            active_pos_dollars, entry_price = 0, 0
            pending_action = None

        # B. Signal Detection
        if active_pos_dollars != 0:
            side = 1 if active_pos_dollars > 0 else -1
            # Exit if edge thins, or at final tick
            if (side == 1 and edge < exit_threshold) or (side == -1 and edge > -exit_threshold) or (i == len(df) - 1):
                pending_action = 'exit'
        elif abs(edge) > entry_threshold and current_cash > 0 and pending_action is None:
            # Kelly sizing
            denom = current_mid if edge > 0 else (1 - current_mid)
            if denom > 0:
                raw_kelly = (abs(edge) / denom) * kelly_fraction
                pending_amount = min(current_cash * raw_kelly, starting_bankroll * max_game_risk)
                if current_cash >= pending_amount:
                    current_cash -= pending_amount
                    pending_action = 'entry_long' if edge > 0 else 'entry_short'

        # C. MTM Valuation
        unrealized = 0
        if active_pos_dollars != 0:
            side = 1 if active_pos_dollars > 0 else -1
            # Current value relative to mid
            ret = (current_mid / entry_price) - 1 if side == 1 else 1 - (current_mid / entry_price)
            unrealized = abs(active_pos_dollars) * ret
        
        equity_history.append(current_cash + abs(active_pos_dollars) + unrealized)
        exposure_history.append(abs(active_pos_dollars))

    df['total_equity'] = equity_history
    df['exposure'] = exposure_history

    # --- DASHBOARD VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Price vs TV
    ax1.plot(df.index, df['tv'], label='Model Theoretical Value', color='#2196F3', lw=2.5)
    ax1.plot(df.index, df['mid'], label='Market Mid Price', color='#444444', alpha=0.5, ls='--')
    ax1.fill_between(df.index, 0, 1, where=(df['exposure'] > 0), color='green', alpha=0.15, label='In-Position')
    
    # Annotate Edge
    ax1.set_title(f"Quant MLB Trading Dashboard | {event}", loc='left', fontweight='bold', fontsize=14)
    ax1.set_ylabel("Probability / Price ($)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left', frameon=True)

    # Subplot 2: Equity and Exposure
    ax2.plot(df.index, df['total_equity'], color='#2E7D32', lw=2, label='Portfolio Value (MTM)')
    ax2.axhline(starting_bankroll, color='red', lw=1, ls=':', alpha=0.7, label='Initial Capital')
    ax2.set_ylabel("Account Value ($)")
    ax2.set_xlabel("Market Ticks (Chronological)")
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_anchored_game_dashboard()