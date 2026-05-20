import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def run_quant_backtest_dashboard():
    # 1. Initialization
    db_path = 'mlb_model_training_cleaned.db'
    bundle_path = 'mlb_summer_quant_bundle.joblib'
    
    con = duckdb.connect(db_path, read_only=True)
    bundle = joblib.load(bundle_path)
    model, features = bundle['model'], bundle['features']
    
    # 2. Select Sample
    event = con.execute("SELECT DISTINCT event_ticker FROM training_data ORDER BY RANDOM() LIMIT 1").fetchone()[0]
    df = con.execute(f"SELECT * FROM training_data WHERE event_ticker = '{event}' ORDER BY timestamp ASC").df()
    
    # 3. Inference
    X = df[features].fillna(0).values
    df['tv'] = model.predict_proba(X)[:, 1]
    df['mid'] = (df['min_buy'] + df['max_sell']) / 2
    df['edge'] = df['tv'] - df['mid']

    # 4. Research-Grade Simulation Engine
    bankroll = 1000.0 
    entry_threshold, exit_threshold = 0.08, 0.02
    max_edge_filter = 0.30 
    max_price_jump = 0.15 
    kelly_fraction = 0.25 
    
    pos_contracts = 0.0
    cash = bankroll
    equity_curve = []
    state_history = []
    
    # Track previous state for smoothing
    prev_mid = df.iloc[0]['mid']
    prev_bid = df.iloc[0]['max_sell']
    prev_ask = df.iloc[0]['min_buy']
    
    for i, row in df.iterrows():
        # --- Jump Detection ---
        is_jump = abs(row['mid'] - prev_mid) > max_price_jump
        
        # --- Smooth MTM Logic ---
        # If it's a jump, use the previous 'known good' prices for MTM
        mtm_bid = prev_bid if is_jump else row['max_sell']
        mtm_ask = prev_ask if is_jump else row['min_buy']
        
        mtm_value = 0
        if pos_contracts > 0: mtm_value = pos_contracts * mtm_bid
        elif pos_contracts < 0: mtm_value = pos_contracts * mtm_ask
        equity_curve.append(cash + mtm_value)
        
        # --- Strategy Logic ---
        if not is_jump: # Only execute if not a jump
            # Exit
            if pos_contracts != 0:
                exit_signal = (pos_contracts > 0 and row['edge'] < exit_threshold) or \
                              (pos_contracts < 0 and row['edge'] > -exit_threshold) or \
                              (i == len(df) - 1)
                if exit_signal:
                    liquid_price = row['max_sell'] if pos_contracts > 0 else row['min_buy']
                    cash += (pos_contracts * liquid_price)
                    pos_contracts = 0
            
            # Entry
            elif entry_threshold < abs(row['edge']) <= max_edge_filter:
                side = 1 if row['edge'] > 0 else -1
                entry_price = row['min_buy'] if side == 1 else row['max_sell']
                
                # Smart Kelly Sizing
                if side == 1:
                    fraction = (row['edge'] / (1 - entry_price)) * kelly_fraction
                else:
                    fraction = (abs(row['edge']) / entry_price) * kelly_fraction
                fraction = np.clip(fraction, 0, 1.0)
                
                new_contracts = ((cash * fraction) / entry_price) * side
                cash -= (new_contracts * entry_price)
                pos_contracts = new_contracts
        
        # Track state and update "previous" prices
        state_history.append(1 if pos_contracts > 0 else (-1 if pos_contracts < 0 else 0))
        prev_mid, prev_bid, prev_ask = row['mid'], row['max_sell'], row['min_buy']

    df['equity'] = equity_curve
    df['state'] = state_history
    
    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(df.index, df['tv'], label='TV', color='#2196F3', lw=2)
    ax1.plot(df.index, df['mid'], label='Mid Price', color='black', alpha=0.4, ls='--')
    ax1.plot(df.index, df['min_buy'], label='Min Buy (Ask)', color='red', alpha=0.3, lw=1)
    ax1.plot(df.index, df['max_sell'], label='Max Sell (Bid)', color='green', alpha=0.3, lw=1)
    ax1.fill_between(df.index, 0, 1, where=(df['state'] == 1), color='green', alpha=0.1)
    ax1.fill_between(df.index, 0, 1, where=(df['state'] == -1), color='red', alpha=0.1)
    
    ax1.set_title(f"Market Structure & Strategy State (Jump Smoothed: {max_price_jump}) | {event}")
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Price")

    ax2.plot(df.index, df['equity'], color='green', lw=2, label='Portfolio PnL')
    ax2.axhline(bankroll, color='black', ls=':', alpha=0.5)
    ax2.set_title("Portfolio PnL (Smooth MTM)")
    ax2.set_ylabel("Account Value ($)")
    ax2.set_xlabel("Time Ticks")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_quant_backtest_dashboard()