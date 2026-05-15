import requests
import time
import pandas as pd
import duckdb
import base64
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization

# API Config
headers = {"KALSHI-ACCESS-KEY": "f561daf1-9b9e-49f1-8c68-c13abddfa9f9"}
base_url = "https://api.elections.kalshi.com/trade-api/v2/events"
market_url = "https://api.elections.kalshi.com/trade-api/v2/historical/markets"
trade_url = "https://api.elections.kalshi.com/trade-api/v2/historical/trades"
candlestick_url = "https://api.elections.kalshi.com/trade-api/v2/historical/markets/{ticker}/candlesticks"

# Filtering
cutoff_date = datetime(2025, 9, 28)
start_date = datetime(2025, 3, 27)

def fetch_market_hierarchy():
    cursor = ""
    iteration = 1
    # This will store a list of dicts: {'event_ticker': '...', 'market_ticker': '...'}
    hierarchy_data = []

    while True:
        params = {
            "limit": 200,
            "with_nested_markets": "true",
            "series_ticker": "KXMLBGAME",
            "cursor": cursor
        }

        print(f"--- Fetching Events Iteration {iteration} ---")
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code != 200:
            break

        data = response.json()
        events = data.get('events', [])
        if not events:
            break

        for event in events:
            curr_event = event.get('event_ticker')
            
            # Date filtering logic
            try:
                date_str = curr_event[10:17] 
                event_date = datetime.strptime(date_str, '%y%b%d')
            except:
                continue

            if curr_event.startswith('KXMLBGAME-25') and start_date <= event_date <= cutoff_date:
                time.sleep(0.1) 
                
                m_params = {"limit": 200, "event_ticker": curr_event}
                m_res = requests.get(market_url, params=m_params, headers=headers)
                m_data = m_res.json()
                
                if 'markets' in m_data:
                    for m in m_data['markets']:
                        hierarchy_data.append({
                            "event_ticker": curr_event,
                            "market_ticker": m.get('ticker')
                        })
                        print(f"Mapped {curr_event} -> {m.get('ticker')}")

        cursor = data.get('cursor', '')
        if not cursor:
            break
        iteration += 1

    return hierarchy_data

def fetch_and_store_all(hierarchy_data, db_name="kalshi_mlb.db"):
    con = duckdb.connect(db_name)
    
    # Clean setup
    con.execute("DROP TABLE IF EXISTS trades")
    con.execute("DROP TABLE IF EXISTS event_market_map")
    
    # Table 1: The Map
    con.execute("CREATE TABLE event_market_map (event_ticker VARCHAR, market_ticker VARCHAR)")
    map_df = pd.DataFrame(hierarchy_data)
    con.append('event_market_map', map_df)

    # Table 2: The Trades
    con.execute("""
        CREATE TABLE trades (
            ticker VARCHAR,
            trade_id VARCHAR,
            yes_price_dollars DOUBLE,
            no_price_dollars DOUBLE,
            count_fp DOUBLE,
            taker_side VARCHAR,
            created_time TIMESTAMP
        )
    """)

    # Get unique market tickers to avoid redundant API calls
    unique_markets = map_df['market_ticker'].unique()

    for ticker in unique_markets:
        print(f"📦 Fetching trades for: {ticker}")
        cursor = ""
        while True:
            params = {"limit": 1000, "ticker": ticker, "cursor": cursor}
            time.sleep(0.15) 
            res = requests.get(trade_url, params=params, headers=headers)
            
            if res.status_code == 429:
                time.sleep(2)
                continue
            if res.status_code != 200:
                break

            data = res.json()
            trades = data.get('trades', [])
            if not trades:
                break
                
            df = pd.DataFrame(trades)
            df['created_time'] = pd.to_datetime(df['created_time'], format='ISO8601', utc=True)
            df['yes_price_dollars'] = pd.to_numeric(df['yes_price_dollars']).astype(float)
            df['no_price_dollars'] = pd.to_numeric(df['no_price_dollars']).astype(float)
            df['count_fp'] = pd.to_numeric(df['count_fp']).astype(float)

            con.append('trades', df[['ticker', 'trade_id', 'yes_price_dollars', 'no_price_dollars', 'count_fp', 'taker_side', 'created_time']])
            
            cursor = data.get('cursor', '')
            if not cursor:
                break
    
    con.close()
    print("🏁 Done mapping and storing.")

def fetch_and_store_candlesticks(hierarchy_data, db_name="kalshi_mlb.db"):
    con = duckdb.connect(db_name)
    con.execute("DROP TABLE IF EXISTS candlesticks")
    
    # Expanded schema for Ask and Bid OHLC
    con.execute("""
        CREATE TABLE candlesticks (
            ticker VARCHAR,
            end_period_ts TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            ask_open DOUBLE, ask_high DOUBLE, ask_low DOUBLE, ask_close DOUBLE,
            bid_open DOUBLE, bid_high DOUBLE, bid_low DOUBLE, bid_close DOUBLE,
            volume INTEGER
        )
    """)

    # Helper to handle NoneType safely
    def get_val(obj, key):
        if obj is None: return None
        val = obj.get(key)
        return float(val) if val is not None else None

    unique_markets = pd.DataFrame(hierarchy_data)['market_ticker'].unique()

    for ticker in unique_markets:
        print(f"🕯️ Fetching full OHLC for: {ticker}")
        
        params = {
            "start_ts": int(start_date.timestamp()),
            "end_ts": int(cutoff_date.timestamp()),
            "period_interval": 1 
        }
        
        res = requests.get(candlestick_url.format(ticker=ticker), params=params, headers=headers)
        if res.status_code != 200: continue

        data = res.json()
        processed_candles = []
        
        for c in data.get('candlesticks', []):
            # Extract nested OHLC objects
            p = c.get('price')
            a = c.get('yes_ask')
            b = c.get('yes_bid')

            raw_volume = c.get('volume', 0)
            clean_volume = int(float(raw_volume)) if raw_volume is not None else 0

            processed_candles.append({
                "ticker": ticker,
                "end_period_ts": datetime.fromtimestamp(c['end_period_ts']),
                # Trade Price
                "open": get_val(p, 'open'),
                "high": get_val(p, 'high'),
                "low": get_val(p, 'low'),
                "close": get_val(p, 'close'),
                # Yes Ask
                "ask_open": get_val(a, 'open'),
                "ask_high": get_val(a, 'high'),
                "ask_low": get_val(a, 'low'),
                "ask_close": get_val(a, 'close'),
                # Yes Bid
                "bid_open": get_val(b, 'open'),
                "bid_high": get_val(b, 'high'),
                "bid_low": get_val(b, 'low'),
                "bid_close": get_val(b, 'close'),
                "volume": clean_volume
            })
            
        if processed_candles:
            con.append('candlesticks', pd.DataFrame(processed_candles))
        time.sleep(0.15)

    con.close()
    print("🏁 Done.")

def build_unified_market_table(hierarchy_data, db_name="kalshi_unified_market_data.db"):
    con = duckdb.connect(db_name)
    
    print("🛠️  Preparing unified_market_data table...")
    con.execute("DROP TABLE IF EXISTS unified_market_data")
    con.execute("""
        CREATE TABLE unified_market_data (
            timestamp TIMESTAMP,
            ticker VARCHAR,
            yes_price DOUBLE,
            no_price DOUBLE,
            volume DOUBLE,
            source VARCHAR
        )
    """)

    unique_markets = pd.DataFrame(hierarchy_data)['market_ticker'].unique()
    total = len(unique_markets)

    for i, ticker in enumerate(unique_markets, 1):
        print(f"🔄 Processing Market [{i}/{total}]: {ticker}")
        
        # We fetch the combined data using a query, but process it via DataFrame
        query = f"""
            SELECT 
                created_time AS timestamp,
                ticker,
                yes_price_dollars AS yes_price,
                no_price_dollars AS no_price,
                count_fp AS volume,
                'TRADE' AS source
            FROM trades
            WHERE ticker = '{ticker}'

            UNION ALL

            SELECT 
                end_period_ts AS timestamp,
                ticker,
                ask_open AS yes_price,
                (1.0 - bid_open) AS no_price,
                0 AS volume,
                'QUOTE' AS source
            FROM candlesticks c
            WHERE ticker = '{ticker}' 
              AND (volume = 0 OR volume IS NULL)
              AND NOT EXISTS (
                  SELECT 1 FROM trades t 
                  WHERE t.ticker = c.ticker 
                  AND t.created_time = c.end_period_ts
              )
            ORDER BY timestamp ASC
        """
        
        # Load this market's unified data into a temporary DataFrame
        market_df = con.execute(query).df()
        
        if not market_df.empty:
            # Append to the main table just like your candlestick function
            con.append('unified_market_data', market_df)
            print(f"   ✅ Appended {len(market_df)} rows")
        else:
            print(f"   ⏩ No data found for {ticker}")

    # Final cleanup and indexing
    print("📊 Finalizing database indexes...")
    con.execute("CREATE INDEX idx_unified_time ON unified_market_data (timestamp)")
    con.close()
    print("🏁 Unified high-precision dataset is complete.")

# Run
mapping = fetch_market_hierarchy()
fetch_and_store_all(mapping)
fetch_and_store_candlesticks(mapping)
build_unified_market_table(mapping)


