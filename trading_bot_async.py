import asyncio
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import requests
import json
import time
import os
import logging
import logging.handlers
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from okx_ws import OKXWebSocket

# ==================== 1. 日志配置 ====================
def setup_trading_logs(log_dir="/root/timesfm_lab", max_bytes=10*1024*1024, backup_count=5):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console)

    trade_log = logging.getLogger("TRADE")
    trade_log.setLevel(logging.INFO)
    trade_h = logging.handlers.RotatingFileHandler(f"{log_dir}/trades.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    trade_h.setFormatter(fmt)
    trade_log.addHandler(trade_h)
    trade_log.propagate = False

    signal_log = logging.getLogger("SIGNAL")
    signal_log.setLevel(logging.DEBUG)
    signal_h = logging.handlers.RotatingFileHandler(f"{log_dir}/signals.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    signal_h.setFormatter(fmt)
    signal_log.addHandler(signal_h)
    signal_log.propagate = False

    sys_log = logging.getLogger("SYSTEM")
    sys_log.setLevel(logging.WARNING)
    sys_h = logging.handlers.RotatingFileHandler(f"{log_dir}/system.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    sys_h.setFormatter(fmt)
    sys_log.addHandler(sys_h)
    sys_log.propagate = False

    return {"trade": trade_log, "signal": signal_log, "system": sys_log}

log_dir = "/root/timesfm_lab"
loggers = setup_trading_logs(log_dir)
log_trade = loggers["trade"].info
log_signal = loggers["signal"].info
log_system_warn = loggers["system"].warning
log_system_err = loggers["system"].error
log_system_exception = loggers["system"].exception

def log(msg):
    if any(k in msg for k in ["✅ 开仓", "🔻 平仓", "盈利", "亏损", "保证金", "止损", "止盈"]):
        log_trade(msg)
    elif any(k in msg for k in ["📉 检查持仓", "触发跟踪止损", "触发初始止损", "趋势反转平仓"]):
        log_trade(msg)
    else:
        log_signal(msg)

def err(msg):
    log_system_err(msg)

# ==================== 2. 核心参数 ====================
TG_BOT_TOKEN = "8722422674:AAGrKmRurQ2G__j-Vxbh5451v0e9_u97CQY"
TG_CHAT_ID = "5372217316"
TG_PROXIES = None

BAR = "5m"
HIGHER_BAR = "15m"
LIMIT = 200                     # 获取200根，但检查至少100根
TOP_N = 50
FINAL_PICK_N = 3
MIN_DIRECTION_CONFIDENCE = 0.70

RSI_PERIOD = 7
MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 3
RSI_LONG_THRESHOLD = 45
RSI_SHORT_THRESHOLD = 75

OUTPUT_FILE = f"{log_dir}/signals_vps.json"
REPORT_FILE = f"{log_dir}/signals_vps_report.json"
STRATEGY_POSITIONS_FILE = f"{log_dir}/strategy_positions.json"

API_KEY = "10d14cf0-79da-4597-9456-3aa1b88e1acf"
API_SECRET = "1B6A940855EC5787CD4E56BEF6D94733"
API_PASS = "kP9!vR2@mN5+"
IS_SANDBOX = False

PREDICTION_INTERVAL = 60
MIN_BALANCE_USDT = 10.0

MIN_VOLUME_USDT = 10_000_000
MIN_MARKET_CAP_USDT = 20_000_000
VOLATILITY_SAMPLE_SIZE = 200

LEVERAGE = 3
USE_PERCENT_OF_AVAILABLE = 0.5
MAX_CONCURRENT_POSITIONS = 3
MAX_TOTAL_MARGIN_RATIO = 1.0

PRICE_POSITION_RATIO = 0.1
FAVORABLE_MOVE_PCT = 0.2

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 2.0

VOLUME_SPIKE_RATIO = 2.5
MIN_ATR_VALUE = 0.0005
EMERGENCY_MOVE_PCT = 1.5

MAX_DEVIATION_FROM_EMA_PCT = 2.5
MAX_CANDLE_BODY_RATIO = 3.0
MAX_VOLUME_SPIKE_RATIO = 3.0
SAFETY_TREND_BAR = "1h"

ENABLE_VOLUME_ANOMALY = True
VOLUME_ANOMALY_THRESHOLD = 5.0
ENABLE_TREND_REVERSAL = True
ENABLE_PRICE_MOMENTUM_FILTER = True
MOMENTUM_LIMIT_PCT = 1.5

ENABLE_NEWS_MONITOR = False

# 信号有效期（秒）
SIGNAL_VALID_SECONDS = 30
# 最大持仓时长（秒）
MAX_HOLD_SECONDS = 90

# ==================== 3. Telegram推送 ====================
def push_telegram(content):
    if not TG_BOT_TOKEN: return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": content, "disable_web_page_preview": True}
    try:
        res = requests.post(url, json=payload, timeout=8)
        return res.json().get("ok")
    except:
        return False

# ==================== 4. 数据获取函数 ====================
def get_all_swap_contracts():
    try:
        url = "https://www.okx.com/api/v5/public/instruments"
        params = {"instType": "SWAP"}
        data = requests.get(url, params=params, timeout=10).json()["data"]
        symbols = []
        for item in data:
            if item["settleCcy"] == "USDT" and item["state"] == "live":
                symbols.append(item["instId"])
        return symbols
    except Exception as e:
        err(f"获取合约列表失败: {e}")
        return []

def fetch_klines_with_retry(instId, bar, limit, max_retries=3):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": instId, "bar": bar, "limit": str(limit)}
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data.get("code") == "0" and data.get("data"):
                df = pd.DataFrame(data["data"], columns=["ts", "o", "h", "l", "c", "v", "vc", "cv", "confirm"])
                df["c"] = df["c"].astype(float)
                df["h"] = df["h"].astype(float)
                df["l"] = df["l"].astype(float)
                df["o"] = df["o"].astype(float)
                df["v"] = df["v"].astype(float)
                df["ts"] = df["ts"].astype(int)
                df = df.sort_values("ts").reset_index(drop=True)
                return df
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                err(f"获取K线失败 {instId}: {e}")
                return None
    return None

def fetch_previous_candle(instId):
    df = fetch_klines_with_retry(instId, BAR, 2)
    if df is None or len(df) < 2:
        return None
    prev = df.iloc[-2]
    return prev['o'], prev['h'], prev['l'], prev['c']

def fetch_volume_usdt(instId):
    try:
        url = "https://www.okx.com/api/v5/market/ticker"
        params = {"instId": instId}
        res = requests.get(url, params=params, timeout=5).json()
        if res.get("code") == "0" and res.get("data"):
            vol_usdt = float(res["data"][0].get("volCcy24h", 0))
            return vol_usdt
        return 0.0
    except:
        return 0.0

def fetch_market_cap(instId):
    try:
        base = instId.split('-')[0]
        search_url = f"https://api.coingecko.com/api/v3/search?query={base}"
        resp = requests.get(search_url, timeout=5).json()
        if resp.get('coins'):
            coin_id = resp['coins'][0]['id']
            coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            coin_data = requests.get(coin_url, timeout=5).json()
            market_cap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
            return market_cap
        return 0
    except:
        return 0

def calculate_volatility(prices):
    if len(prices) < 30:
        return 0
    ret = np.diff(prices) / prices[:-1]
    return np.std(ret) * 1000

# ==================== 5. 技术指标计算 ====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_prev = histogram.iloc[-2] if len(histogram) >= 2 else histogram.iloc[-1]
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1], hist_prev

def get_15min_trend(symbol):
    try:
        df = fetch_klines_with_retry(symbol, HIGHER_BAR, 100)
        if df is None or len(df) < 30:
            return None, None, False, False
        closes = df['c']
        ema20 = closes.ewm(span=20, adjust=False).mean()
        prev_ema20 = ema20.iloc[-2]
        curr_ema20 = ema20.iloc[-1]
        slope = (curr_ema20 - prev_ema20) / prev_ema20 if prev_ema20 != 0 else 0
        is_downtrend = slope < -0.001
        is_uptrend = slope > 0.001
        return curr_ema20, slope, is_downtrend, is_uptrend
    except Exception as e:
        err(f"获取15分钟趋势失败 {symbol}: {e}")
        return None, None, False, False

def get_1h_trend(symbol):
    try:
        df = fetch_klines_with_retry(symbol, "1h", 100)
        if df is None or len(df) < 30:
            return None, None, False, False
        closes = df['c']
        ema20 = closes.ewm(span=20, adjust=False).mean()
        prev_ema20 = ema20.iloc[-2]
        curr_ema20 = ema20.iloc[-1]
        slope = (curr_ema20 - prev_ema20) / prev_ema20 if prev_ema20 != 0 else 0
        is_downtrend = slope < -0.001
        is_uptrend = slope > 0.001
        return curr_ema20, slope, is_downtrend, is_uptrend
    except Exception as e:
        err(f"获取1小时趋势失败 {symbol}: {e}")
        return None, None, False, False

def check_momentum_surge(symbol, current_price, side):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 30)
        if df is None or len(df) < 25:
            return False, 1.0
        volumes = df['v'].iloc[-21:-1].astype(float)
        bodies = (df['c'] - df['o']).abs().iloc[-21:-1]
        avg_volume = volumes.mean()
        avg_body = bodies.mean()
        current_volume = float(df['v'].iloc[-1])
        current_body = abs(float(df['c'].iloc[-1]) - float(df['o'].iloc[-1]))
        prev_close = float(df['c'].iloc[-2])
        price_change_pct = (current_price - prev_close) / prev_close * 100
        if current_volume > avg_volume * VOLUME_SPIKE_RATIO and current_body > avg_body * 1.5:
            if (side == 'long' and price_change_pct > 0) or (side == 'short' and price_change_pct < 0):
                vol_ratio = min(current_volume / avg_volume, 10) / 3.33
                body_ratio = min(current_body / avg_body, 5) / 1.67
                factor = max(1.0, (vol_ratio + body_ratio) / 2)
                return True, min(factor, 3.0)
        return False, 1.0
    except Exception as e:
        log(f"动能突变检测异常 {symbol}: {e}")
        return False, 1.0

def check_emergency_move(symbol, current_price):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 2)
        if df is None or len(df) < 2:
            return False, 0.0
        prev_close = float(df['c'].iloc[-2])
        move_pct = abs((current_price - prev_close) / prev_close * 100)
        return move_pct >= EMERGENCY_MOVE_PCT, move_pct
    except:
        return False, 0.0

def check_technical_indicators(symbol, side, current_price, atr):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 100)
        if df is None or len(df) < 60:
            return True, f"数据不足，跳过指标检查 (当前价格: {current_price:.6f})", 1.0
        closes = df['c']
        rsi = compute_rsi(closes, RSI_PERIOD)
        macd_line, signal_line, histogram, hist_prev = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        MACD_HIST_EPSILON = 0.0005
        side_cn = "多单" if side == 'long' else "空单"
        is_surge, surge_factor = check_momentum_surge(symbol, current_price, side)
        emergency, move_pct = check_emergency_move(symbol, current_price)
        if emergency:
            log(f"🚨 紧急动能信号: {symbol} {side_cn} 价格变动 {move_pct:.1f}%")
            return True, f"紧急动能信号 (变动 {move_pct:.1f}%)", surge_factor
        if side == 'long':
            if rsi >= RSI_LONG_THRESHOLD and not is_surge:
                return False, f"{side_cn} RSI={rsi:.1f} ≥ {RSI_LONG_THRESHOLD}，不符合多单条件", 1.0
        else:
            if rsi <= RSI_SHORT_THRESHOLD and not is_surge:
                return False, f"{side_cn} RSI={rsi:.1f} ≤ {RSI_SHORT_THRESHOLD}，不符合空单条件", 1.0
        if side == 'long':
            if histogram <= -MACD_HIST_EPSILON and not is_surge:
                return False, f"{side_cn} MACD柱状线={histogram:.4f} ≤ -{MACD_HIST_EPSILON}，动能过负", 1.0
        else:
            if histogram >= MACD_HIST_EPSILON and not is_surge:
                return False, f"{side_cn} MACD柱状线={histogram:.4f} ≥ {MACD_HIST_EPSILON}，动能过正", 1.0
        if side == 'long':
            if macd_line <= 0 or signal_line <= 0:
                if not is_surge:
                    return False, f"{side_cn} 快慢线不在零轴上方 (MACD={macd_line:.4f}, Signal={signal_line:.4f})", 1.0
        else:
            if macd_line >= 0 or signal_line >= 0:
                if not is_surge:
                    return False, f"{side_cn} 快慢线不在零轴下方 (MACD={macd_line:.4f}, Signal={signal_line:.4f})", 1.0
        df_higher = fetch_klines_with_retry(symbol, HIGHER_BAR, 100)
        if df_higher is not None and len(df_higher) >= 30:
            closes_higher = df_higher['c']
            macd_higher, signal_higher, _, _ = compute_macd(closes_higher, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            if side == 'long':
                if macd_higher <= signal_higher and not is_surge:
                    return False, f"{side_cn} 15分钟MACD死叉 (MACD={macd_higher:.4f} ≤ Signal={signal_higher:.4f})，方向不符", 1.0
            else:
                if macd_higher >= signal_higher and not is_surge:
                    return False, f"{side_cn} 15分钟MACD金叉 (MACD={macd_higher:.4f} ≥ Signal={signal_higher:.4f})，方向不符", 1.0
        if atr is not None and atr < MIN_ATR_VALUE and not is_surge and not emergency:
            return False, f"ATR={atr:.6f} < {MIN_ATR_VALUE}，波动率过低", 1.0
        desc = f"RSI={rsi:.1f}, MACD={macd_line:.4f}, Signal={signal_line:.4f}, Hist={histogram:.4f}"
        return True, f"技术指标通过: {desc}", surge_factor
    except Exception as e:
        err(f"技术指标计算异常 {symbol}: {e}")
        return True, f"指标计算异常，跳过检查", 1.0

# ==================== 6. 波动率自适应参数 ====================
def detect_volatility_profile(df, price_col='c', period=14):
    try:
        high_low = df['h'] - df['l']
        high_close = np.abs(df['h'] - df['c'].shift())
        low_close = np.abs(df['l'] - df['c'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        atr_pct = (atr / df[price_col]) * 100
        current_atr_pct = atr_pct.iloc[-1] if not atr_pct.empty else 0.5
        if current_atr_pct > 1.5:
            return "EXTREME", current_atr_pct
        elif current_atr_pct > 0.6:
            return "HIGH", current_atr_pct
        else:
            return "NORMAL", current_atr_pct
    except Exception as e:
        log(f"波动率检测异常: {e}")
        return "NORMAL", 0.5

def get_adaptive_trading_params(bar_frame, volatility_profile):
    params = {
        "atr_multiplier": 2.0,
        "stop_loss_pct": 1.5,
        "trailing_stop_pct": 1.0
    }
    if bar_frame == "5m":
        if volatility_profile == "EXTREME":
            params.update({
                "atr_multiplier": 3.5,
                "stop_loss_pct": 4.0,
                "trailing_stop_pct": 0.5
            })
        elif volatility_profile == "HIGH":
            params.update({
                "atr_multiplier": 2.5,
                "stop_loss_pct": 2.5,
                "trailing_stop_pct": 0.8
            })
        else:
            params.update({
                "atr_multiplier": 2.0,
                "stop_loss_pct": 1.5,
                "trailing_stop_pct": 1.0
            })
    elif bar_frame == "3m":
        if volatility_profile == "EXTREME":
            params.update({
                "atr_multiplier": 3.5,
                "stop_loss_pct": 4.0,
                "trailing_stop_pct": 0.5
            })
        elif volatility_profile == "HIGH":
            params.update({
                "atr_multiplier": 2.5,
                "stop_loss_pct": 2.5,
                "trailing_stop_pct": 0.8
            })
        else:
            params.update({
                "atr_multiplier": 2.0,
                "stop_loss_pct": 1.5,
                "trailing_stop_pct": 1.0
            })
    elif bar_frame == "1m":
        if volatility_profile == "EXTREME":
            params.update({
                "atr_multiplier": 4.5,
                "stop_loss_pct": 6.0,
                "trailing_stop_pct": 0.3
            })
        elif volatility_profile == "HIGH":
            params.update({
                "atr_multiplier": 3.0,
                "stop_loss_pct": 3.0,
                "trailing_stop_pct": 0.4
            })
        else:
            params.update({
                "atr_multiplier": 2.2,
                "stop_loss_pct": 1.0,
                "trailing_stop_pct": 0.2
            })
    elif bar_frame == "15m":
        params.update({
            "atr_multiplier": 1.8,
            "stop_loss_pct": 1.2,
            "trailing_stop_pct": 2.0
        })
    return params

# ==================== 7. 纯技术指标信号评分 ====================
def compute_technical_score(symbol, current_price, df_5m):
    try:
        rsi = compute_rsi(df_5m['c'], RSI_PERIOD)
        macd_line, signal_line, hist, hist_prev = compute_macd(df_5m['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        ema20 = df_5m['c'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df_5m['c'].ewm(span=50, adjust=False).mean().iloc[-1] if len(df_5m) >= 50 else ema20
        
        bb_upper, bb_lower = calculate_bollinger_bands(symbol)
        if bb_upper is None:
            bb_upper = ema20 * 1.02
            bb_lower = ema20 * 0.98
        
        _, _, is_15m_downtrend, is_15m_uptrend = get_15min_trend(symbol)
        _, _, is_1h_downtrend, is_1h_uptrend = get_1h_trend(symbol)
        
        long_score = 0
        short_score = 0
        
        if rsi < 35:
            long_score += 30
        elif rsi > 65:
            short_score += 30
        
        if hist > 0:
            long_score += min(30, hist * 10000)
        elif hist < 0:
            short_score += min(30, abs(hist) * 10000)
        
        if current_price > ema20:
            long_score += 20
        elif current_price < ema20:
            short_score += 20
        
        if ema20 > ema50:
            long_score += 20
        else:
            short_score += 20
        
        if current_price < bb_lower:
            long_score += 20
        elif current_price > bb_upper:
            short_score += 20
        
        if is_15m_uptrend or is_1h_uptrend:
            long_score += 15
        if is_15m_downtrend or is_1h_downtrend:
            short_score += 15
        
        long_score = min(100, long_score)
        short_score = min(100, short_score)
        return long_score, short_score, None
    except Exception as e:
        err(f"技术评分异常 {symbol}: {e}")
        return 0, 0, None

def calculate_bollinger_bands(symbol, period=20, std=2):
    try:
        df = fetch_klines_with_retry(symbol, BAR, period+1)
        if df is None or len(df) < period:
            return None, None
        closes = df['c']
        sma = closes.rolling(window=period).mean().iloc[-1]
        std_dev = closes.rolling(window=period).std().iloc[-1]
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, lower
    except Exception as e:
        err(f"计算布林带失败 {symbol}: {e}")
        return None, None

def get_adx(symbol, period=14):
    try:
        df = fetch_klines_with_retry(symbol, BAR, period+20)
        if df is None or len(df) < period+10:
            return None
        high = df['h'].values
        low = df['l'].values
        close = df['c'].values
        plus_dm = np.zeros(len(high)-1)
        minus_dm = np.zeros(len(high)-1)
        tr = np.zeros(len(high)-1)
        for i in range(1, len(high)):
            move_up = high[i] - high[i-1]
            move_down = low[i-1] - low[i]
            plus_dm[i-1] = move_up if move_up > move_down and move_up > 0 else 0
            minus_dm[i-1] = move_down if move_down > move_up and move_down > 0 else 0
            tr[i-1] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.mean(tr[-period:])
        if atr == 0:
            return 0
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
        return dx
    except Exception as e:
        err(f"计算ADX失败 {symbol}: {e}")
        return None

def get_atr_percent(symbol, period=ATR_PERIOD):
    try:
        df = fetch_klines_with_retry(symbol, BAR, period+5)
        if df is None or len(df) < period+1:
            return None
        high = df['h'].values
        low = df['l'].values
        close = df['c'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        atr = np.mean(tr[-period:])
        atr_pct = atr / close[-1] * 100
        return atr_pct
    except Exception as e:
        err(f"计算ATR百分比失败 {symbol}: {e}")
        return None

def check_volume_anomaly(symbol, current_price):
    if not ENABLE_VOLUME_ANOMALY:
        return False, 1.0, ""
    try:
        df = fetch_klines_with_retry(symbol, BAR, 10)
        if df is None or len(df) < 6:
            return False, 1.0, "数据不足"
        volumes = df['v'].astype(float)
        avg_vol = volumes.iloc[-6:-1].mean()
        current_vol = volumes.iloc[-1]
        if avg_vol == 0:
            return False, 1.0, "平均成交量为0"
        ratio = current_vol / avg_vol
        if ratio > VOLUME_ANOMALY_THRESHOLD:
            closes = df['c'].astype(float)
            price_change = abs((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)
            if price_change > 1.0:
                return True, ratio, f"成交量突增{ratio:.1f}倍，价格波动{price_change:.1f}%"
        return False, ratio, ""
    except Exception as e:
        log(f"成交异动检测异常 {symbol}: {e}")
        return False, 1.0, ""

def check_trend_reversal(symbol, side, current_price, ema20_15m, slope_15m):
    if not ENABLE_TREND_REVERSAL:
        return False, ""
    df_15m = fetch_klines_with_retry(symbol, HIGHER_BAR, 10)
    if df_15m is None or len(df_15m) < 6:
        return False, "数据不足"
    closes = df_15m['c'].astype(float)
    highs = df_15m['h'].astype(float)
    lows = df_15m['l'].astype(float)
    opens = df_15m['o'].astype(float)
    volumes = df_15m['v'].astype(float)
    
    recent_high = highs.iloc[-6:-1].max()
    recent_low = lows.iloc[-6:-1].min()
    last_candle_bullish = closes.iloc[-1] > opens.iloc[-1]
    last_candle_bearish = closes.iloc[-1] < opens.iloc[-1]
    avg_volume = volumes.iloc[-6:-1].mean()
    current_volume = volumes.iloc[-1]
    volume_surge = current_volume > avg_volume * 1.5
    
    macd_line, signal_line, _, _ = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_golden_cross = macd_line > signal_line and (macd_line - signal_line) > 0.0001
    macd_death_cross = macd_line < signal_line and (signal_line - macd_line) > 0.0001
    
    if side == 'long':
        rise_from_low = (current_price - recent_low) / recent_low * 100 if recent_low > 0 else 0
        if rise_from_low > 1.5 and last_candle_bullish and volume_surge:
            return True, f"底部反转：反弹{rise_from_low:.1f}%，放量阳线"
        if macd_golden_cross and slope_15m is not None and slope_15m > -0.0005:
            return True, f"MACD金叉，趋势转多"
        try:
            df_rsi = fetch_klines_with_retry(symbol, BAR, 20)
            if df_rsi is not None:
                rsi = compute_rsi(df_rsi['c'], RSI_PERIOD)
                if rsi < 35 and rsi > 30:
                    return True, f"RSI从超卖区回升"
        except:
            pass
        return False, "无底部反转信号"
    else:
        drop_from_high = (recent_high - current_price) / recent_high * 100 if recent_high > 0 else 0
        if drop_from_high > 1.5 and last_candle_bearish and volume_surge:
            return True, f"顶部反转：回落{drop_from_high:.1f}%，放量阴线"
        if macd_death_cross and slope_15m is not None and slope_15m < 0.0005:
            return True, f"MACD死叉，趋势转空"
        try:
            df_rsi = fetch_klines_with_retry(symbol, BAR, 20)
            if df_rsi is not None:
                rsi = compute_rsi(df_rsi['c'], RSI_PERIOD)
                if rsi > 65 and rsi < 70:
                    return True, f"RSI从超买区回落"
        except:
            pass
        return False, "无顶部反转信号"

def check_price_momentum_filter(symbol, side, current_price):
    if not ENABLE_PRICE_MOMENTUM_FILTER:
        return True, ""
    try:
        df = fetch_klines_with_retry(symbol, BAR, 5)
        if df is None or len(df) < 4:
            return True, "数据不足"
        closes = df['c'].astype(float)
        start_price = closes.iloc[-4]
        end_price = closes.iloc[-1]
        total_ret = (end_price - start_price) / start_price * 100
        if side == 'long' and total_ret > MOMENTUM_LIMIT_PCT:
            return False, f"最近3根K线已累计上涨{total_ret:.2f}% > {MOMENTUM_LIMIT_PCT}%，追高风险大"
        if side == 'short' and total_ret < -MOMENTUM_LIMIT_PCT:
            return False, f"最近3根K线已累计下跌{abs(total_ret):.2f}% > {MOMENTUM_LIMIT_PCT}%，追跌风险大"
        return True, ""
    except Exception as e:
        log(f"动量过滤异常 {symbol}: {e}")
        return True, ""

def predict_and_score(instId):
    try:
        df = fetch_klines_with_retry(instId, BAR, LIMIT)
        if df is None or len(df) < 100:                     # 要求至少100根5分钟K线
            return None, f"数据不足: 只有 {len(df) if df is not None else 0} 根K线，需要100根"
        current_price = float(df['c'].iloc[-1])
        
        long_score, short_score, _ = compute_technical_score(instId, current_price, df)
        
        _, _, is_15m_downtrend, is_15m_uptrend = get_15min_trend(instId)
        _, _, is_1h_downtrend, is_1h_uptrend = get_1h_trend(instId)
        
        adx = get_adx(instId)
        atr_pct = get_atr_percent(instId)
        rsi_val = compute_rsi(df['c'], RSI_PERIOD)
        macd_line, signal_line, hist, hist_prev = compute_macd(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        
        # 确定最佳方向
        best_side = None
        if long_score > short_score + 15:
            best_side = 'long'
        elif short_score > long_score + 15:
            best_side = 'short'
        
        if best_side is None:
            return None, f"多空评分差距不足 (多:{long_score:.1f}, 空:{short_score:.1f})"
        
        # 计算该方向的置信度（使用分数占总分的比例）
        total = long_score + short_score
        if total > 0:
            side_confidence = long_score / total if best_side == 'long' else short_score / total
        else:
            side_confidence = 0.5
        
        if side_confidence < MIN_DIRECTION_CONFIDENCE:
            return None, f"{best_side.upper()}方向置信度不足: {side_confidence:.2f} < {MIN_DIRECTION_CONFIDENCE} (多:{long_score:.1f},空:{short_score:.1f})"
        
        # 技术指标否决
        if best_side == 'long':
            if rsi_val > 65:
                return None, f"RSI={rsi_val:.1f} 超买区，禁止追多"
            if hist <= -0.0002:
                return None, f"MACD柱状线为负({hist:.4f})，动能不足，拒绝做多"
        else:
            if rsi_val < 35:
                return None, f"RSI={rsi_val:.1f} 超卖区，禁止追空"
            if hist >= 0.0002:
                return None, f"MACD柱状线为正({hist:.4f})，动能向上，拒绝做空"
        
        anomaly, vol_ratio_anom, anom_reason = check_volume_anomaly(instId, current_price)
        if anomaly:
            return None, f"成交异动拒绝: {anom_reason}"
        
        momentum_ok, momentum_reason = check_price_momentum_filter(instId, best_side, current_price)
        if not momentum_ok:
            return None, f"动量过滤: {momentum_reason}"
        
        # 构建价格位置信息
        candle = fetch_previous_candle(instId)
        if candle:
            open_p, high, low, close = candle
            body_top = max(open_p, close)
            body_bottom = min(open_p, close)
            body_len = body_top - body_bottom
            if body_len > 0:
                is_with_trend = (best_side == 'long' and (is_15m_uptrend or is_1h_uptrend)) or \
                                (best_side == 'short' and (is_15m_downtrend or is_1h_downtrend))
                if best_side == 'long':
                    if is_with_trend:
                        long_entry_max = body_top
                        entry_desc = f"顺势多单，建议当前价 ≤ {body_top:.6f} (实体顶部)"
                    else:
                        long_entry_max = body_bottom + body_len * PRICE_POSITION_RATIO
                        entry_desc = f"逆势多单，建议入场价 ≤ {long_entry_max:.6f}"
                    short_entry_min = None
                else:
                    if is_with_trend:
                        short_entry_min = body_bottom
                        entry_desc = f"顺势空单，建议当前价 ≥ {body_bottom:.6f} (实体底部)"
                    else:
                        short_entry_min = body_top - body_len * PRICE_POSITION_RATIO
                        entry_desc = f"逆势空单，建议入场价 ≥ {short_entry_min:.6f}"
                    long_entry_max = None
                price_info = {
                    'current_price': current_price,
                    'body_top': body_top,
                    'body_bottom': body_bottom,
                    'long_entry_max': long_entry_max if best_side == 'long' else None,
                    'short_entry_min': short_entry_min if best_side == 'short' else None,
                    'entry_desc': entry_desc,
                    'is_with_trend': is_with_trend
                }
            else:
                price_info = None
        else:
            price_info = None
        
        hold_minutes = MAX_HOLD_SECONDS / 60.0   # 90秒 = 1.5分钟
        expected_return = 0.003 if best_side == 'long' else -0.003
        
        result = {
            "symbol": instId,
            "signal": best_side,
            "expected_return": expected_return,
            "r_squared": 0.5,
            "consistency": 0.7,
            "direction_confidence": side_confidence,
            "score": max(long_score, short_score),
            "last_price": current_price,
            "price_info": price_info,
            "tech_msg": f"技术指标评分: 多{long_score:.1f}/空{short_score:.1f}，选择{best_side}",
            "long_score": long_score,
            "short_score": short_score,
            "adx": adx if adx else 0,
            "atr_pct": atr_pct if atr_pct else 0,
            "rsi": rsi_val,
            "estimated_hold_minutes": hold_minutes
        }
        return result, ""
    
    except Exception as e:
        return None, f"异常: {str(e)[:50]}"

# ==================== 8. 预测循环 ====================
def run_prediction_cycle():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n============================================================")
    log(f"🔄 [{now_str}] {BAR}周期 | 纯技术指标信号 | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮 | 要求至少100根K线")
    log(f"============================================================")

    symbols = get_all_swap_contracts()
    if not symbols:
        push_telegram("❌ 获取合约失败")
        return {}
    log(f"✅ 合约总数：{len(symbols)}")

    vol_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sym = {executor.submit(lambda s: fetch_klines_with_retry(s, BAR, 80), s): s for s in symbols}
        for future in as_completed(future_to_sym):
            s = future_to_sym[future]
            try:
                df = future.result()
                if df is not None and len(df) >= 30:
                    p = df['c'].values.astype(np.float32)
                    vol = calculate_volatility(p)
                    vol_list.append({"symbol": s, "vol": vol})
            except:
                continue
    if not vol_list:
        log("❌ 无法获取任何合约的K线数据")
        return {}
    df_vol = pd.DataFrame(vol_list).sort_values("vol", ascending=False).head(VOLATILITY_SAMPLE_SIZE)
    high_vol_symbols = df_vol["symbol"].tolist()
    log(f"📊 波动率前{VOLATILITY_SAMPLE_SIZE}名: {len(high_vol_symbols)} 个")

    filtered = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sym = {}
        for s in high_vol_symbols:
            future_vol_usdt = executor.submit(fetch_volume_usdt, s)
            future_mcap = executor.submit(fetch_market_cap, s)
            future_to_sym[s] = (future_vol_usdt, future_mcap)
        for s, (f_vol, f_mcap) in future_to_sym.items():
            try:
                vol_usdt = f_vol.result()
                mcap = f_mcap.result()
                if vol_usdt >= MIN_VOLUME_USDT and mcap >= MIN_MARKET_CAP_USDT:
                    filtered.append(s)
            except:
                continue

    log(f"💰 成交额 ≥ {MIN_VOLUME_USDT/1_000_000:.0f}M USDT 且 市值 ≥ {MIN_MARKET_CAP_USDT/1_000_000:.0f}M USDT 的合约: {len(filtered)} 个")
    if not filtered:
        log("❌ 没有满足流动性和市值要求的合约")
        push_telegram("❌ 市场流动性或市值不足，无候选合约")
        return {}

    filtered_vol = [x for x in vol_list if x["symbol"] in filtered]
    df_filtered = pd.DataFrame(filtered_vol).sort_values("vol", ascending=False).head(TOP_N)
    candidates = df_filtered["symbol"].tolist()
    log(f"🎯 最终候选池（波动率Top{TOP_N}）: {len(candidates)} 个")
    log("📈 开始技术指标评分...")
    log("-" * 80)

    valid = []
    candidate_details = []

    for s in candidates:
        res, reason = predict_and_score(s)
        if res is not None:
            valid.append(res)
            candidate_details.append(res)
            log(f"  [{len(valid)}] {res['symbol']}")
            log(f"      预期涨跌: {res['expected_return']*100:+.2f}%")
            log(f"      方向置信度: {res['direction_confidence']:.2f} | 得分: {res['score']:.4f}")
            log(f"      技术指标: {res['tech_msg']}")
            log(f"      预估持仓时间: {res.get('estimated_hold_minutes', 1.5)} 分钟")
        else:
            long_score = 0.0
            short_score = 0.0
            # 尝试从拒绝原因中提取评分
            match_long = re.search(r'多:([\d.]+)', reason)
            match_short = re.search(r'空:([\d.]+)', reason)
            if match_long:
                long_score = float(match_long.group(1))
            if match_short:
                short_score = float(match_short.group(1))
            
            try:
                adx = get_adx(s)
                if adx is None: adx = 0
            except:
                adx = 0
            try:
                atr_pct = get_atr_percent(s)
                if atr_pct is None: atr_pct = 0
            except:
                atr_pct = 0
            try:
                df_rsi = fetch_klines_with_retry(s, BAR, 20)
                rsi = compute_rsi(df_rsi['c'], RSI_PERIOD) if df_rsi is not None else 50
            except:
                rsi = 50

            candidate_details.append({
                "symbol": s,
                "long_score": long_score,
                "short_score": short_score,
                "adx": adx,
                "atr_pct": atr_pct,
                "rsi": rsi,
                "reject_reason": reason,
                "signal": None
            })

    detail_lines = []
    detail_lines.append("🎯 最终候选池（波动率Top{}，含技术指标评分）：".format(TOP_N))
    for idx, cd in enumerate(candidate_details, 1):
        signal_flag = " ✅ SHORT信号" if cd.get('signal') == 'short' else (" ✅ LONG信号" if cd.get('signal') == 'long' else "")
        reject_msg = f" | 拒绝: {cd.get('reject_reason', '')}" if cd.get('reject_reason') else ""
        detail_lines.append(
            f"{idx}. {cd['symbol']} | 多:{cd['long_score']:.1f} / 空:{cd['short_score']:.1f} | "
            f"ADX:{cd['adx']:.1f} | ATR%:{cd['atr_pct']:.2f}% | RSI:{cd['rsi']:.1f}{signal_flag}{reject_msg}"
        )
    push_telegram("\n".join(detail_lines[:20]))
    if len(detail_lines) > 20:
        push_telegram("... (更多见日志)")
        for chunk in [detail_lines[i:i+20] for i in range(20, len(detail_lines), 20)]:
            push_telegram("\n".join(chunk))

    if not valid:
        log("❌ 无符合条件信号")
        push_telegram("❌ 本轮无高质量技术指标信号")
        return {}

    df_results = pd.DataFrame(valid).sort_values("score", ascending=False)
    top = df_results.head(FINAL_PICK_N)

    msg = ["✅ 技术指标信号："]
    for _, row in top.iterrows():
        symbol = row['symbol']
        signal = row['signal'].upper()
        confidence = row['direction_confidence']
        score = row['score']
        price_info = row.get('price_info')
        tech_msg = row['tech_msg']
        hold_min = row.get('estimated_hold_minutes', 1.5)
        msg.append(f"#{symbol} | {signal}")
        msg.append(f"  置信度: {confidence:.2f} | 得分: {score:.4f}")
        msg.append(f"  预估持仓时间: {hold_min} 分钟（{MAX_HOLD_SECONDS}秒）")
        if price_info:
            current = price_info['current_price']
            body_bottom = price_info['body_bottom']
            body_top = price_info['body_top']
            msg.append(f"  当前价格: {current:.6f}")
            msg.append(f"  上一根K线实体区间: [{body_bottom:.6f} - {body_top:.6f}]")
            entry_desc = price_info.get('entry_desc', '')
            if entry_desc:
                msg.append(f"  {entry_desc}")
            elif signal == 'LONG' and price_info.get('long_entry_max'):
                msg.append(f"  建议多单入场: 价格 ≤ {price_info['long_entry_max']:.6f}")
            elif signal == 'SHORT' and price_info.get('short_entry_min'):
                msg.append(f"  建议空单入场: 价格 ≥ {price_info['short_entry_min']:.6f}")
        else:
            msg.append("  ⚠️ 无法获取价格位置信息")
        msg.append(f"  技术确认: {tech_msg}")
        msg.append("")
    push_telegram("\n".join(msg))

    output_dict = {row['symbol']: (row['signal'], row['expected_return'], row.get('estimated_hold_minutes', 1.5), row['direction_confidence']) for _, row in top.iterrows()}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({k: v[0] for k, v in output_dict.items()}, f, indent=2, ensure_ascii=False)
    return output_dict

# ==================== 9. 异步交易类 ====================
class OKXTraderAsync:
    def __init__(self):
        self.exchange = None
        self.strategy_positions = {}
        self.pending_signals = []
        self.ws_client = None
        self.latest_mark_prices = {}
        self.position_lock = asyncio.Lock()

    async def init(self):
        log("🚀 初始化异步OKX交易客户端...")
        self.exchange = ccxt_async.okx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASS,
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "swap"}
        })
        self.exchange.set_sandbox_mode(IS_SANDBOX)
        await self.exchange.load_markets()
        try:
            await self.exchange.set_position_mode(True)
            log("✅ 已开启双向持仓模式")
        except Exception as e:
            err(f"设置双向持仓模式失败: {e}")
        self.ws_client = OKXWebSocket(self.on_mark_price)
        asyncio.create_task(self.ws_client.run())
        log("✅ 异步客户端初始化完成")

    async def on_mark_price(self, inst_id, mark_price):
        try:
            market = self.exchange.market(inst_id)
            symbol = market['symbol']
        except:
            symbol = inst_id.replace('-SWAP', '').replace('-', '/') + ':USDT'
        self.latest_mark_prices[symbol] = mark_price
        async with self.position_lock:
            if symbol in self.strategy_positions:
                await self.check_single_position_stop_loss(symbol, mark_price)

    async def check_single_position_stop_loss(self, symbol, mark_price):
        info = self.strategy_positions.get(symbol)
        if not info:
            return
        stop_price = info.get('stop_loss_price')
        if stop_price is None:
            return
        side = info['side']
        triggered = False
        if side == 'long' and mark_price <= stop_price:
            triggered = True
        elif side == 'short' and mark_price >= stop_price:
            triggered = True
        if triggered:
            log(f"💥 WebSocket 触发止损: {symbol} 标记价 {mark_price:.6f} 触及 {stop_price:.6f}")
            await self.close_position(symbol, reason="WebSocket实时止损")

    async def get_available_balance(self):
        balance = await self.exchange.fetch_balance()
        return balance.get('USDT', {}).get('free', 0.0)

    async def sync_positions(self):
        try:
            positions = await self.exchange.fetch_positions()
            pos_map = {}
            for p in positions:
                contracts = float(p.get('contracts', 0))
                if contracts == 0:
                    continue
                pos_side = p.get('info', {}).get('posSide')
                if pos_side in ['long', 'short']:
                    side = pos_side
                else:
                    side = 'long' if contracts > 0 else 'short'
                pos_map[p['symbol']] = side
            return pos_map
        except Exception as e:
            err(f"同步持仓失败: {e}")
            return {}

    def _check_reversal_signal(self, symbol, side, current_price, ema20_15m, slope_15m, rsi_val, macd_hist, macd_prev):
        is_uptrend = slope_15m > 0.001
        is_downtrend = slope_15m < -0.001
        if side == 'long':
            if not is_uptrend:
                return False, "非上涨趋势，无需检测反转"
            reasons = []
            if rsi_val > 70:
                reasons.append(f"RSI={rsi_val:.1f}超买")
            if macd_hist < 0 and macd_prev > 0:
                reasons.append("MACD柱由正转负")
            if ema20_15m and abs((current_price - ema20_15m)/ema20_15m) > 0.02:
                reasons.append(f"偏离均线{((current_price-ema20_15m)/ema20_15m*100):.1f}%")
            if len(reasons) >= 2:
                return True, f"顶部反转信号: {', '.join(reasons)}"
            else:
                return False, f"未达到反转阈值 (RSI:{rsi_val:.1f}, MACD柱:{macd_hist:.4f})"
        else:
            if not is_downtrend:
                return False, "非下跌趋势，无需检测反转"
            reasons = []
            if rsi_val < 30:
                reasons.append(f"RSI={rsi_val:.1f}超卖")
            if macd_hist > 0 and macd_prev < 0:
                reasons.append("MACD柱由负转正")
            if ema20_15m and abs((current_price - ema20_15m)/ema20_15m) > 0.02:
                reasons.append(f"偏离均线{((ema20_15m-current_price)/ema20_15m*100):.1f}%")
            if len(reasons) >= 2:
                return True, f"底部反转信号: {', '.join(reasons)}"
            else:
                return False, f"未达到反转阈值"

    async def open_position(self, symbol, side, expected_return, expected_hold_minutes,
                            ignore_price_position=False, is_with_trend=False, signal_price=None):
        async with self.position_lock:
            current_pos = await self.sync_positions()
            ccxt_symbol = self.exchange.market(symbol)['symbol']
            if ccxt_symbol in current_pos:
                log(f"⏸️ {symbol} 已有持仓 {current_pos[ccxt_symbol]}，拒绝重复开仓")
                return False

            available = await self.get_available_balance()
            margin_usdt = available * USE_PERCENT_OF_AVAILABLE
            if margin_usdt < MIN_BALANCE_USDT:
                log(f"💰 保证金不足 {margin_usdt:.2f} < {MIN_BALANCE_USDT}")
                return False
            log(f"💰 动态保证金: {margin_usdt:.2f} USDT")

            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            df_15m = fetch_klines_with_retry(symbol, HIGHER_BAR, 20)
            if df_15m is not None and len(df_15m) >= 20:
                closes_15m = df_15m['c']
                rsi_val = compute_rsi(closes_15m, RSI_PERIOD)
                _, _, macd_hist, macd_prev = compute_macd(closes_15m, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            else:
                rsi_val = 50
                macd_hist = 0
                macd_prev = 0

            ema20_15m, slope_15m, _, _ = get_15min_trend(symbol)

            is_reversal, rev_reason = self._check_reversal_signal(symbol, side, current_price, ema20_15m, slope_15m, rsi_val, macd_hist, macd_prev)
            if is_reversal:
                log(f"⛔ 反转检测拒绝开仓 {symbol} {side.upper()}: {rev_reason}")
                push_telegram(f"⛔ 反转检测拒单: {symbol} {side.upper()}\n{rev_reason}")
                return False

            df_vol = fetch_klines_with_retry(symbol, BAR, 30)
            if df_vol is not None and len(df_vol) >= 20:
                vol_profile, atr_pct = detect_volatility_profile(df_vol)
                adaptive_params = get_adaptive_trading_params(BAR, vol_profile)
                atr_multiplier = adaptive_params["atr_multiplier"]
                stop_loss_pct = adaptive_params["stop_loss_pct"]
                trailing_stop_pct = adaptive_params["trailing_stop_pct"]
                log(f"自适应参数: {vol_profile} ({atr_pct:.2f}%)")
            else:
                atr_multiplier = ATR_MULTIPLIER
                stop_loss_pct = STOP_LOSS_PCT
                trailing_stop_pct = TRAILING_STOP_PCT

            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1.0)
            amount = (margin_usdt * LEVERAGE) / (current_price * contract_size)
            amount = self.exchange.amount_to_precision(symbol, amount)
            amount = float(amount)

            try:
                await self.exchange.set_leverage(LEVERAGE, symbol, params={'mgnMode': 'isolated', 'posSide': side})
            except Exception as e:
                err(f"设置杠杆失败 {symbol}: {e}")
                return False

            try:
                order_side = 'buy' if side == 'long' else 'sell'
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=order_side,
                    amount=amount,
                    params={'positionSide': side, 'tdMode': 'isolated'}
                )
                actual_open_price = order.get('average', current_price)
                actual_filled = order.get('filled', amount)
                if actual_filled == 0:
                    info = order.get('info', {})
                    actual_filled = float(info.get('filledSz', 0))
                    actual_open_price = float(info.get('avgPx', current_price))
                if actual_filled == 0:
                    err(f"订单未成交: {order}")
                    return False

                atr = self.get_atr_sync(symbol)
                if atr is not None:
                    if side == 'long':
                        stop_loss_price = actual_open_price - atr * atr_multiplier
                    else:
                        stop_loss_price = actual_open_price + atr * atr_multiplier
                else:
                    if side == 'long':
                        stop_loss_price = actual_open_price * (1 - stop_loss_pct / 100)
                    else:
                        stop_loss_price = actual_open_price * (1 + stop_loss_pct / 100)

                used_margin = actual_filled * actual_open_price / LEVERAGE
                self.strategy_positions[ccxt_symbol] = {
                    'side': side,
                    'open_price': actual_open_price,
                    'open_time': time.time(),
                    'open_qty': actual_filled,
                    'open_margin': used_margin,
                    'open_nominal': actual_filled * actual_open_price,
                    'stop_loss_price': stop_loss_price,
                    'highest_price': actual_open_price,
                    'lowest_price': actual_open_price,
                    'trailing_stop_pct': trailing_stop_pct,
                    'trailing_activated': False,
                    'expected_return': expected_return,
                    'expected_met': False,
                    'max_hold_seconds': MAX_HOLD_SECONDS,
                    'half_closed': False
                }
                self._save_strategy_positions()
                inst_id = self.exchange.market(ccxt_symbol)['id']
                await self.ws_client.subscribe_mark_price([inst_id])

                msg = (f"✅ 开仓成功\n"
                       f"币种: {symbol}\n"
                       f"方向: {side.upper()}\n"
                       f"当前价格: {current_price:.6f}\n"
                       f"开仓价: {actual_open_price:.6f}\n"
                       f"建议开仓价: {signal_price if signal_price else actual_open_price:.6f}\n"
                       f"止损价: {stop_loss_price:.6f}\n"
                       f"张数: {actual_filled}\n"
                       f"保证金: {used_margin:.2f} USDT\n"
                       f"最大持仓: {MAX_HOLD_SECONDS}秒")
                push_telegram(msg)
                log(f"✅ 开仓成功 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 止损: {stop_loss_price:.4f}")
                return True
            except Exception as e:
                err(f"开仓异常 {symbol}: {e}")
                return False

    async def close_position(self, symbol, reason=""):
        async with self.position_lock:
            if symbol not in self.strategy_positions:
                return False
            info = self.strategy_positions[symbol]
            side = info['side']
            open_price = info['open_price']
            open_qty = info['open_qty']
            open_margin = info['open_margin']
            open_time = info['open_time']
            try:
                positions = await self.exchange.fetch_positions([symbol])
                real_pos = None
                for p in positions:
                    if p.get('side') == side and float(p.get('contracts', 0)) > 0:
                        real_pos = p
                        break
                if not real_pos:
                    log(f"⚠️ 无实际持仓 {symbol}，清理记录")
                    del self.strategy_positions[symbol]
                    self._save_strategy_positions()
                    return True
                amount = abs(float(real_pos['contracts']))
                close_price = float(real_pos.get('last', 0)) or float(real_pos.get('markPrice', 0))
                if close_price == 0:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    close_price = ticker['last']
                order_side = 'sell' if side == 'long' else 'buy'
                params = {'reduceOnly': True, 'tdMode': 'isolated', 'mgnMode': 'isolated'}
                config = await self.exchange.public_get_account_config()
                pos_mode = config.get('data', [{}])[0].get('posMode', 'net_mode')
                if pos_mode == 'long_short_mode':
                    params['posSide'] = side
                order = await self.exchange.create_order(symbol, 'market', order_side, amount, params=params)

                if side == 'long':
                    pnl_usdt = (close_price - open_price) * open_qty
                else:
                    pnl_usdt = (open_price - close_price) * open_qty
                pnl_pct = (pnl_usdt / open_margin) * 100 if open_margin > 0 else 0
                hold_seconds = time.time() - open_time
                msg = (f"🔻 平仓\n"
                       f"币种: {symbol}\n"
                       f"方向: {side.upper()}\n"
                       f"开仓价: {open_price:.6f}\n"
                       f"平仓价: {close_price:.6f}\n"
                       f"持仓时长: {hold_seconds:.1f}秒\n"
                       f"盈亏: {pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)\n"
                       f"原因: {reason}")
                push_telegram(msg)
                log(f"✅ 平仓成功 {symbol} | 盈亏 {pnl_usdt:+.2f} | {reason}")
                del self.strategy_positions[symbol]
                self._save_strategy_positions()
                return True
            except Exception as e:
                err(f"平仓失败 {symbol}: {e}")
                return False

    def get_atr_sync(self, symbol):
        try:
            df = fetch_klines_with_retry(symbol, BAR, ATR_PERIOD+1)
            if df is None or len(df) < ATR_PERIOD+1:
                return None
            high = df['h'].values
            low = df['l'].values
            close = df['c'].values
            tr = np.maximum(high[1:] - low[1:],
                            np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1]))
            atr = np.mean(tr[-ATR_PERIOD:])
            return atr
        except Exception as e:
            err(f"计算ATR失败 {symbol}: {e}")
            return None

    def _save_strategy_positions(self):
        try:
            to_save = {}
            for sym, info in self.strategy_positions.items():
                to_save[sym] = {
                    'side': info['side'],
                    'open_price': info['open_price'],
                    'open_time': info['open_time'],
                    'open_qty': info['open_qty'],
                    'open_margin': info['open_margin'],
                    'open_nominal': info['open_nominal'],
                    'stop_loss_price': info.get('stop_loss_price'),
                    'highest_price': info.get('highest_price'),
                    'lowest_price': info.get('lowest_price'),
                    'trailing_stop_pct': info.get('trailing_stop_pct'),
                    'trailing_activated': info.get('trailing_activated', False),
                    'expected_return': info.get('expected_return'),
                    'expected_met': info.get('expected_met', False),
                    'max_hold_seconds': info.get('max_hold_seconds', MAX_HOLD_SECONDS),
                    'half_closed': info.get('half_closed', False)
                }
            with open(STRATEGY_POSITIONS_FILE, 'w') as f:
                json.dump(to_save, f, indent=2)
        except Exception as e:
            err(f"保存策略持仓失败: {e}")

    async def check_manual_close(self):
        actual = await self.sync_positions()
        removed = [sym for sym in list(self.strategy_positions.keys()) if sym not in actual]
        for sym in removed:
            log(f"人工平仓检测: {sym} 已不存在")
            del self.strategy_positions[sym]
        if removed:
            self._save_strategy_positions()

    def set_pending_signals(self, signals_list):
        self.pending_signals = []
        for raw_symbol, side, expected_return, hold_minutes, signal_price in signals_list:
            try:
                market = self.exchange.market(raw_symbol)
                ccxt_symbol = market['symbol']
                if signal_price is None:
                    ticker = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={raw_symbol}").json()
                    signal_price = float(ticker['data'][0]['last'])
                self.pending_signals.append({
                    'raw_symbol': raw_symbol,
                    'ccxt_symbol': ccxt_symbol,
                    'side': side,
                    'signal_price': signal_price,
                    'expected_return': expected_return,
                    'expected_hold_minutes': hold_minutes,
                    'timestamp': time.time()
                })
            except Exception as e:
                err(f"获取市场信息失败 {raw_symbol}: {e}")
        log(f"📋 设置待开仓信号: {len(self.pending_signals)} 个（有效期{SIGNAL_VALID_SECONDS}秒）")

    def clear_pending_signals(self):
        self.pending_signals = []

    async def check_and_open_pending(self):
        if not self.pending_signals:
            return
        to_remove = []
        now = time.time()
        for idx, sig in enumerate(self.pending_signals):
            if now - sig['timestamp'] > SIGNAL_VALID_SECONDS:
                log(f"⏰ 信号过期: {sig['raw_symbol']} {sig['side'].upper()}，已超过 {SIGNAL_VALID_SECONDS} 秒")
                to_remove.append(idx)
                continue
                
            raw_symbol = sig['raw_symbol']
            side = sig['side']
            expected_return = sig['expected_return']
            hold_min = sig['expected_hold_minutes']
            signal_price = sig['signal_price']
            try:
                ticker = await self.exchange.fetch_ticker(raw_symbol)
                current_price = ticker['last']
            except:
                continue
            current_pos = await self.sync_positions()
            ccxt_symbol = self.exchange.market(raw_symbol)['symbol']
            if ccxt_symbol in current_pos:
                log(f"⏸️ {raw_symbol} 已有持仓，取消信号")
                to_remove.append(idx)
                continue
            success = await self.open_position(raw_symbol, side, expected_return, hold_min,
                                               ignore_price_position=True, is_with_trend=False,
                                               signal_price=signal_price)
            if success:
                to_remove.append(idx)
        for idx in sorted(to_remove, reverse=True):
            self.pending_signals.pop(idx)

    async def check_reversal_close(self):
        for symbol, info in list(self.strategy_positions.items()):
            side = info['side']
            _, _, is_downtrend, is_uptrend = get_15min_trend(symbol)
            if side == 'long' and is_downtrend:
                log(f"🔄 趋势反转平仓: {symbol} 多单，15m趋势已转为下跌")
                await self.close_position(symbol, reason="趋势反转（多转空）")
            elif side == 'short' and is_uptrend:
                log(f"🔄 趋势反转平仓: {symbol} 空单，15m趋势已转为上涨")
                await self.close_position(symbol, reason="趋势反转（空转多）")

    async def check_and_close_positions(self):
        closed_any = False
        try:
            all_positions = await self.exchange.fetch_positions()
        except Exception as e:
            err(f"获取持仓列表失败: {e}")
            return False

        pos_map = {}
        for p in all_positions:
            contracts = float(p.get('contracts', 0))
            if contracts != 0:
                pos_map[p['symbol']] = p

        for sym, info in list(self.strategy_positions.items()):
            if sym not in pos_map:
                log(f"⚠️ 策略持仓 {sym} 未在交易所持仓中找到，可能已被平仓")
                del self.strategy_positions[sym]
                self._save_strategy_positions()
                continue

            pos = pos_map[sym]
            current_price = (
                float(pos.get('last', 0)) or
                float(pos.get('markPrice', 0)) or
                float(pos.get('info', {}).get('last', 0)) or
                float(pos.get('info', {}).get('markPrice', 0))
            )
            if current_price == 0:
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    current_price = ticker['last']
                except:
                    log(f"⚠️ 无法获取 {sym} 最新价格，跳过本次检查")
                    continue

            metrics = self.safe_calc_position_metrics(pos, current_price)
            if not metrics["valid"]:
                continue
            pnl_percent = metrics["pnl_pct"]
            
            if info['side'] == 'long':
                if current_price > info.get('highest_price', info['open_price']):
                    info['highest_price'] = current_price
                peak = info.get('highest_price', info['open_price'])
                drawdown_pct = (peak - current_price) / peak * 100 if peak > 0 else 0
            else:
                if current_price < info.get('lowest_price', info['open_price']):
                    info['lowest_price'] = current_price
                trough = info.get('lowest_price', info['open_price'])
                drawdown_pct = (current_price - trough) / trough * 100 if trough > 0 else 0

            hold_seconds = time.time() - info['open_time']
            max_hold = info.get('max_hold_seconds', MAX_HOLD_SECONDS)

            # 初始止损
            stop_price = info.get('stop_loss_price')
            if stop_price is not None:
                if (info['side'] == 'long' and current_price <= stop_price) or \
                   (info['side'] == 'short' and current_price >= stop_price):
                    log(f"💥 触发初始止损: {sym} 当前价 {current_price:.4f} 触及止损价 {stop_price:.4f}")
                    await self.close_position(sym, reason=f"初始止损 {stop_price:.4f}")
                    closed_any = True
                    continue

            # 跟踪止损
            if pnl_percent > 3.0:
                info['trailing_activated'] = True
            if info.get('trailing_activated', False) and drawdown_pct >= info.get('trailing_stop_pct', TRAILING_STOP_PCT):
                log(f"📉 触发跟踪止损: {sym} 回撤 {drawdown_pct:.2f}% (阈值{info.get('trailing_stop_pct', TRAILING_STOP_PCT)}%)")
                await self.close_position(sym, reason=f"跟踪止损回撤{drawdown_pct:.1f}%")
                closed_any = True
                continue

            # 超时强制平仓
            if hold_seconds >= max_hold:
                log(f"⏰ 持仓超时: {sym} 已持仓 {hold_seconds:.1f} 秒，强制平仓")
                if pnl_percent > 0:
                    reason = f"超时止盈（{hold_seconds:.0f}秒）"
                else:
                    reason = f"超时止损（{hold_seconds:.0f}秒）"
                await self.close_position(sym, reason=reason)
                closed_any = True
                continue

            log(f"📉 检查持仓 {sym}: 盈亏 {pnl_percent:.2f}%, 跟踪激活: {info.get('trailing_activated', False)}, 持仓时长 {hold_seconds:.1f}/{max_hold} 秒")

        return closed_any

    def safe_calc_position_metrics(self, pos_info, current_price):
        try:
            entry = float(pos_info.get('entryPrice', 0) or 0)
            size = float(pos_info.get('contracts', 0) or 0)
            side = pos_info.get('side', 'unknown').lower()
            margin = float(pos_info.get('initialMargin', 0) or 0)
            
            if entry == 0 or size == 0 or margin == 0:
                return {"pnl_pct": 0.0, "unrealized_pnl": 0.0, "valid": False}
            
            if side == 'long':
                pnl = (current_price - entry) * size
            else:
                pnl = (entry - current_price) * size
            
            pnl_pct = (pnl / margin) * 100
            return {"pnl_pct": round(pnl_pct, 2), "unrealized_pnl": round(pnl, 4), "valid": True}
        except (TypeError, ValueError, ZeroDivisionError) as e:
            log_system_warn(f"⚠️ 持仓数据容错触发: {e}")
            return {"pnl_pct": 0.0, "unrealized_pnl": 0.0, "valid": False}

# ==================== 10. 异步主程序 ====================
async def main_async():
    trader = OKXTraderAsync()
    await trader.init()
    last_pred = datetime.now() - timedelta(seconds=PREDICTION_INTERVAL)
    has_set_pending_this_cycle = False

    log("========== 纯技术指标异步交易系统启动 | 杠杆3倍 | WebSocket实时止损 | 信号30秒有效 | 持仓最多90秒 ==========")
    push_telegram("🤖 技术指标机器人启动 (杠杆3x, WebSocket风控, 无AI模型, 信号30秒有效, 持仓90秒上限)")

    while True:
        try:
            now = datetime.now()
            await trader.check_manual_close()
            await trader.check_reversal_close()
            await trader.check_and_close_positions()
            await trader.check_and_open_pending()

            if (now - last_pred).total_seconds() >= PREDICTION_INTERVAL:
                has_set_pending_this_cycle = False
                signals_dict = await asyncio.to_thread(run_prediction_cycle)
                last_pred = now
                trader.clear_pending_signals()
                if signals_dict and not has_set_pending_this_cycle:
                    existing = set(trader.strategy_positions.keys())
                    filtered = {sym: (sig, ret, hold, conf) for sym, (sig, ret, hold, conf) in signals_dict.items()
                                if sym not in existing}
                    if filtered:
                        signals_list = []
                        for sym, (sig, ret, hold, conf) in filtered.items():
                            try:
                                ticker_data = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={sym}").json()
                                sp = float(ticker_data['data'][0]['last'])
                            except:
                                sp = None
                            signals_list.append((sym, sig, ret, hold, sp))
                        trader.set_pending_signals(signals_list)
                        has_set_pending_this_cycle = True
                        push_telegram(f"📋 设置 {len(signals_list)} 个技术指标信号（有效期{SIGNAL_VALID_SECONDS}秒）")
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            log("🛑 收到停止信号")
            break
        except Exception as e:
            log_system_exception(f"主循环异常: {e}")
            push_telegram(f"🚨 异常: {str(e)[:100]}")
            await asyncio.sleep(10)

    if trader.ws_client:
        await trader.ws_client.close()
    await trader.exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log("手动停止")
        push_telegram("🛑 机器人停止")
