import numpy as np
import pandas as pd
import requests
import torch
import warnings
from datetime import datetime, timedelta
import json
import time
import traceback
import sys
import logging
import ccxt
import os
import timesfm
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import hmac
import hashlib
import re

# ==================== 1. 全局配置 & 日志初始化 ====================
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
log_dir = "/root/timesfm_lab"
if not os.path.exists(log_dir): os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(f"{log_dir}/trading_bot.log", encoding="utf-8"), logging.StreamHandler()]
)
log = logging.info
err = logging.error

# ==================== 2. 核心参数 ====================
TG_BOT_TOKEN = "8722422674:AAGrKmRurQ2G__j-Vxbh5451v0e9_u97CQY"
TG_CHAT_ID = "5372217316"
TG_PROXIES = None

BAR = "5m"
HIGHER_BAR = "15m"
LIMIT = 900
HORIZON = 3

TOP_N = 50
FINAL_PICK_N = 3
TREND_FOLLOWING_RETURN = 0.003
COUNTER_TREND_RETURN = 0.008

MIN_R_SQUARED = 0.2
MIN_DIRECTION_CONFIDENCE = 0.65

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
MAX_SINGLE_TRADE_USDT = 50
MAX_MARGIN_MULTIPLIER = 2

TAKE_PROFIT_PCT = 8.0
STOP_LOSS_PCT = 1.0
MAX_HOLD_SECONDS = 300

MIN_VOLUME_USDT = 10_000_000
MIN_MARKET_CAP_USDT = 20_000_000
VOLATILITY_SAMPLE_SIZE = 200

LEVERAGE = 3
PRICE_POSITION_RATIO = 0.1
FAVORABLE_MOVE_PCT = 0.2

MAX_CONCURRENT_POSITIONS = 3
MAX_TOTAL_MARGIN_RATIO = 0.5

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

ENABLE_NEWS_MONITOR = True
NEWS_PAUSE_MINUTES = 5
ENABLE_VOLUME_ANOMALY = True
VOLUME_ANOMALY_THRESHOLD = 5.0
ENABLE_TREND_REVERSAL = True
ENABLE_PRICE_MOMENTUM_FILTER = True
MOMENTUM_LIMIT_PCT = 1.5

FINNHUB_API_KEY = "d7krlm1r01qiqbcvgihgd7krlm1r01qiqbcvgii0"

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

# ==================== 4. TimesFM模型 ====================
log("🚀 加载 TimesFM 模型...")
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"使用设备: {device}")

def _build_timesfm_model():
    errors = []
    try:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m = m.to(device)
        try:
            m = torch.compile(m)
        except:
            pass
        try:
            fc = timesfm.ForecastConfig(max_context=8192, max_horizon=512)
            m.compile(forecast_config=fc)
        except:
            pass
        return m
    except Exception as e:
        errors.append(f"TimesFM_2p5_200M_torch 失败: {e}")
    try:
        m = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if torch.cuda.is_available() else "cpu",
                per_core_batch_size=32,
                horizon_len=HORIZON,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.5-200m-pytorch"
            ),
        )
        return m
    except Exception as e:
        errors.append(f"TimesFm 经典接口失败: {e}")
    raise RuntimeError("TimesFM 初始化失败:\n- " + "\n- ".join(errors))

model = _build_timesfm_model()
log("✅ 模型加载完成")

# ==================== 5. OKX数据获取 ====================
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

# ==================== 6. 技术指标计算 ====================
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
        df = fetch_klines_with_retry(symbol, HIGHER_BAR, 50)
        if df is None or len(df) < 30:
            return None, None
        closes = df['c']
        ema20 = closes.ewm(span=20, adjust=False).mean()
        prev_ema20 = ema20.iloc[-2]
        curr_ema20 = ema20.iloc[-1]
        slope = (curr_ema20 - prev_ema20) / prev_ema20 if prev_ema20 != 0 else 0
        return curr_ema20, slope
    except Exception as e:
        err(f"获取15分钟趋势失败 {symbol}: {e}")
        return None, None

def get_1h_trend(symbol):
    try:
        df = fetch_klines_with_retry(symbol, "1h", 50)
        if df is None or len(df) < 30:
            return None, None, False, False
        closes = df['c']
        ema20 = closes.ewm(span=20, adjust=False).mean()
        prev_ema20 = ema20.iloc[-2]
        curr_ema20 = ema20.iloc[-1]
        slope = (curr_ema20 - prev_ema20) / prev_ema20 if prev_ema20 != 0 else 0
        recent_closes = closes.iloc[-6:].values
        below_ema_count = sum(recent_closes < ema20.iloc[-6:].values)
        is_downtrend = (slope < -0.002) or (below_ema_count >= 4)
        is_uptrend = (slope > 0.002) or (sum(recent_closes > ema20.iloc[-6:].values) >= 4)
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

# ==================== 6.5 波动率自适应参数 ====================
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

# ==================== 7. 预测评分 + 多空相对评分制 ====================
def get_ema20_5m(symbol):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 30)
        if df is None or len(df) < 25:
            return None
        closes = df['c'].astype(float)
        ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        return ema20
    except:
        return None

def check_weak_rally(symbol, current_price, rsi):
    try:
        ema20 = get_ema20_5m(symbol)
        if ema20 is None:
            return False
        if abs(current_price - ema20) / ema20 > 0.001:
            return False
        if rsi >= 50:
            return False
        df = fetch_klines_with_retry(symbol, BAR, 3)
        if df is None or len(df) < 2:
            return False
        prev_close = float(df['c'].iloc[-2])
        curr_close = float(df['c'].iloc[-1])
        if prev_close >= ema20 and curr_close < ema20:
            return True
        return False
    except:
        return False

def compute_signal_score(symbol, side, current_price, expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m, is_1h_downtrend=False, is_1h_uptrend=False, is_15m_downtrend=False, is_15m_uptrend=False):
    base_conf = 0.7 * consistency + 0.3 * max(0.0, min(1.0, r_squared))
    trend_factor = 1.0
    if side == 'long':
        if is_1h_downtrend:
            trend_factor = 0.0
        elif is_15m_downtrend:
            trend_factor = 0.3
    else:
        if is_1h_uptrend:
            trend_factor = 0.0
        elif is_15m_uptrend:
            trend_factor = 0.3

    direction_confidence = base_conf * trend_factor
    if trend_factor < 1.0 and abs(expected_return) < 0.015:
        direction_confidence = min(direction_confidence, 0.6)
    base_score = abs(expected_return) * 100 * 0.4 + r_squared * 0.3 + consistency * 0.3
    base_score = min(1.0, base_score / 2.0)
    is_with_trend = (side == 'short' and slope_15m is not None and slope_15m < -0.002) or \
                    (side == 'long' and slope_15m is not None and slope_15m > 0.002)
    if is_with_trend:
        final_score = min(1.0, base_score * 1.5)
    else:
        final_score = base_score * 0.6
    if vol_ratio > 2.0:
        final_score = min(1.0, final_score + min(0.3, (vol_ratio - 2.0) * 0.1))
    return direction_confidence, final_score * 100

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

def validate_signal(signal_type, symbol, current_price, rsi, adx, atr_pct, forecast_values=None):
    if atr_pct is not None and atr_pct > 5.0:
        return False, f"波动率过高 ({atr_pct:.2f}%)，暂停开仓"

    bb_upper, bb_lower = calculate_bollinger_bands(symbol)
    
    if signal_type == 'LONG':
        if rsi is not None and rsi > 65:
            return False, f"RSI={rsi:.1f} 超买区，禁止追多"
        if bb_upper is not None and current_price > bb_upper:
            return False, f"价格突破布林带上轨 {bb_upper:.6f}，过高"
        if adx is not None and adx > 60:
            return False, f"ADX={adx:.1f} 趋势极端，可能衰竭"
        if forecast_values is not None and len(forecast_values) >= 3:
            forecast_high_max = max(forecast_values[:3])
            if forecast_high_max < current_price:
                return False, f"TimesFM预测未来最高价 {forecast_high_max:.6f} 低于当前价，上涨乏力"

    elif signal_type == 'SHORT':
        if rsi is not None and rsi < 35:
            return False, f"RSI={rsi:.1f} 超卖区，禁止追空"
        if bb_lower is not None and current_price < bb_lower:
            return False, f"价格跌破布林带下轨 {bb_lower:.6f}，过低"
        if adx is not None and adx > 60:
            return False, f"ADX={adx:.1f} 趋势极端，可能衰竭"
        if forecast_values is not None and len(forecast_values) >= 3:
            forecast_low_min = min(forecast_values[:3])
            if forecast_low_min > current_price:
                return False, f"TimesFM预测未来最低价 {forecast_low_min:.6f} 高于当前价，下跌空间不足"

    try:
        df_vol = fetch_klines_with_retry(symbol, BAR, 21)
        if df_vol is not None and len(df_vol) >= 21:
            avg_volume = df_vol['v'].iloc[-21:-1].mean()
            current_volume = df_vol['v'].iloc[-1]
            if current_volume > avg_volume * 5:
                prev_close = df_vol['c'].iloc[-2]
                if signal_type == 'LONG' and current_price < prev_close:
                    return False, f"成交量突增 {current_volume/avg_volume:.1f} 倍，价格下跌，拒绝做多"
                elif signal_type == 'SHORT' and current_price > prev_close:
                    return False, f"成交量突增 {current_volume/avg_volume:.1f} 倍，价格上涨，拒绝做空"
    except Exception as e:
        log(f"成交量过滤异常 {symbol}: {e}")

    return True, "信号有效"

def estimate_hold_minutes(forecast_values, current_price, target_return_pct, side):
    if forecast_values is None or len(forecast_values) == 0:
        return HORIZON * 5
    price_seq = [current_price] + list(forecast_values[:HORIZON])
    bar_minutes = 5
    for i in range(1, len(price_seq)):
        if side == 'long':
            ret = (price_seq[i] - current_price) / current_price
        else:
            ret = (current_price - price_seq[i]) / current_price
        if ret >= target_return_pct:
            return i * bar_minutes
    return HORIZON * bar_minutes

# ==================== 新闻监控函数（使用 Finnhub）====================
def check_news_impact(symbol):
    if not ENABLE_NEWS_MONITOR:
        return False, ""
    if not FINNHUB_API_KEY:
        return False, ""
    
    coin = symbol.split('-')[0].upper()
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {
            "category": "crypto",
            "token": FINNHUB_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            log(f"Finnhub API 响应异常: {resp.status_code} - {resp.text[:100]}")
            return False, ""
        
        data = resp.json()
        now = datetime.now()
        for article in data[:10]:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            if coin in headline.upper() or coin in summary.upper():
                published_time = article.get('datetime')
                if published_time:
                    published = datetime.fromtimestamp(published_time)
                    if (now - published).total_seconds() < NEWS_PAUSE_MINUTES * 60:
                        log(f"📰 Finnhub 检测到相关新闻: {headline[:50]}...")
                        return True, headline
        return False, ""
    except Exception as e:
        log(f"Finnhub 新闻 API 调用失败 {symbol}: {e}")
        return False, ""

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
    """
    检测趋势反转
    side: 'long' 表示想要开多单，需要检测下跌趋势是否转为上涨（底部反转）
           'short' 表示想要开空单，需要检测上涨趋势是否转为下跌（顶部反转）
    返回 (is_reversal, reason)
    """
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
    
    # 计算MACD
    macd_line, signal_line, _, _ = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_golden_cross = macd_line > signal_line and (macd_line - signal_line) > 0.0001
    macd_death_cross = macd_line < signal_line and (signal_line - macd_line) > 0.0001
    
    if side == 'long':
        # 想要开多单：需要底部反转信号（下跌趋势末端）
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
    else:  # side == 'short'
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
        if df is None or len(df) < 50:
            return None, "数据不足"
        ts = df['c'].values.astype(np.float32)
        current_price = float(ts[-1])

        df_vol = fetch_klines_with_retry(instId, BAR, 30)
        if df_vol is not None and len(df_vol) >= 21:
            avg_volume = df_vol['v'].iloc[-21:-1].mean()
            current_volume = float(df_vol['v'].iloc[-1])
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            vol_ratio = 1.0

        with torch.inference_mode():
            try:
                point, _ = model.forecast(horizon=HORIZON, inputs=[ts])
                forecast_values = np.array(point[0], dtype=np.float32)
            except TypeError:
                out = model.forecast(inputs=[ts], freq=[0])
                if isinstance(out, tuple):
                    forecast_values = np.array(out[0][0], dtype=np.float32)
                else:
                    forecast_values = np.array(out[0], dtype=np.float32)
            except Exception:
                out = model.forecast(inputs=[ts])
                if isinstance(out, tuple):
                    forecast_values = np.array(out[0][0], dtype=np.float32)
                else:
                    forecast_values = np.array(out[0], dtype=np.float32)

        final_forecast = forecast_values[-1]
        expected_return = (final_forecast - current_price) / current_price

        steps = np.arange(len(forecast_values))
        z = np.polyfit(steps, forecast_values, 1)
        p = np.poly1d(z)
        y_average = np.mean(forecast_values)
        ss_res = np.sum((forecast_values - p(steps)) ** 2)
        ss_tot = np.sum((forecast_values - y_average) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        diffs = np.diff(np.insert(forecast_values, 0, current_price))
        direction = 1 if expected_return > 0 else -1
        consistency = np.sum(np.sign(diffs) == direction) / len(diffs)

        ema20_15m, slope_15m = get_15min_trend(instId)
        ema20_1h, slope_1h, is_1h_downtrend, is_1h_uptrend = get_1h_trend(instId)
        is_15m_downtrend = slope_15m is not None and slope_15m < -0.002
        is_15m_uptrend = slope_15m is not None and slope_15m > 0.002

        long_conf, long_score = compute_signal_score(instId, 'long', current_price, expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m, is_1h_downtrend, is_1h_uptrend, is_15m_downtrend, is_15m_uptrend)
        short_conf, short_score = compute_signal_score(instId, 'short', current_price, -expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m, is_1h_downtrend, is_1h_uptrend, is_15m_downtrend, is_15m_uptrend)

        adx = get_adx(instId)
        atr_pct = get_atr_percent(instId)
        try:
            df_rsi = fetch_klines_with_retry(instId, BAR, 20)
            rsi_val = compute_rsi(df_rsi['c'], RSI_PERIOD) if df_rsi is not None else 50
        except:
            rsi_val = 50

        # 修复：将 numpy 数组转为 Series 再传给 MACD 函数
        ts_series = pd.Series(ts)
        macd_line, signal_line, hist, _ = compute_macd(ts_series, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

        if is_15m_uptrend:
            min_ret_long = TREND_FOLLOWING_RETURN
            min_ret_short = COUNTER_TREND_RETURN
        elif is_15m_downtrend:
            min_ret_long = COUNTER_TREND_RETURN
            min_ret_short = TREND_FOLLOWING_RETURN
        else:
            min_ret_long = TREND_FOLLOWING_RETURN
            min_ret_short = TREND_FOLLOWING_RETURN

        best_side = None
        best_score = 0
        best_conf = 0
        best_ret = 0
        reason_detail = ""

        if consistency < 0.2:
            if short_score > 0.4 and abs(expected_return) >= 0.003:
                best_side = 'short'
                best_score = short_score
                best_conf = short_conf
                best_ret = -abs(expected_return)
                reason_detail = f"一致性极低({consistency:.2f})，强制空单扫描"
                valid, reject_reason = validate_signal('SHORT', instId, current_price, rsi_val, adx, atr_pct, forecast_values)
                if not valid:
                    return None, f"信号过滤器拦截 (SHORT): {reject_reason}"
                candle = fetch_previous_candle(instId)
                price_info = None
                if candle:
                    open_p, high, low, close = candle
                    body_top = max(open_p, close)
                    body_bottom = min(open_p, close)
                    body_len = body_top - body_bottom
                    if body_len > 0:
                        is_with_trend = (best_side == 'long' and is_15m_uptrend) or (best_side == 'short' and is_15m_downtrend)
                        if best_side == 'long':
                            if is_with_trend:
                                long_entry_max = body_top
                                entry_desc = f"顺势多单，建议当前价 ≤ {body_top:.6f} (实体顶部)"
                            else:
                                long_entry_max = body_bottom + body_len * PRICE_POSITION_RATIO
                                entry_desc = f"逆势多单，建议入场价 ≤ {long_entry_max:.6f} (实体底部+{PRICE_POSITION_RATIO*100:.0f}%区域)"
                            short_entry_min = None
                        else:
                            if is_with_trend:
                                short_entry_min = body_bottom
                                entry_desc = f"顺势空单，建议当前价 ≥ {body_bottom:.6f} (实体底部)"
                            else:
                                short_entry_min = body_top - body_len * PRICE_POSITION_RATIO
                                entry_desc = f"逆势空单，建议入场价 ≥ {short_entry_min:.6f} (实体顶部-{PRICE_POSITION_RATIO*100:.0f}%区域)"
                            long_entry_max = None
                        price_info = {
                            'current_price': current_price,
                            'body_top': body_top,
                            'body_bottom': body_bottom,
                            'long_entry_max': long_entry_max,
                            'short_entry_min': short_entry_min,
                            'entry_desc': entry_desc,
                            'is_with_trend': is_with_trend
                        }
                hold_minutes = estimate_hold_minutes(forecast_values, current_price, abs(best_ret), best_side)
                result = {
                    "symbol": instId, "signal": best_side, "expected_return": best_ret,
                    "r_squared": r_squared, "consistency": consistency, "direction_confidence": best_conf,
                    "score": best_score, "last_price": current_price, "price_info": price_info,
                    "tech_msg": reason_detail, "long_score": long_score, "short_score": short_score,
                    "adx": adx if adx else 0, "atr_pct": atr_pct if atr_pct else 0,
                    "vol_median_pct": 0, "rsi": rsi_val,
                    "estimated_hold_minutes": hold_minutes
                }
                return result, ""

        weak_rally = check_weak_rally(instId, current_price, rsi_val)
        diff = long_score - short_score

        if long_conf < 0.3 and is_15m_downtrend:
            if short_score > 0.6 and abs(expected_return) >= min_ret_short:
                best_side = 'short'
                best_score = short_score
                best_conf = short_conf
                best_ret = -abs(expected_return)
                reason_detail = f"多单置信度极低({long_conf:.2f})且趋势向下，强制空单(降门槛至0.6)"
        elif weak_rally and (short_score - long_score) > 0.15:
            if abs(expected_return) >= TREND_FOLLOWING_RETURN:
                best_side = 'short'
                best_score = short_score
                best_conf = short_conf
                best_ret = -abs(expected_return)
                reason_detail = f"弱势反抽触发顺势补票空单 (短分{short_score:.1f} > 多分{long_score:.1f}+0.15)"
        else:
            if diff > 0.2 and long_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_long:
                best_side = 'long'
                best_score = long_score
                best_conf = long_conf
                best_ret = abs(expected_return)
                reason_detail = f"多空相对评分胜出 (diff={diff:.2f})"
            elif diff < -0.2 and short_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_short:
                best_side = 'short'
                best_score = short_score
                best_conf = short_conf
                best_ret = -abs(expected_return)
                reason_detail = f"多空相对评分胜出 (diff={diff:.2f})"
            else:
                if is_15m_downtrend and short_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_short:
                    best_side = 'short'
                    best_score = short_score
                    best_conf = short_conf
                    best_ret = -abs(expected_return)
                    reason_detail = f"顺势空单（diff={diff:.2f}不足但趋势配合）"
                elif is_15m_uptrend and long_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_long:
                    best_side = 'long'
                    best_score = long_score
                    best_conf = long_conf
                    best_ret = abs(expected_return)
                    reason_detail = f"顺势多单（diff={diff:.2f}不足但趋势配合）"

        if best_side is None:
            return None, f"多空均未通过动态评分 (多: {long_conf:.2f}/{long_score:.1f}, 空: {short_conf:.2f}/{short_score:.1f}, diff={diff:.2f})"

        # 强化技术指标与趋势过滤
        volume_surge = vol_ratio > 2.0
        is_reversal = False
        rev_reason = ""

        if best_side == 'long':
            if slope_15m is not None and slope_15m < -0.001:
                is_reversal, rev_reason = check_trend_reversal(instId, 'long', current_price, ema20_15m, slope_15m)
                if not is_reversal:
                    return None, f"趋势向下且无反转信号，拒绝做多: {rev_reason}"
            if rsi_val > 65 and not is_reversal:
                return None, f"RSI={rsi_val:.1f} 超买区，禁止追多"
            if hist <= -0.0002 and not is_reversal:
                return None, f"MACD柱状线为负({hist:.4f})，动能不足，拒绝做多"
        else:
            if slope_15m is not None and slope_15m > 0.001:
                is_reversal, rev_reason = check_trend_reversal(instId, 'short', current_price, ema20_15m, slope_15m)
                if not is_reversal:
                    return None, f"趋势向上且无反转信号，拒绝做空: {rev_reason}"
            if rsi_val < 35 and not is_reversal:
                return None, f"RSI={rsi_val:.1f} 超卖区，禁止追空"
            if hist >= 0.0002 and not is_reversal:
                return None, f"MACD柱状线为正({hist:.4f})，动能向上，拒绝做空"

        # 趋势强制否决
        if best_side == 'long':
            if is_1h_downtrend:
                return None, f"1小时处于下跌趋势，禁止开多"
            if is_15m_downtrend and not is_reversal:
                return None, f"15分钟处于下跌趋势且无反转，禁止开多"
        else:
            if is_1h_uptrend:
                return None, f"1小时处于上涨趋势，禁止开空"
            if is_15m_uptrend and not is_reversal:
                return None, f"15分钟处于上涨趋势且无反转，禁止开空"

        # 新闻检查
        news_impact, news_title = check_news_impact(instId)
        if news_impact:
            return None, f"新闻影响暂停: {news_title[:50]}"

        anomaly, vol_ratio_anom, anom_reason = check_volume_anomaly(instId, current_price)
        if anomaly:
            return None, f"成交异动拒绝: {anom_reason}"

        reversal, reversal_reason = check_trend_reversal(instId, best_side, current_price, ema20_15m, slope_15m)
        if reversal:
            log(f"✅ 趋势反转确认: {reversal_reason}")

        momentum_ok, momentum_reason = check_price_momentum_filter(instId, best_side, current_price)
        if not momentum_ok:
            return None, f"动量过滤: {momentum_reason}"

        valid, reject_reason = validate_signal(
            signal_type=best_side.upper(),
            symbol=instId,
            current_price=current_price,
            rsi=rsi_val,
            adx=adx,
            atr_pct=atr_pct,
            forecast_values=forecast_values
        )
        if not valid:
            return None, f"信号过滤器拦截 ({best_side.upper()}): {reject_reason}"

        candle = fetch_previous_candle(instId)
        if candle is None:
            price_info = None
        else:
            open_p, high, low, close = candle
            body_top = max(open_p, close)
            body_bottom = min(open_p, close)
            body_len = body_top - body_bottom
            if body_len > 0:
                # 判断当前信号是否顺势
                is_with_trend = (best_side == 'long' and is_15m_uptrend) or (best_side == 'short' and is_15m_downtrend)
                if best_side == 'long':
                    if is_with_trend:
                        long_entry_max = body_top
                        entry_desc = f"顺势多单，建议当前价 ≤ {body_top:.6f} (实体顶部)"
                    else:
                        long_entry_max = body_bottom + body_len * PRICE_POSITION_RATIO
                        entry_desc = f"逆势多单，建议入场价 ≤ {long_entry_max:.6f} (实体底部+{PRICE_POSITION_RATIO*100:.0f}%区域)"
                    short_entry_min = None
                else:  # short
                    if is_with_trend:
                        short_entry_min = body_bottom
                        entry_desc = f"顺势空单，建议当前价 ≥ {body_bottom:.6f} (实体底部)"
                    else:
                        short_entry_min = body_top - body_len * PRICE_POSITION_RATIO
                        entry_desc = f"逆势空单，建议入场价 ≥ {short_entry_min:.6f} (实体顶部-{PRICE_POSITION_RATIO*100:.0f}%区域)"
                    long_entry_max = None
                price_info = {
                    'current_price': current_price,
                    'body_top': body_top,
                    'body_bottom': body_bottom,
                    'long_entry_max': long_entry_max,
                    'short_entry_min': short_entry_min,
                    'entry_desc': entry_desc,
                    'is_with_trend': is_with_trend
                }
            else:
                price_info = None

        hold_minutes = estimate_hold_minutes(forecast_values, current_price, abs(best_ret), best_side)

        result = {
            "symbol": instId,
            "signal": best_side,
            "expected_return": best_ret,
            "r_squared": r_squared,
            "consistency": consistency,
            "direction_confidence": best_conf,
            "score": best_score,
            "last_price": current_price,
            "price_info": price_info,
            "tech_msg": f"动态多空评分 (多{long_score:.1f}/空{short_score:.1f})，选择{best_side}，原因：{reason_detail}",
            "long_score": long_score,
            "short_score": short_score,
            "adx": adx if adx else 0,
            "atr_pct": atr_pct if atr_pct else 0,
            "vol_median_pct": 0,
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
    log(f"🔄 [{now_str}] {BAR}周期 | 预测{HORIZON}步（{HORIZON*5}分钟） | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮")
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
    push_telegram(f"🎯 最终候选池（波动率Top{TOP_N}）: {len(candidates)} 个\n{', '.join(candidates)}")
    log("📈 开始专业评分...")
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
            log(f"      R²: {res['r_squared']:.2f} | 一致性: {res['consistency']:.2f} | 方向置信度: {res['direction_confidence']:.2f} | 得分: {res['score']:.4f}")
            log(f"      技术指标: {res['tech_msg']}")
            log(f"      预估持仓时间: {res.get('estimated_hold_minutes', 15)} 分钟")
        else:
            long_score = 0.0
            short_score = 0.0
            try:
                match_long = re.search(r'多:\s*[\d.]+/([\d.]+)', reason)
                match_short = re.search(r'空:\s*[\d.]+/([\d.]+)', reason)
                if match_long:
                    long_score = float(match_long.group(1))
                if match_short:
                    short_score = float(match_short.group(1))
            except:
                pass

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
    detail_lines.append("🎯 最终候选池（波动率Top{}，含多空评分详情）：".format(TOP_N))
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
        push_telegram("❌ 本轮无高质量交易信号\n\n候选池及过滤原因见上方详情")
        return {}

    df_results = pd.DataFrame(valid).sort_values("score", ascending=False)
    top = df_results.head(FINAL_PICK_N)

    msg = ["✅ 高质量交易信号："]
    for _, row in top.iterrows():
        symbol = row['symbol']
        signal = row['signal'].upper()
        exp_return = row['expected_return'] * 100
        confidence = row['direction_confidence']
        score = row['score']
        price_info = row.get('price_info')
        tech_msg = row['tech_msg']
        hold_min = row.get('estimated_hold_minutes', 15)
        msg.append(f"#{symbol} | {signal}")
        msg.append(f"  预期收益: {exp_return:+.2f}% | 置信度: {confidence:.2f} | 得分: {score:.4f}")
        msg.append(f"  预估持仓时间: {hold_min} 分钟（达到预期收益后平仓）")
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

    # 保存信号时带上方向置信度
    output_dict = {row['symbol']: (row['signal'], row['expected_return'], row.get('estimated_hold_minutes', 15), row['direction_confidence']) for _, row in top.iterrows()}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({k: v[0] for k, v in output_dict.items()}, f, indent=2, ensure_ascii=False)
    return output_dict

# ==================== 9. 交易模块 ====================
class OKXTrader:
    def __init__(self):
        self.exchange = self._init()
        self.strategy_positions = {}
        self.pending_signals = []
        self.last_positions = {}
        self._load_strategy_positions()

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
                    'expected_hold_minutes': info.get('expected_hold_minutes', 15),
                    'half_closed': info.get('half_closed', False)
                }
            with open(STRATEGY_POSITIONS_FILE, 'w') as f:
                json.dump(to_save, f, indent=2)
            log(f"💾 已保存策略持仓 ({len(self.strategy_positions)} 个)")
        except Exception as e:
            err(f"保存策略持仓失败: {e}")

    def _load_strategy_positions(self):
        if not os.path.exists(STRATEGY_POSITIONS_FILE):
            log("📄 无历史策略持仓文件")
            return
        try:
            with open(STRATEGY_POSITIONS_FILE, 'r') as f:
                data = json.load(f)
            for sym, info in data.items():
                self.strategy_positions[sym] = {
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
                    'expected_hold_minutes': info.get('expected_hold_minutes', 15),
                    'half_closed': info.get('half_closed', False)
                }
            log(f"📂 已加载 {len(self.strategy_positions)} 个策略持仓记录")
        except Exception as e:
            err(f"加载策略持仓失败: {e}")

    def _init(self):
        log("🚀 初始化OKX交易客户端...")
        proxies = TG_PROXIES if TG_PROXIES else None
        ex = ccxt.okx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASS,
            "enableRateLimit": True,
            "timeout": 30000,
            "proxies": proxies,
            "options": {"defaultType": "swap"}
        })
        # 调整速率限制，提高并发性能
        ex.rateLimit = 100  # 毫秒，原默认1000ms
        ex.set_sandbox_mode(IS_SANDBOX)
        for attempt in range(3):
            try:
                ex.fetch_balance()
                try:
                    ex.set_position_mode(True)
                    log("✅ 已开启双向持仓模式")
                except Exception as e:
                    err(f"设置双向持仓模式失败（可能已开启）: {e}")
                log("✅ OKX客户端连接成功")
                return ex
            except Exception as e:
                err(f"连接尝试 {attempt+1}/3 失败: {e}")
                if attempt == 2:
                    raise
                time.sleep(2)
        return ex

    def get_account_equity(self):
        try:
            balance = self.exchange.fetch_balance()
            equity = balance.get('USDT', {}).get('total', 0.0)
            if equity == 0:
                free = balance.get('USDT', {}).get('free', 0.0)
                used = balance.get('USDT', {}).get('used', 0.0)
                equity = free + used
            return equity
        except Exception as e:
            err(f"获取账户权益失败: {e}")
            return 0.0

    def get_available_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            return balance.get('USDT', {}).get('free', 0.0)
        except Exception as e:
            err(f"获取余额失败: {e}")
            return 0.0

    def check_balance(self, required_amount):
        available = self.get_available_balance()
        if available < required_amount:
            msg = f"⚠️ 余额不足！可用: {available:.2f} USDT，需要: {required_amount:.2f} USDT，无法开仓。"
            log(msg)
            push_telegram(msg)
            return False
        return True

    def sync_positions(self):
        try:
            all_pos = {}
            positions = self.exchange.fetch_positions()
            for p in positions:
                contracts = float(p.get('contracts', 0))
                if contracts == 0:
                    continue
                pos_side = p.get('info', {}).get('posSide')
                if pos_side in ['long', 'short']:
                    side = pos_side
                else:
                    side = 'long' if contracts > 0 else 'short'
                all_pos[p['symbol']] = side
            return all_pos
        except Exception as e:
            err(f"同步持仓失败: {e}")
            return {}

    def sync_strategy_positions_with_exchange(self):
        try:
            actual = self.sync_positions()
            for sym, side in actual.items():
                if sym in self.strategy_positions:
                    continue
                matched = False
                for sig in self.pending_signals:
                    if (sig['ccxt_symbol'] == sym or sig['raw_symbol'] == sym) and sig['side'] == side:
                        matched = True
                        break
                if not matched:
                    continue
                pos = None
                for p in self.exchange.fetch_positions():
                    if p['symbol'] == sym:
                        pos = p
                        break
                if pos:
                    open_price = float(pos.get('entryPrice', 0))
                    contracts = float(pos.get('contracts', 0))
                    margin = float(pos.get('margin', 0))
                    if margin == 0 and contracts != 0 and open_price != 0:
                        market = self.exchange.market(sym)
                        contract_size = float(market.get('contractSize', 1.0))
                        nominal = contracts * open_price * contract_size
                        margin = nominal / LEVERAGE
                    if open_price != 0 and contracts != 0:
                        self.strategy_positions[sym] = {
                            'side': side,
                            'open_price': open_price,
                            'open_time': time.time(),
                            'open_qty': contracts,
                            'open_margin': margin,
                            'open_nominal': contracts * open_price,
                            'stop_loss_price': None,
                            'highest_price': open_price,
                            'lowest_price': open_price,
                            'trailing_stop_pct': TRAILING_STOP_PCT,
                            'trailing_activated': False,
                            'expected_return': None,
                            'expected_met': False,
                            'expected_hold_minutes': 15,
                            'half_closed': False
                        }
                        self._save_strategy_positions()
                        log(f"🔄 接管孤儿持仓: {sym} {side.upper()} 已纳入管理")
                        push_telegram(f"🔄 策略接管持仓: {sym} {side.upper()}，现由程序自动管理")
                        self.pending_signals = [sig for sig in self.pending_signals if sig['ccxt_symbol'] != sym and sig['raw_symbol'] != sym]
        except Exception as e:
            err(f"同步策略持仓异常: {e}")

    def sync_position_sides(self, pos_map):
        try:
            for sym, pos in pos_map.items():
                if sym in self.strategy_positions:
                    pos_side = pos.get('info', {}).get('posSide')
                    if pos_side in ['long', 'short']:
                        self.strategy_positions[sym]['side'] = pos_side
                    else:
                        contracts = float(pos.get('contracts', 0))
                        if contracts > 0:
                            self.strategy_positions[sym]['side'] = 'long'
                        elif contracts < 0:
                            self.strategy_positions[sym]['side'] = 'short'
        except:
            pass

    def set_leverage(self, symbol, leverage=LEVERAGE):
        try:
            body = {
                "instId": symbol,
                "lever": str(leverage),
                "mgnMode": "isolated"
            }
            timestamp = str(int(time.time() * 1000))
            method = 'POST'
            request_path = '/api/v5/account/set-leverage'
            body_json = json.dumps(body)
            message = timestamp + method + request_path + body_json
            signature = base64.b64encode(
                hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
            ).decode('utf-8')
            headers = {
                'OK-ACCESS-KEY': API_KEY,
                'OK-ACCESS-SIGN': signature,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': API_PASS,
                'Content-Type': 'application/json'
            }
            url = "https://www.okx.com" + request_path
            response = requests.post(url, headers=headers, data=body_json, timeout=5)
            result = response.json()
            if result.get('code') == '0':
                log(f"设置杠杆 {symbol} {leverage}x 逐仓成功")
            else:
                err(f"设置杠杆失败 {symbol}: {result}")
        except Exception as e:
            err(f"设置杠杆异常 {symbol}: {e}")

    def get_atr(self, symbol, period=ATR_PERIOD):
        try:
            df = fetch_klines_with_retry(symbol, BAR, period+1)
            if df is None or len(df) < period+1:
                return None
            high = df['h'].values
            low = df['l'].values
            close = df['c'].values
            tr = np.maximum(high[1:] - low[1:],
                            np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1]))
            atr = np.mean(tr[-period:])
            return atr
        except Exception as e:
            err(f"计算ATR失败 {symbol}: {e}")
            return None

    def calculate_contracts_with_adjustment(self, symbol, base_margin_usdt, leverage=LEVERAGE):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            market = self.exchange.market(symbol)
            contract_size = float(market.get('contractSize', 1.0))
            min_amount = None
            if 'limits' in market and 'amount' in market['limits'] and 'min' in market['limits']['amount']:
                min_amount = float(market['limits']['amount']['min'])
            nominal = base_margin_usdt * leverage
            contracts = nominal / (price * contract_size)
            adjusted_margin = base_margin_usdt
            is_adjusted = False
            if min_amount is not None and contracts < min_amount:
                required_nominal = min_amount * price * contract_size
                required_margin = required_nominal / leverage
                if required_margin <= base_margin_usdt * MAX_MARGIN_MULTIPLIER:
                    adjusted_margin = required_margin
                    contracts = min_amount
                    is_adjusted = True
                    log(f"⚠️ 合约 {symbol} 最小张数 {min_amount}, 原保证金 {base_margin_usdt:.2f} 不足, 自动调整为 {adjusted_margin:.2f} USDT")
                    push_telegram(f"⚠️ {symbol} 合约最小张数限制，保证金已自动从 {base_margin_usdt} 上调至 {adjusted_margin:.2f} USDT")
                else:
                    err(f"合约 {symbol} 最小张数 {min_amount} 需要保证金 {required_margin:.2f} USDT，超出最大允许 {base_margin_usdt * MAX_MARGIN_MULTIPLIER:.2f}，放弃开仓")
                    return None, None, None, False
            amount_prec = self.exchange.amount_to_precision(symbol, contracts)
            actual_contracts = float(amount_prec)
            actual_nominal = actual_contracts * price * contract_size
            actual_margin = actual_nominal / leverage
            if abs(actual_margin - adjusted_margin) > 0.01:
                adjusted_margin = actual_margin
            log(f"合约计算: 原保证金 {base_margin_usdt:.2f} -> 实际保证金 {adjusted_margin:.2f}, 张数 {actual_contracts}, 价格 {price:.4f}")
            return adjusted_margin, actual_contracts, price, is_adjusted
        except Exception as e:
            err(f"计算合约张数失败 {symbol}: {e}")
            return None, None, None, False

    def check_price_position_entity(self, symbol, side, is_with_trend=False):
        try:
            candle = fetch_previous_candle(symbol)
            if candle is None:
                return False, "无法获取上一根K线数据"
            open_p, high, low, close = candle
            body_top = max(open_p, close)
            body_bottom = min(open_p, close)
            body_len = body_top - body_bottom
            if body_len <= 0:
                return False, "实体长度为零"
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            if side == 'long':
                if is_with_trend:
                    max_allowed = body_top
                else:
                    max_allowed = body_bottom + body_len * PRICE_POSITION_RATIO
                if current_price <= max_allowed:
                    return True, f"价格位置满足多单条件: 当前{current_price:.6f} ≤ {max_allowed:.6f}"
                else:
                    return False, f"价格位置不满足多单条件: 当前{current_price:.6f} > {max_allowed:.6f}"
            else:
                if is_with_trend:
                    min_allowed = body_bottom
                else:
                    min_allowed = body_top - body_len * PRICE_POSITION_RATIO
                if current_price >= min_allowed:
                    return True, f"价格位置满足空单条件: 当前{current_price:.6f} ≥ {min_allowed:.6f}"
                else:
                    return False, f"价格位置不满足空单条件: 当前{current_price:.6f} < {min_allowed:.6f}"
        except Exception as e:
            err(f"检查价格位置异常 {symbol}: {e}")
            return False, f"检查异常: {e}"

    def check_favorable_move(self, signal_price, side, current_price):
        if side == 'long':
            pct_change = (signal_price - current_price) / signal_price * 100
            return pct_change >= FAVORABLE_MOVE_PCT, pct_change
        else:
            pct_change = (current_price - signal_price) / signal_price * 100
            return pct_change >= FAVORABLE_MOVE_PCT, pct_change

    def set_pending_signals(self, signals_list):
        """
        signals_list: list of (raw_symbol, side, expected_return, hold_minutes, margin_amount)
        """
        self.pending_signals = []
        for raw_symbol, side, expected_return, hold_minutes, margin_amount in signals_list:
            try:
                market = self.exchange.market(raw_symbol)
                ccxt_symbol = market['symbol']
                ticker = self.exchange.fetch_ticker(raw_symbol)
                signal_price = ticker['last']
                self.pending_signals.append({
                    'raw_symbol': raw_symbol,
                    'ccxt_symbol': ccxt_symbol,
                    'side': side,
                    'margin': margin_amount,
                    'signal_price': signal_price,
                    'expected_return': expected_return,
                    'expected_hold_minutes': hold_minutes
                })
            except Exception as e:
                err(f"获取市场信息失败 {raw_symbol}: {e}")
        log(f"📋 设置待开仓信号: {len(self.pending_signals)} 个")

    def check_pre_open_safety(self, symbol, side, current_price, is_uptrend, is_downtrend):
        is_with_trend = (side == 'long' and is_uptrend) or (side == 'short' and is_downtrend)
        if is_with_trend:
            return True, "顺势信号，跳过安全检查"

        df_1h = fetch_klines_with_retry(symbol, SAFETY_TREND_BAR, 50)
        if df_1h is not None and len(df_1h) >= 30:
            closes_1h = df_1h['c'].astype(float)
            ema20_1h = closes_1h.ewm(span=20, adjust=False).mean().iloc[-1]
            if ema20_1h != 0:
                deviation_pct = abs((current_price - ema20_1h) / ema20_1h) * 100
                if deviation_pct > MAX_DEVIATION_FROM_EMA_PCT:
                    return False, f"逆势开仓但价格偏离1小时EMA20达 {deviation_pct:.2f}% > {MAX_DEVIATION_FROM_EMA_PCT}%，可能处于极端位置"

        df_15m = fetch_klines_with_retry(symbol, HIGHER_BAR, 20)
        if df_15m is not None and len(df_15m) >= 20:
            last_candle = df_15m.iloc[-1]
            body = abs(last_candle['c'] - last_candle['o'])
            prev_bodies = [abs(df_15m.iloc[-i-2]['c'] - df_15m.iloc[-i-2]['o']) for i in range(5)]
            avg_body = np.mean(prev_bodies) if prev_bodies else body
            if avg_body > 0:
                body_ratio = body / avg_body
                if body_ratio > MAX_CANDLE_BODY_RATIO:
                    return False, f"逆势开仓但15分钟K线实体是平均的 {body_ratio:.1f} 倍 > {MAX_CANDLE_BODY_RATIO}，可能为爆发行情"

            volumes_15m = df_15m['v'].astype(float)
            avg_vol = volumes_15m.iloc[-6:-1].mean()
            current_vol = volumes_15m.iloc[-1]
            if avg_vol > 0 and current_vol / avg_vol > MAX_VOLUME_SPIKE_RATIO:
                return False, f"逆势开仓但成交量突增 {current_vol/avg_vol:.1f} 倍 > {MAX_VOLUME_SPIKE_RATIO}，可能为消息驱动"

        if side == 'long':
            _, _, is_1h_downtrend, _ = get_1h_trend(symbol)
            _, slope_15m = get_15min_trend(symbol)
            is_15m_downtrend = slope_15m is not None and slope_15m < -0.001
            if is_1h_downtrend or is_15m_downtrend:
                return False, "逆势多单被大周期下跌趋势否决"
        else:
            _, _, _, is_1h_uptrend = get_1h_trend(symbol)
            _, slope_15m = get_15min_trend(symbol)
            is_15m_uptrend = slope_15m is not None and slope_15m > 0.001
            if is_1h_uptrend or is_15m_uptrend:
                return False, "逆势空单被大周期上涨趋势否决"

        return True, "安全检查通过"

    def check_and_open_pending(self):
        if not self.pending_signals:
            return
        to_remove = []
        for idx, sig in enumerate(self.pending_signals):
            raw_symbol = sig['raw_symbol']
            side = sig['side']
            margin = sig['margin']
            signal_price = sig['signal_price']
            expected_return = sig['expected_return']
            hold_minutes = sig.get('expected_hold_minutes', 15)
            try:
                ticker = self.exchange.fetch_ticker(raw_symbol)
                current_price = ticker['last']
            except:
                log(f"❌ 无法获取 {raw_symbol} 当前价格")
                continue
            ccxt_symbol = self.exchange.market(raw_symbol)['symbol']
            current_positions = self.sync_positions()
            if ccxt_symbol in current_positions:
                if current_positions[ccxt_symbol] == side:
                    log(f"⏸️ {raw_symbol} 已有同向持仓，取消待开仓信号")
                    to_remove.append(idx)
                else:
                    log(f"⚠️ {raw_symbol} 已有反向持仓 {current_positions[ccxt_symbol]}，信号 {side} 仍可开仓")
                continue

            ema20_15m, slope_15m = get_15min_trend(raw_symbol)
            is_uptrend = slope_15m is not None and slope_15m > 0.002
            is_downtrend = slope_15m is not None and slope_15m < -0.002
            is_with_trend = (side == 'long' and is_uptrend) or (side == 'short' and is_downtrend)

            df_last = fetch_klines_with_retry(raw_symbol, BAR, 2)
            is_breakout = False
            if df_last is not None and len(df_last) >= 2:
                prev_candle = df_last.iloc[-2]
                prev_high = prev_candle['h']
                prev_low = prev_candle['l']
                is_breakout = (side == 'long' and current_price > prev_high) or (side == 'short' and current_price < prev_low)

            if is_breakout and is_with_trend:
                log(f"🚀 顺势突破开仓 {raw_symbol} {side.upper()} 价格 {current_price}")
                success = self.open_position(raw_symbol, side, margin, expected_return, hold_minutes, ignore_price_position=True, is_with_trend=True)
                if success:
                    to_remove.append(idx)
                else:
                    log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
                continue

            favorable, pct_change = self.check_favorable_move(signal_price, side, current_price)
            if favorable:
                log(f"🚀 价格向有利方向移动 {pct_change:.2f}% ≥ {FAVORABLE_MOVE_PCT}%，立即开仓 {raw_symbol} {side.upper()}")
                success = self.open_position(raw_symbol, side, margin, expected_return, hold_minutes, ignore_price_position=True, is_with_trend=is_with_trend)
                if success:
                    to_remove.append(idx)
                else:
                    log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
            else:
                ok, msg = self.check_price_position_entity(raw_symbol, side, is_with_trend=is_with_trend)
                if ok:
                    log(f"🚀 待开仓信号价格满足实体位置条件: {raw_symbol} {side.upper()}")
                    success = self.open_position(raw_symbol, side, margin, expected_return, hold_minutes, ignore_price_position=False, is_with_trend=is_with_trend)
                    if success:
                        to_remove.append(idx)
                    else:
                        log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
                else:
                    log(f"⏸️ 待开仓信号 {raw_symbol} {side.upper()} 价格不满足: {msg}")
        for idx in sorted(to_remove, reverse=True):
            self.pending_signals.pop(idx)

    def open_position(self, symbol, side, base_margin_usdt, expected_return, expected_hold_minutes, ignore_price_position=False, is_with_trend=False):
        current_positions = self.sync_positions()
        ccxt_symbol = self.exchange.market(symbol)['symbol']
        if ccxt_symbol in current_positions:
            log(f"⏸️ {symbol} 已有持仓 {current_positions[ccxt_symbol]}，拒绝重复开仓")
            return False

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except:
            log(f"❌ 无法获取 {symbol} 当前价格")
            return False

        emergency, move_pct = check_emergency_move(symbol, current_price)
        if emergency:
            ignore_price_position = True
            log(f"🚨 检测到紧急价格变动 {move_pct:.1f}%，将忽略实体位置检查直接开仓")

        if not ignore_price_position:
            ok, msg = self.check_price_position_entity(symbol, side, is_with_trend=is_with_trend)
            if not ok:
                log(f"⏸️ 跳过开仓 {symbol} {side}: {msg}")
                return False
        else:
            log(f"🚀 忽略实体位置检查，因有利移动或紧急变动触发开仓 {symbol} {side}")

        ema20_15m, slope_15m = get_15min_trend(symbol)
        is_uptrend = slope_15m is not None and slope_15m > 0.002
        is_downtrend = slope_15m is not None and slope_15m < -0.002
        safe, reason = self.check_pre_open_safety(symbol, side, current_price, is_uptrend, is_downtrend)
        if not safe:
            log(f"⛔ 开仓前安全检查拒绝 {symbol} {side}: {reason}")
            push_telegram(f"⛔ 开仓前安全检查拒绝 {symbol} {side}: {reason}")
            return False
        else:
            log(f"✅ 开仓前安全检查通过: {reason}")

        df_volatility = fetch_klines_with_retry(symbol, BAR, 30)
        if df_volatility is not None and len(df_volatility) >= 20:
            vol_profile, atr_pct = detect_volatility_profile(df_volatility)
            adaptive_params = get_adaptive_trading_params(BAR, vol_profile)
            atr_multiplier = adaptive_params["atr_multiplier"]
            stop_loss_pct = adaptive_params["stop_loss_pct"]
            trailing_stop_pct = adaptive_params["trailing_stop_pct"]
            log(f"自适应参数: 波动率等级={vol_profile} ({atr_pct:.2f}%), ATR倍数={atr_multiplier}, 固定止损={stop_loss_pct}%, 跟踪回撤={trailing_stop_pct}%")
        else:
            atr_multiplier = ATR_MULTIPLIER
            stop_loss_pct = STOP_LOSS_PCT
            trailing_stop_pct = TRAILING_STOP_PCT

        adjusted_margin, amount, price, is_adjusted = self.calculate_contracts_with_adjustment(symbol, base_margin_usdt, LEVERAGE)
        if amount is None or price is None:
            push_telegram(f"❌ 无法计算有效张数或调整失败: {symbol}")
            return False

        if not self.check_balance(adjusted_margin):
            return False

        if len(self.strategy_positions) >= MAX_CONCURRENT_POSITIONS:
            log(f"⏸️ 已达最大持仓数 {MAX_CONCURRENT_POSITIONS}，拒绝开仓 {symbol}")
            push_telegram(f"⚠️ 已达最大持仓数 {MAX_CONCURRENT_POSITIONS}，无法开仓 {symbol}")
            return False
        equity = self.get_account_equity()
        total_margin_used = sum(p['open_margin'] for p in self.strategy_positions.values())
        if total_margin_used + adjusted_margin > equity * MAX_TOTAL_MARGIN_RATIO:
            log(f"⚠️ 总保证金占用将超过 {MAX_TOTAL_MARGIN_RATIO*100}% 账户权益，拒绝开仓")
            push_telegram(f"⚠️ 保证金占用超限，当前占用 {total_margin_used:.2f} U，权益 {equity:.2f} U")
            return False

        try:
            self.set_leverage(symbol)
            if side == 'long':
                order_side = 'buy'
                position_side = 'long'
            else:
                order_side = 'sell'
                position_side = 'short'

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=amount,
                params={
                    'positionSide': position_side,
                    'tdMode': 'isolated'
                }
            )

            actual_open_price = order.get('average', price)
            actual_filled = order.get('filled', amount)
            if actual_filled == 0:
                info = order.get('info', {})
                actual_filled = float(info.get('filledSz', 0))
                actual_open_price = float(info.get('avgPx', price))
            if actual_filled == 0:
                err(f"订单未成交: {order}")
                time.sleep(1)
                current_positions = self.sync_positions()
                if ccxt_symbol in current_positions:
                    log(f"⚠️ 订单未返回成交但发现已有持仓 {ccxt_symbol}，尝试补救")
                    pos = None
                    for p in self.exchange.fetch_positions():
                        if p['symbol'] == ccxt_symbol:
                            pos = p
                            break
                    if pos:
                        actual_open_price = float(pos.get('entryPrice', price))
                        actual_filled = float(pos.get('contracts', 0))
                        actual_margin = float(pos.get('margin', 0))
                        if actual_margin == 0 and actual_filled != 0 and actual_open_price != 0:
                            market = self.exchange.market(ccxt_symbol)
                            contract_size = float(market.get('contractSize', 1.0))
                            nominal = actual_filled * actual_open_price * contract_size
                            actual_margin = nominal / LEVERAGE
                        actual_nominal = actual_filled * actual_open_price
                        atr = self.get_atr(symbol)
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
                        self.strategy_positions[ccxt_symbol] = {
                            'side': side,
                            'open_price': actual_open_price,
                            'open_time': time.time(),
                            'open_qty': actual_filled,
                            'open_margin': actual_margin,
                            'open_nominal': actual_nominal,
                            'stop_loss_price': stop_loss_price,
                            'highest_price': actual_open_price,
                            'lowest_price': actual_open_price,
                            'trailing_stop_pct': trailing_stop_pct,
                            'trailing_activated': False,
                            'expected_return': expected_return,
                            'expected_met': False,
                            'expected_hold_minutes': expected_hold_minutes,
                            'half_closed': False
                        }
                        self._save_strategy_positions()
                        log(f"✅ 补救记录持仓 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_margin:.2f} | 止损: {stop_loss_price:.4f}")
                        push_telegram(f"✅ 策略开仓成功（补救）\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_margin:.2f}\n动态止损: ${stop_loss_price:.4f}\n预期持仓: {expected_hold_minutes} 分钟")
                        return True
                return False

            actual_nominal = actual_filled * actual_open_price
            actual_used_margin = actual_nominal / LEVERAGE

            atr = self.get_atr(symbol)
            if atr is not None:
                if side == 'long':
                    stop_loss_price = actual_open_price - atr * atr_multiplier
                else:
                    stop_loss_price = actual_open_price + atr * atr_multiplier
                log(f"ATR={atr:.4f}, 动态止损价={stop_loss_price:.4f}")
            else:
                if side == 'long':
                    stop_loss_price = actual_open_price * (1 - stop_loss_pct / 100)
                else:
                    stop_loss_price = actual_open_price * (1 + stop_loss_pct / 100)
                log(f"ATR计算失败，使用固定止损 {stop_loss_pct}% -> {stop_loss_price:.4f}")

            self.strategy_positions[ccxt_symbol] = {
                'side': side,
                'open_price': actual_open_price,
                'open_time': time.time(),
                'open_qty': actual_filled,
                'open_margin': actual_used_margin,
                'open_nominal': actual_nominal,
                'stop_loss_price': stop_loss_price,
                'highest_price': actual_open_price,
                'lowest_price': actual_open_price,
                'trailing_stop_pct': trailing_stop_pct,
                'trailing_activated': False,
                'expected_return': expected_return,
                'expected_met': False,
                'expected_hold_minutes': expected_hold_minutes,
                'half_closed': False
            }
            self._save_strategy_positions()
            log(f"✅ 开仓成功 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_used_margin:.2f} | 止损: {stop_loss_price:.4f}")
            msg = f"✅ 策略开仓成功\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_used_margin:.2f}\n动态止损: ${stop_loss_price:.4f}\n预期收益: {expected_return*100:.2f}%\n预期持仓: {expected_hold_minutes} 分钟"
            if is_adjusted:
                msg += f"\n⚠️ 保证金已从 {base_margin_usdt} 自动上调至 {adjusted_margin:.2f} USDT（满足最小张数）"
            for _ in range(3):
                if push_telegram(msg):
                    break
                time.sleep(1)
            return True
        except Exception as e:
            err(f"开仓失败 {symbol} {side}: {e}")
            time.sleep(1)
            current_positions = self.sync_positions()
            if ccxt_symbol in current_positions:
                log(f"⚠️ 开仓异常但发现已有持仓 {ccxt_symbol}，尝试补救")
                pos = None
                for p in self.exchange.fetch_positions():
                    if p['symbol'] == ccxt_symbol:
                        pos = p
                        break
                if pos:
                    actual_open_price = float(pos.get('entryPrice', 0))
                    actual_filled = float(pos.get('contracts', 0))
                    actual_margin = float(pos.get('margin', 0))
                    if actual_margin == 0 and actual_filled != 0 and actual_open_price != 0:
                        market = self.exchange.market(ccxt_symbol)
                        contract_size = float(market.get('contractSize', 1.0))
                        nominal = actual_filled * actual_open_price * contract_size
                        actual_margin = nominal / LEVERAGE
                    if actual_filled != 0:
                        atr = self.get_atr(symbol)
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
                        self.strategy_positions[ccxt_symbol] = {
                            'side': side,
                            'open_price': actual_open_price,
                            'open_time': time.time(),
                            'open_qty': actual_filled,
                            'open_margin': actual_margin,
                            'open_nominal': actual_filled * actual_open_price,
                            'stop_loss_price': stop_loss_price,
                            'highest_price': actual_open_price,
                            'lowest_price': actual_open_price,
                            'trailing_stop_pct': trailing_stop_pct,
                            'trailing_activated': False,
                            'expected_return': expected_return,
                            'expected_met': False,
                            'expected_hold_minutes': expected_hold_minutes,
                            'half_closed': False
                        }
                        self._save_strategy_positions()
                        log(f"✅ 异常后补救记录持仓 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_margin:.2f} | 止损: {stop_loss_price:.4f}")
                        push_telegram(f"✅ 策略开仓成功（异常补救）\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_margin:.2f}\n动态止损: ${stop_loss_price:.4f}\n预期持仓: {expected_hold_minutes} 分钟")
                        return True
            if "insufficient" in str(e).lower() or "balance" in str(e).lower():
                push_telegram(f"❌ 开仓失败，余额不足: {symbol} {side}")
            return False

    def close_position(self, ccxt_symbol, reason=""):
        if ccxt_symbol not in self.strategy_positions:
            log(f"⏭️ {ccxt_symbol} 不是策略持仓，跳过平仓")
            return False

        pos_info = self.strategy_positions[ccxt_symbol]
        side = pos_info['side']
        open_price = pos_info['open_price']
        open_time = pos_info['open_time']
        open_qty = pos_info['open_qty']
        open_margin = pos_info['open_margin']
        hold_seconds = time.time() - open_time

        try:
            order_side = 'sell' if side == 'long' else 'buy'
            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=order_side,
                amount=abs(open_qty),
                params={
                    'reduceOnly': True,
                    'posSide': side.capitalize()
                }
            )
            close_price = order.get('average', 0)
            if close_price == 0:
                ticker = self.exchange.fetch_ticker(ccxt_symbol)
                close_price = ticker['last']
            pnl_usdt, pnl_percent = self._calc_pnl(side, open_price, close_price, open_qty, open_margin)
            del self.strategy_positions[ccxt_symbol]
            self._save_strategy_positions()
            self._log_close(ccxt_symbol, side, open_price, close_price, open_qty, open_margin, pnl_usdt, pnl_percent, hold_seconds, reason)
            return True
        except Exception as e:
            err(f"平仓失败 {ccxt_symbol}: {e}")
            try:
                positions = self.exchange.fetch_positions()
                for p in positions:
                    if p['symbol'] == ccxt_symbol and float(p.get('contracts', 0)) != 0:
                        actual_qty = abs(float(p['contracts']))
                        order_side = 'sell' if side == 'long' else 'buy'
                        self.exchange.create_order(
                            symbol=ccxt_symbol,
                            type='market',
                            side=order_side,
                            amount=actual_qty,
                            params={'reduceOnly': True, 'posSide': side.capitalize()}
                        )
                        del self.strategy_positions[ccxt_symbol]
                        self._save_strategy_positions()
                        log(f"✅ 备用平仓成功 {ccxt_symbol}")
                        return True
                    else:
                        del self.strategy_positions[ccxt_symbol]
                        self._save_strategy_positions()
                        log(f"⚠️ 交易所无 {ccxt_symbol} 持仓，已清理本地记录")
                        return True
            except Exception as e2:
                err(f"备用平仓也失败 {ccxt_symbol}: {e2}")
            return False

    def _half_close_position(self, ccxt_symbol, reason=""):
        if ccxt_symbol not in self.strategy_positions:
            return False
        info = self.strategy_positions[ccxt_symbol]
        current_qty = info['open_qty']
        half_qty = current_qty / 2.0
        if half_qty < 0.01:
            return False
        try:
            order_side = 'sell' if info['side'] == 'long' else 'buy'
            self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=order_side,
                amount=half_qty,
                params={'reduceOnly': True, 'posSide': info['side'].capitalize()}
            )
            info['open_qty'] -= half_qty
            ratio = info['open_qty'] / (info['open_qty'] + half_qty)
            info['open_margin'] *= ratio
            info['half_closed'] = True
            self._save_strategy_positions()
            push_telegram(f"✂️ 部分平仓 {ccxt_symbol} {info['side']}\n数量: {half_qty} 张\n原因: {reason}")
            return True
        except Exception as e:
            err(f"部分平仓失败: {e}")
            return False

    def _calc_pnl(self, side, open_price, close_price, open_qty, open_margin):
        if side == 'long':
            pnl_usdt = (close_price - open_price) * open_qty
        else:
            pnl_usdt = (open_price - close_price) * open_qty
        pnl_percent = (pnl_usdt / open_margin) * 100 if open_margin > 0 else 0
        return pnl_usdt, pnl_percent

    def _log_close(self, symbol, side, open_price, close_price, open_qty, open_margin, pnl_usdt, pnl_percent, hold_seconds, reason):
        reason_text = f"原因: {reason}" if reason else ""
        push_telegram(f"🔻 策略平仓\n币种: {symbol}\n方向: {side.upper()}\n开仓价: ${open_price:.4f}\n平仓价: ${close_price:.4f}\n盈亏金额: ${pnl_usdt:+.2f}\n盈亏百分比(保证金): {pnl_percent:+.2f}%\n持仓时长: {hold_seconds/60:.1f}分钟\n{reason_text}")
        log(f"平仓记录 {symbol} | 盈亏: {pnl_usdt:+.2f} USDT ({pnl_percent:+.2f}%) | {reason}")

    def close_all(self):
        log("🔻 平仓全部策略持仓")
        for sym in list(self.strategy_positions.keys()):
            self.close_position(sym, reason="清仓")
        self.strategy_positions = {}
        self._save_strategy_positions()

    def check_and_close_positions(self):
        closed_any = False
        try:
            all_positions = self.exchange.fetch_positions()
        except Exception as e:
            err(f"获取持仓列表失败: {e}")
            return False

        pos_map = {}
        for p in all_positions:
            contracts = float(p.get('contracts', 0))
            if contracts != 0:
                pos_map[p['symbol']] = p

        self.sync_strategy_positions_with_exchange()
        self.sync_position_sides(pos_map)

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
                    ticker = self.exchange.fetch_ticker(sym)
                    current_price = ticker['last']
                except:
                    log(f"⚠️ 无法获取 {sym} 最新价格，跳过本次检查")
                    continue

            pnl_percent = float(pos.get('percentage', 0))
            if abs(pnl_percent) < 1:
                pnl_percent = pnl_percent * 100

            hold_seconds = time.time() - info['open_time']
            expected_hold_minutes = info.get('expected_hold_minutes', 15)

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

            # 跟踪止损：浮盈 > 3% 自动激活
            trailing_pct = info.get('trailing_stop_pct', TRAILING_STOP_PCT)
            if pnl_percent > 3.0:
                info['trailing_activated'] = True
            if info.get('trailing_activated', False) and drawdown_pct >= trailing_pct:
                log(f"📉 触发跟踪止损: {sym} 回撤 {drawdown_pct:.2f}%")
                self.close_position(sym, reason=f"跟踪止损回撤{drawdown_pct:.1f}%")
                closed_any = True
                continue

            # 超时分级处理（仅对非接管仓位有效，接管仓位 expected_hold_minutes 极大）
            if hold_seconds >= expected_hold_minutes * 60:
                if pnl_percent > 20.0 and not info.get('half_closed', False):
                    self._half_close_position(sym, reason=f"超时浮盈{pnl_percent:.1f}%减半")
                    info['half_closed'] = True
                    info['trailing_activated'] = True
                    continue
                elif pnl_percent > 10.0:
                    info['trailing_activated'] = True
                    log(f"⏰ 超时浮盈{pnl_percent:.1f}%，继续跟踪")
                elif pnl_percent > 0:
                    log(f"⏰ 超时小盈 {pnl_percent:.1f}%，平仓落袋")
                    self.close_position(sym, reason="超时止盈")
                    closed_any = True
                    continue
                else:
                    log(f"⏰ 超时亏损 {pnl_percent:.1f}%，止损平仓")
                    self.close_position(sym, reason="超时止损")
                    closed_any = True
                    continue

            # 初始动态止损（仅当未禁用且未超过超时）
            stop_price = info.get('stop_loss_price')
            if stop_price is not None:
                if (info['side'] == 'long' and current_price <= stop_price) or \
                   (info['side'] == 'short' and current_price >= stop_price):
                    log(f"💥 触发初始止损: {sym} 当前价 {current_price:.4f} 触及止损价 {stop_price:.4f}")
                    self.close_position(sym, reason=f"初始止损 {stop_price:.4f}")
                    closed_any = True
                    continue

            log(f"📉 检查持仓 {sym}: 盈亏 {pnl_percent:.2f}%, 跟踪激活: {info.get('trailing_activated', False)}, 持仓时长 {hold_seconds/60:.1f}/{expected_hold_minutes} 分钟")

        return closed_any

    def check_manual_close(self):
        actual_positions = self.sync_positions()
        closed_by_manual = []
        for sym in list(self.strategy_positions.keys()):
            if sym not in actual_positions:
                closed_by_manual.append(sym)
        for sym in closed_by_manual:
            info = self.strategy_positions[sym]
            side = info['side']
            open_price = info['open_price']
            open_margin = info['open_margin']
            open_qty = info['open_qty']
            try:
                ticker = self.exchange.fetch_ticker(sym)
                current_price = ticker['last']
            except:
                current_price = 0
            if side == 'long':
                pnl_usdt = (current_price - open_price) * open_qty if current_price else 0
            else:
                pnl_usdt = (open_price - current_price) * open_qty if current_price else 0
            pnl_percent = (pnl_usdt / open_margin) * 100 if open_margin > 0 else 0
            push_telegram(f"🔻 人工平仓（策略持仓）\n币种: {sym}\n方向: {side.upper()}\n开仓价: ${open_price:.4f}\n估算平仓价: ${current_price:.4f}\n估算盈亏: ${pnl_usdt:+.2f} ({pnl_percent:+.2f}%)")
            log(f"人工平仓检测: {sym} 策略持仓已不存在，清理记录")
            del self.strategy_positions[sym]
        if closed_by_manual:
            self._save_strategy_positions()
            self.last_positions = self.sync_positions()

    def check_manual_open(self):
        """检查并自动接管所有未被策略管理的持仓（包括人工开仓和策略开仓但记录丢失的情况）"""
        current_positions = self.sync_positions()
        for sym, side in current_positions.items():
            if sym not in self.strategy_positions:
                log(f"⚠️ 发现未管理持仓 {sym} {side.upper()}，自动接管")
                self._adopt_position(sym, side)
                push_telegram(f"🔄 策略自动接管持仓 {sym} {side.upper()}（原人工或记录丢失）")
        self.last_positions = current_positions.copy()

    def _adopt_position(self, sym, side):
        """接管一个现有的持仓，不设初始止损，只依靠跟踪止盈（基于原始开仓价），超时不平仓"""
        try:
            pos = None
            for p in self.exchange.fetch_positions():
                if p['symbol'] == sym:
                    pos = p
                    break
            if not pos:
                log(f"❌ 无法获取 {sym} 的持仓详情，接管失败")
                return

            open_price = float(pos.get('entryPrice', 0))
            contracts = float(pos.get('contracts', 0))
            margin = float(pos.get('margin', 0))
            if margin == 0 and contracts != 0 and open_price != 0:
                market = self.exchange.market(sym)
                contract_size = float(market.get('contractSize', 1.0))
                nominal = contracts * open_price * contract_size
                margin = nominal / LEVERAGE

            ticker = self.exchange.fetch_ticker(sym)
            current_price = ticker['last']

            # 接管仓位：不设初始止损，只靠跟踪止盈；超时时间设为极大值，永不超时平仓
            self.strategy_positions[sym] = {
                'side': side,
                'open_price': open_price,
                'open_time': time.time(),
                'open_qty': contracts,
                'open_margin': margin,
                'open_nominal': contracts * open_price,
                'stop_loss_price': None,                     # 无初始止损
                'highest_price': open_price,
                'lowest_price': open_price,
                'trailing_stop_pct': TRAILING_STOP_PCT,
                'trailing_activated': False,
                'expected_return': 0.003,                    # 默认预期收益0.3%，仅用于超时判断（但超时时间极大）
                'expected_met': False,
                'expected_hold_minutes': 999999,             # 永不超时
                'half_closed': False
            }
            self._save_strategy_positions()
            log(f"✅ 成功接管持仓 {sym} {side.upper()} {contracts} 张 @ {open_price:.4f} | 保证金: ${margin:.2f} | 无初始止损，仅跟踪止盈，永不超时平仓")
            push_telegram(f"✅ 策略接管持仓 {sym} {side.upper()} (仅跟踪止盈，不自损)\n开仓价: {open_price:.4f}\n当前价: {current_price:.4f}\n数量: {contracts} 张\n保证金: ${margin:.2f}")
        except Exception as e:
            err(f"接管持仓失败 {sym}: {e}")

    def clear_pending_signals(self):
        self.pending_signals = []

# ==================== 10. 主程序 ====================
def main():
    with open("/tmp/trading_bot.pid", "w") as f:
        f.write(str(os.getpid()))
    
    trader = OKXTrader()
    last_pred = datetime.now() - timedelta(seconds=PREDICTION_INTERVAL)
    has_set_pending_this_cycle = False

    log("\n========== 全自动交易系统已启动 ==========")
    push_telegram(f"🤖 交易机器人启动\nK线: {BAR} | 预测: {HORIZON}根 ({HORIZON*5}分钟) | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮\n止盈: 动态止损+跟踪止损 | 最长持仓: 动态预测时间\n固定保证金: {MAX_SINGLE_TRADE_USDT} USDT/币\n流动性: 成交额≥{MIN_VOLUME_USDT/1_000_000:.0f}M, 市值≥{MIN_MARKET_CAP_USDT/1_000_000:.0f}M\n仓位模式: 逐仓 {LEVERAGE}x\n信号门槛: 置信度≥{MIN_DIRECTION_CONFIDENCE}, R²≥{MIN_R_SQUARED}\n技术指标: RSI周期{RSI_PERIOD} 多单<{RSI_LONG_THRESHOLD} 空单>{RSI_SHORT_THRESHOLD}; MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})\n风控: 最多{MAX_CONCURRENT_POSITIONS}仓, 总保证金≤{MAX_TOTAL_MARGIN_RATIO*100}%权益\n开仓条件: 实体位置(顺势放宽)、有利移动、紧急动能、强势突破追单\n多空相对评分制 | 动态预期收益率: 顺势0.3% / 逆势0.8%\n持仓时间: 基于TimesFM预测路径动态估算，超时智能处理（盈利则跟踪止盈，亏损则止损，横盘则平仓）\n开仓前安全检查: 逆势信号检查1小时EMA20偏离、15分钟实体放大、成交量突增\n新闻监控: Finnhub Crypto News API (Free Tier)\n大周期趋势过滤: 1小时/15分钟下跌趋势禁止开多，上涨趋势禁止开空")

    while True:
        try:
            now = datetime.now()
            trader.sync_strategy_positions_with_exchange()
            trader.check_and_close_positions()
            trader.check_manual_close()
            trader.check_and_open_pending()
            trader.check_manual_open()

            # 检查是否允许开新仓（通过文件控制）
            disable_file = "/tmp/disable_new_trades"
            disable_trading = os.path.exists(disable_file)

            if (now - last_pred).total_seconds() >= PREDICTION_INTERVAL:
                has_set_pending_this_cycle = False
                signals_dict = run_prediction_cycle()
                last_pred = now
                trader.clear_pending_signals()

                if signals_dict and not has_set_pending_this_cycle:
                    if disable_trading:
                        log("⏸️ 检测到禁用开仓标志文件，跳过本轮开仓")
                        push_telegram("⏸️ 新开仓已手动暂停（检测到 /tmp/disable_new_trades）")
                    else:
                        current_positions_count = len(trader.strategy_positions)
                        if current_positions_count >= MAX_CONCURRENT_POSITIONS:
                            push_telegram(f"⚠️ 当前已有 {current_positions_count} 个策略持仓，达到上限 {MAX_CONCURRENT_POSITIONS}，本次信号暂不开仓")
                        else:
                            available_balance = trader.get_available_balance()
                            existing_symbols = set(trader.strategy_positions.keys())
                            filtered_signals = {sym: (sig, exp_ret, hold_min, conf) for sym, (sig, exp_ret, hold_min, conf) in signals_dict.items() 
                                                if sym not in existing_symbols}
                            if filtered_signals:
                                signals_list = []
                                for sym, (sig, exp_ret, hold_min, conf) in filtered_signals.items():
                                    margin_amount = 50 if conf > 0.95 else 30
                                    if margin_amount > available_balance:
                                        push_telegram(f"⚠️ 余额不足，跳过信号 {sym} 需要 {margin_amount} U")
                                        continue
                                    signals_list.append((sym, sig, exp_ret, hold_min, margin_amount))
                                if signals_list:
                                    trader.set_pending_signals(signals_list)
                                    has_set_pending_this_cycle = True
                                    push_telegram(f"📋 已设置 {len(signals_list)} 个待开仓信号（高信50U，普通30U）")
                                else:
                                    push_telegram(f"⚠️ 所有信号币种均已有持仓或余额不足，本次无新开仓")
                            else:
                                push_telegram(f"⚠️ 所有信号币种均已有持仓，本次无新开仓")

            all_pos = trader.sync_positions()
            strategy_symbols = list(trader.strategy_positions.keys())
            pending_count = len(trader.pending_signals)
            log(f"\n📊 当前所有持仓（含人工）: {list(all_pos.keys())}")
            log(f"🎮 策略管理持仓: {strategy_symbols}")
            log(f"⏳ 待开仓信号数: {pending_count}")
            log(f"💤 等待10秒刷新...\n")
            time.sleep(10)

        except Exception as e:
            err(f"主循环异常: {traceback.format_exc()}")
            push_telegram(f"❌ 机器人异常崩溃: {str(e)[:100]}")
            time.sleep(10)

if __name__ == "__main__":
    trader_obj = None
    try:
        trader_obj = main()
    except KeyboardInterrupt:
        log("\n🛑 手动停止")
        if trader_obj:
            trader_obj.close_all()
        push_telegram("🛑 交易机器人已停止")
