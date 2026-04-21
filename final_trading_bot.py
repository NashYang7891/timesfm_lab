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

BAR = "3m"
HIGHER_BAR = "15m"
LIMIT = 900
HORIZON = 4

TOP_N = 50
FINAL_PICK_N = 3
BASE_MIN_EXPECTED_RETURN = 0.006
TREND_FOLLOWING_THRESHOLD = 0.003
COUNTER_TREND_THRESHOLD = 0.008

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
MAX_SINGLE_TRADE_USDT = 30
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

# ========= 多空对冲优化参数 =========
MIN_SCORE_GAP = 0.3
ADX_THRESHOLD = 25
RSI_SHORT_LIMIT = 45
VOLATILITY_MEDIAN_PERIOD = 20
LONG_CONF_LOW_THRESHOLD = 40.0
LONG_CONF_LOW_BARS = 3

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

def _build_timesfm_model():
    errors = []
    try:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
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
            log(f"🚨 紧急动能信号: {symbol} {side_cn} 价格变动 {move_pct:.1f}% >= {EMERGENCY_MOVE_PCT}%")
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
    if bar_frame == "3m":
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

# ==================== 7. 预测评分 + 多空对冲优化 ====================
history_long_scores = {}

def compute_signal_score(symbol, side, current_price, expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m):
    base_conf = 0.7 * consistency + 0.3 * max(0.0, min(1.0, r_squared))
    trend_factor = 1.0
    if ema20_15m is not None and slope_15m is not None:
        if side == 'long':
            if current_price < ema20_15m:
                trend_factor = 0.5
                if slope_15m < -0.001:
                    trend_factor *= 0.7
        else:
            if current_price > ema20_15m:
                trend_factor = 0.5
                if slope_15m > 0.001:
                    trend_factor *= 0.7
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

def get_prev_high_2bars(symbol):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 3)
        if df is None or len(df) < 3:
            return None
        highs = df['h'].iloc[-3:-1].values
        return max(highs) if len(highs) > 0 else None
    except:
        return None

def get_ema9(symbol):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 20)
        if df is None or len(df) < 10:
            return None
        closes = df['c'].astype(float)
        ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
        return ema9
    except:
        return None

def check_short_optimized(symbol, long_score, short_score, current_price, is_downtrend, adx, atr_pct, vol_median_pct, long_score_history):
    # 安全处理 None 值
    adx_str = f"{adx:.1f}" if adx is not None else "N/A"
    if adx is None or adx < ADX_THRESHOLD:
        return False, f"ADX={adx_str} < {ADX_THRESHOLD}，趋势弱"
    
    atr_pct_str = f"{atr_pct:.2f}" if atr_pct is not None else "N/A"
    vol_median_pct_str = f"{vol_median_pct:.2f}" if vol_median_pct is not None else "N/A"
    if atr_pct is None or vol_median_pct is None or atr_pct <= vol_median_pct:
        return False, f"波动率{atr_pct_str}% ≤ 中位数{vol_median_pct_str}%，市场横盘"
    
    if len(long_score_history) < LONG_CONF_LOW_BARS:
        return False, f"历史多头评分不足{LONG_CONF_LOW_BARS}根"
    if not all(score < LONG_CONF_LOW_THRESHOLD for score in long_score_history[-LONG_CONF_LOW_BARS:]):
        return False, f"多头评分未连续{LONG_CONF_LOW_BARS}根低于{LONG_CONF_LOW_THRESHOLD}"
    
    prev_high = get_prev_high_2bars(symbol)
    ema9 = get_ema9(symbol)
    if prev_high is None or ema9 is None:
        return False, "无法获取前高或EMA9"
    if current_price >= prev_high:
        return False, f"当前价{current_price:.6f} ≥ 前高{prev_high:.6f}"
    if current_price >= ema9:
        return False, f"当前价{current_price:.6f} ≥ EMA9{ema9:.6f}"
    
    score_gap = short_score - long_score
    if score_gap < MIN_SCORE_GAP:
        return False, f"多空梯度差值{score_gap:.2f} < {MIN_SCORE_GAP}"
    if not is_downtrend:
        return False, "15分钟趋势并非下降"
    
    return True, f"通过所有优化条件 (gap={score_gap:.2f}, ADX={adx_str}, 波动率{atr_pct_str}%>{vol_median_pct_str}%)"

def calculate_bollinger_bands(symbol, period=20, std=2):
    """计算布林带上下轨"""
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
    """
    信号过滤器：防止高位接多、低位追空
    返回 (是否通过, 拒绝原因)
    """
    # 1. 极端波动率拦截（ATR% > 5% 时，市场可能插针）
    if atr_pct is not None and atr_pct > 5.0:
        return False, f"波动率过高 ({atr_pct:.2f}%)，暂停开仓"

    # 2. 布林带位置计算
    bb_upper, bb_lower = calculate_bollinger_bands(symbol)
    
    if signal_type == 'LONG':
        if rsi is not None and rsi > 65:
            return False, f"RSI={rsi:.1f} 超买区，禁止追多"
        if bb_upper is not None and current_price > bb_upper:
            return False, f"价格突破布林带上轨 {bb_upper:.6f}，过高"
        if adx is not None and adx > 60:
            return False, f"ADX={adx:.1f} 趋势极端，可能衰竭"
        # TimesFM 预测区间校验
        if forecast_values is not None and len(forecast_values) >= 5:
            forecast_high_max = max(forecast_values[:5])
            if forecast_high_max < current_price:
                return False, f"TimesFM预测未来最高价 {forecast_high_max:.6f} 低于当前价，上涨乏力"

    elif signal_type == 'SHORT':
        if rsi is not None and rsi < 35:
            return False, f"RSI={rsi:.1f} 超卖区，禁止追空"
        if bb_lower is not None and current_price < bb_lower:
            return False, f"价格跌破布林带下轨 {bb_lower:.6f}，过低"
        if adx is not None and adx > 60:
            return False, f"ADX={adx:.1f} 趋势极端，可能衰竭"
        if forecast_values is not None and len(forecast_values) >= 5:
            forecast_low_min = min(forecast_values[:5])
            if forecast_low_min > current_price:
                return False, f"TimesFM预测未来最低价 {forecast_low_min:.6f} 高于当前价，下跌空间不足"

    # 4. 成交量异常拦截（选做）
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

        with torch.no_grad():
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
        is_downtrend = slope_15m is not None and slope_15m < -0.002
        is_uptrend = slope_15m is not None and slope_15m > 0.002

        long_conf, long_score = compute_signal_score(instId, 'long', current_price, expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m)
        short_conf, short_score = compute_signal_score(instId, 'short', current_price, -expected_return, r_squared, consistency, vol_ratio, ema20_15m, slope_15m)

        # 存储历史多头评分
        global history_long_scores
        if instId not in history_long_scores:
            history_long_scores[instId] = []
        history_long_scores[instId].append(long_score)
        if len(history_long_scores[instId]) > LONG_CONF_LOW_BARS + 5:
            history_long_scores[instId] = history_long_scores[instId][-(LONG_CONF_LOW_BARS+5):]

        # 获取ADX、ATR%和中位数波动率
        adx = get_adx(instId)
        atr_pct = get_atr_percent(instId)
        vol_median_pct = None
        if atr_pct is not None:
            df_atr = fetch_klines_with_retry(instId, BAR, VOLATILITY_MEDIAN_PERIOD+10)
            if df_atr is not None and len(df_atr) >= VOLATILITY_MEDIAN_PERIOD:
                atr_list = []
                for i in range(len(df_atr)-ATR_PERIOD):
                    sub = df_atr.iloc[i:i+ATR_PERIOD]
                    high = sub['h'].values
                    low = sub['l'].values
                    close = sub['c'].values
                    tr = np.maximum(high[1:] - low[1:],
                                    np.abs(high[1:] - close[:-1]),
                                    np.abs(low[1:] - close[:-1]))
                    atr_val = np.mean(tr)
                    atr_pct_val = atr_val / close[-1] * 100
                    atr_list.append(atr_pct_val)
                if len(atr_list) >= VOLATILITY_MEDIAN_PERIOD:
                    vol_median_pct = np.median(atr_list[-VOLATILITY_MEDIAN_PERIOD:])

        # 根据趋势方向设置预期收益门槛
        if is_downtrend:
            min_ret_long = COUNTER_TREND_THRESHOLD
            min_ret_short = TREND_FOLLOWING_THRESHOLD
        elif is_uptrend:
            min_ret_long = TREND_FOLLOWING_THRESHOLD
            min_ret_short = COUNTER_TREND_THRESHOLD
        else:
            min_ret_long = BASE_MIN_EXPECTED_RETURN
            min_ret_short = BASE_MIN_EXPECTED_RETURN

        # 先检查空单是否满足强化条件
        short_optimized_pass = False
        short_optimized_reason = ""
        if is_downtrend and long_score < LONG_CONF_LOW_THRESHOLD:
            short_optimized_pass, short_optimized_reason = check_short_optimized(
                instId, long_score, short_score, current_price, is_downtrend,
                adx, atr_pct, vol_median_pct, history_long_scores.get(instId, [])
            )

        # 选择最佳方向
        best_side = None
        best_score = 0
        best_conf = 0
        best_ret = 0

        # 空单候选
        if short_optimized_pass or (short_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_short):
            try:
                df_rsi = fetch_klines_with_retry(instId, BAR, 20)
                if df_rsi is not None and len(df_rsi) >= 14:
                    rsi_val = compute_rsi(df_rsi['c'], RSI_PERIOD)
                    if rsi_val >= RSI_SHORT_LIMIT:
                        raise ValueError(f"RSI={rsi_val:.1f} ≥ {RSI_SHORT_LIMIT}，不满足空单条件")
                else:
                    raise ValueError("无法获取RSI")
            except Exception:
                short_optimized_pass = False
                short_conf = 0
            if short_optimized_pass or short_conf >= MIN_DIRECTION_CONFIDENCE:
                best_side = 'short'
                best_score = short_score
                best_conf = short_conf
                best_ret = -abs(expected_return)

        # 多单候选
        if long_conf >= MIN_DIRECTION_CONFIDENCE and abs(expected_return) >= min_ret_long:
            if best_side is None or long_score > best_score:
                best_side = 'long'
                best_score = long_score
                best_conf = long_conf
                best_ret = abs(expected_return)

        # ========== 新增：信号过滤器拦截 ==========
        if best_side is not None:
            # 获取当前 RSI（用于过滤器）
            try:
                df_rsi = fetch_klines_with_retry(instId, BAR, 20)
                if df_rsi is not None and len(df_rsi) >= 14:
                    rsi_filter = compute_rsi(df_rsi['c'], RSI_PERIOD)
                else:
                    rsi_filter = 50
            except:
                rsi_filter = 50

            valid, reject_reason = validate_signal(
                signal_type=best_side.upper(),
                symbol=instId,
                current_price=current_price,
                rsi=rsi_filter,
                adx=adx,
                atr_pct=atr_pct,
                forecast_values=forecast_values  # 传入 TimesFM 预测序列
            )
            if not valid:
                return None, f"信号过滤器拦截 ({best_side.upper()}): {reject_reason}"
        # ==========================================

        if best_side is None:
            reject_reason = f"多空均未通过阈值 (多: {long_conf:.2f}/{long_score:.1f}, 空: {short_conf:.2f}/{short_score:.1f})"
            if is_downtrend and long_score < LONG_CONF_LOW_THRESHOLD:
                reject_reason += f" | 空单优化过滤: {short_optimized_reason}"
            return None, reject_reason

        # 获取价格实体信息用于开仓点位
        candle = fetch_previous_candle(instId)
        if candle is None:
            price_info = None
        else:
            open_p, high, low, close = candle
            body_top = max(open_p, close)
            body_bottom = min(open_p, close)
            body_len = body_top - body_bottom
            if body_len > 0:
                long_entry_max = body_bottom + body_len * PRICE_POSITION_RATIO
                short_entry_min = body_top - body_len * PRICE_POSITION_RATIO
                price_info = {
                    'current_price': current_price,
                    'body_top': body_top,
                    'body_bottom': body_bottom,
                    'long_entry_max': long_entry_max,
                    'short_entry_min': short_entry_min
                }
            else:
                price_info = None

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
            "tech_msg": f"多空对冲评分 (多{long_score:.1f}/空{short_score:.1f})，选择{best_side}",
            "long_score": long_score,
            "short_score": short_score,
            "adx": adx if adx is not None else 0,
            "atr_pct": atr_pct if atr_pct is not None else 0,
            "vol_median_pct": vol_median_pct if vol_median_pct is not None else 0,
            "rsi": None,
        }
        # 补充RSI值
        try:
            df_rsi = fetch_klines_with_retry(instId, BAR, 20)
            if df_rsi is not None:
                result["rsi"] = compute_rsi(df_rsi['c'], RSI_PERIOD)
        except:
            result["rsi"] = 50
        return result, ""

    except Exception as e:
        return None, f"异常: {str(e)[:50]}"

# ==================== 8. 预测循环 ====================
def run_prediction_cycle():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n============================================================")
    log(f"🔄 [{now_str}] {BAR}周期 | 预测{HORIZON}步（{HORIZON*3}分钟） | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮")
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
        if res:
            valid.append(res)
            candidate_details.append(res)
            log(f"  [{len(valid)}] {res['symbol']}")
            log(f"      预期涨跌: {res['expected_return']*100:+.2f}%")
            log(f"      R²: {res['r_squared']:.2f} | 一致性: {res['consistency']:.2f} | 方向置信度: {res['direction_confidence']:.2f} | 得分: {res['score']:.4f}")
            log(f"      技术指标: {res['tech_msg']}")
        else:
    # 从 reject_reason 中尝试提取多空评分（格式如 "多: 0.35/32.2, 空: 0.00/80.6"）
    long_score = 0.0
    short_score = 0.0
    try:
        # 匹配 "多: X/Y" 和 "空: X/Y"
        import re
        match_long = re.search(r'多:\s*[\d.]+/([\d.]+)', reason)
        match_short = re.search(r'空:\s*[\d.]+/([\d.]+)', reason)
        if match_long:
            long_score = float(match_long.group(1))
        if match_short:
            short_score = float(match_short.group(1))
    except:
        pass
    
    # 获取 ADX、ATR%、RSI（如果失败则用0）
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
    df_results = df_results[df_results['score'] >= 95]
    if df_results.empty:
        log("❌ 无得分 ≥95 的高质量信号")
        push_telegram("❌ 本轮无得分 ≥95 的高质量信号")
        return {}

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
        msg.append(f"#{symbol} | {signal}")
        msg.append(f"  预期收益: {exp_return:+.2f}% | 置信度: {confidence:.2f} | 得分: {score:.4f}")
        if price_info:
            current = price_info['current_price']
            body_bottom = price_info['body_bottom']
            body_top = price_info['body_top']
            msg.append(f"  当前价格: {current:.6f}")
            msg.append(f"  上一根K线实体区间: [{body_bottom:.6f} - {body_top:.6f}]")
            if signal == 'LONG':
                entry_max = price_info['long_entry_max']
                msg.append(f"  建议多单入场: 价格 ≤ {entry_max:.6f} (实体底部+{PRICE_POSITION_RATIO*100:.0f}%区域)")
            else:
                entry_min = price_info['short_entry_min']
                msg.append(f"  建议空单入场: 价格 ≥ {entry_min:.6f} (实体顶部-{PRICE_POSITION_RATIO*100:.0f}%区域)")
        else:
            msg.append("  ⚠️ 无法获取价格位置信息")
        msg.append(f"  技术确认: {tech_msg}")
        msg.append("")
    push_telegram("\n".join(msg))

    output_dict = {row['symbol']: (row['signal'], row['expected_return']) for _, row in top.iterrows()}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({k: v[0] for k, v in output_dict.items()}, f, indent=2, ensure_ascii=False)
    return output_dict

# ==================== 9. 交易模块（保持原有完整逻辑，因篇幅限制此处仅做占位，实际请使用之前完整版） ====================
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
                    'expected_met': info.get('expected_met', False)
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
                    'expected_met': info.get('expected_met', False)
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
                            'expected_met': False
                        }
                        self._save_strategy_positions()
                        log(f"🔄 接管孤儿持仓: {sym} {side.upper()} 已纳入管理")
                        push_telegram(f"🔄 策略接管持仓: {sym} {side.upper()}，现由程序自动管理")
                        self.pending_signals = [sig for sig in self.pending_signals if sig['ccxt_symbol'] != sym and sig['raw_symbol'] != sym]
        except Exception as e:
            err(f"同步策略持仓异常: {e}")

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

    def check_price_position_entity(self, symbol, side):
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
                max_allowed = body_bottom + body_len * PRICE_POSITION_RATIO
                if current_price <= max_allowed:
                    return True, f"价格位置满足多单条件: 当前{current_price:.6f} ≤ {max_allowed:.6f}"
                else:
                    return False, f"价格位置不满足多单条件: 当前{current_price:.6f} > {max_allowed:.6f}"
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

    def set_pending_signals(self, signals_list, margin_amount):
        self.pending_signals = []
        for raw_symbol, side, expected_return in signals_list:
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
                    'expected_return': expected_return
                })
            except Exception as e:
                err(f"获取市场信息失败 {raw_symbol}: {e}")
        log(f"📋 设置待开仓信号: {len(self.pending_signals)} 个")

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
            try:
                ticker = self.exchange.fetch_ticker(raw_symbol)
                current_price = ticker['last']
            except:
                log(f"❌ 无法获取 {raw_symbol} 当前价格")
                continue
            ccxt_symbol = self.exchange.market(raw_symbol)['symbol']
            current_positions = self.sync_positions()
            if ccxt_symbol in current_positions:
                log(f"⏸️ {raw_symbol} 已有持仓，取消待开仓信号")
                to_remove.append(idx)
                continue
            favorable, pct_change = self.check_favorable_move(signal_price, side, current_price)
            if favorable:
                log(f"🚀 价格向有利方向移动 {pct_change:.2f}% ≥ {FAVORABLE_MOVE_PCT}%，立即开仓 {raw_symbol} {side.upper()}")
                success = self.open_position(raw_symbol, side, margin, expected_return, ignore_price_position=True)
                if success:
                    to_remove.append(idx)
                else:
                    log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
            else:
                ok, msg = self.check_price_position_entity(raw_symbol, side)
                if ok:
                    log(f"🚀 待开仓信号价格满足实体位置条件: {raw_symbol} {side.upper()}")
                    success = self.open_position(raw_symbol, side, margin, expected_return, ignore_price_position=False)
                    if success:
                        to_remove.append(idx)
                    else:
                        log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
                else:
                    log(f"⏸️ 待开仓信号 {raw_symbol} {side.upper()} 价格不满足: {msg}")
        for idx in sorted(to_remove, reverse=True):
            self.pending_signals.pop(idx)

    def open_position(self, symbol, side, base_margin_usdt, expected_return, ignore_price_position=False):
        current_positions = self.sync_positions()
        ccxt_symbol = self.exchange.market(symbol)['symbol']
        if ccxt_symbol in current_positions:
            log(f"⏸️ {symbol} 已有持仓 {current_positions[ccxt_symbol]}，拒绝重复开仓")
            return False

        emergency, move_pct = check_emergency_move(symbol, self.exchange.fetch_ticker(symbol)['last'])
        if emergency:
            ignore_price_position = True
            log(f"🚨 检测到紧急价格变动 {move_pct:.1f}%，将忽略实体位置检查直接开仓")

        if not ignore_price_position:
            ok, msg = self.check_price_position_entity(symbol, side)
            if not ok:
                log(f"⏸️ 跳过开仓 {symbol} {side}: {msg}")
                return False
        else:
            log(f"🚀 忽略实体位置检查，因有利移动或紧急变动触发开仓 {symbol} {side}")

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
                            'expected_met': False
                        }
                        self._save_strategy_positions()
                        log(f"✅ 补救记录持仓 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_margin:.2f} | 止损: {stop_loss_price:.4f}")
                        push_telegram(f"✅ 策略开仓成功（补救）\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_margin:.2f}\n动态止损: ${stop_loss_price:.4f}")
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
                'expected_met': False
            }
            self._save_strategy_positions()
            log(f"✅ 开仓成功 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_used_margin:.2f} | 止损: {stop_loss_price:.4f}")
            msg = f"✅ 策略开仓成功\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_used_margin:.2f}\n动态止损: ${stop_loss_price:.4f}\n预期收益: {expected_return*100:.2f}%"
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
                            'expected_met': False
                        }
                        self._save_strategy_positions()
                        log(f"✅ 异常后补救记录持仓 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_margin:.2f} | 止损: {stop_loss_price:.4f}")
                        push_telegram(f"✅ 策略开仓成功（异常补救）\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_margin:.2f}\n动态止损: ${stop_loss_price:.4f}")
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
            if side == 'long':
                order_side = 'sell'
                position_side = 'long'
            else:
                order_side = 'buy'
                position_side = 'short'

            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=order_side,
                amount=open_qty,
                params={'positionSide': position_side, 'reduceOnly': True}
            )
            close_price = order.get('average', 0)
            if close_price == 0:
                ticker = self.exchange.fetch_ticker(ccxt_symbol)
                close_price = ticker['last']
        except Exception as e:
            err(f"平仓失败 {ccxt_symbol}: {e}, 尝试备用方法")
            try:
                order = self.exchange.create_market_close_order(ccxt_symbol, side=side)
                close_price = order.get('average', 0)
                if close_price == 0:
                    ticker = self.exchange.fetch_ticker(ccxt_symbol)
                    close_price = ticker['last']
            except Exception as e2:
                err(f"备用平仓也失败 {ccxt_symbol}: {e2}")
                ticker = self.exchange.fetch_ticker(ccxt_symbol)
                close_price = ticker['last']

        if side == 'long':
            pnl_usdt = (close_price - open_price) * open_qty
        else:
            pnl_usdt = (open_price - close_price) * open_qty
        pnl_percent = (pnl_usdt / open_margin) * 100 if open_margin > 0 else 0

        del self.strategy_positions[ccxt_symbol]
        self._save_strategy_positions()

        reason_text = f"原因: {reason}" if reason else ""
        push_telegram(f"🔻 策略平仓\n币种: {ccxt_symbol}\n方向: {side.upper()}\n开仓价: ${open_price:.4f}\n平仓价: ${close_price:.4f}\n盈亏金额: ${pnl_usdt:+.2f}\n盈亏百分比(保证金): {pnl_percent:+.2f}%\n持仓时长: {hold_seconds/60:.1f}分钟\n{reason_text}")
        log(f"平仓记录 {ccxt_symbol} | 盈亏: {pnl_usdt:+.2f} USDT ({pnl_percent:+.2f}%) | {reason}")
        return True

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
        log(f"🔍 检查持仓: 策略持仓 {list(self.strategy_positions.keys())}, 交易所持仓 {list(pos_map.keys())}")
        self.sync_strategy_positions_with_exchange()

        for sym, info in list(self.strategy_positions.items()):
            if sym not in pos_map:
                log(f"⚠️ 策略持仓 {sym} 未在交易所持仓中找到，可能已被平仓")
                del self.strategy_positions[sym]
                self._save_strategy_positions()
                continue
            pos = pos_map[sym]
            current_price = float(pos.get('last', 0))
            if current_price == 0:
                continue

            pnl_percent = float(pos.get('percentage', 0))
            if abs(pnl_percent) < 1:
                pnl_percent = pnl_percent * 100

            hold_seconds = time.time() - info['open_time']
            expected_return = info.get('expected_return', 0)
            expected_pct = abs(expected_return * 100) if expected_return else 0

            # 保险：开仓12分钟后，如果从未达到预期收益，强制平仓
            if hold_seconds >= 12 * 60 and not info.get('expected_met', False):
                log(f"⏰ 开仓12分钟未达到预期收益，强制平仓: {sym} 盈亏 {pnl_percent:.2f}%, 预期 {expected_pct:.2f}%")
                self.close_position(sym, reason=f"12分钟未达预期 (盈亏 {pnl_percent:.2f}%)")
                closed_any = True
                continue

            # 判断方向是否正确
            direction_correct = (info['side'] == 'long' and expected_return > 0) or (info['side'] == 'short' and expected_return < 0)

            # 如果方向正确且尚未达到预期收益，检查是否已达到
            if direction_correct and not info.get('expected_met', False):
                if abs(pnl_percent) >= expected_pct:
                    info['expected_met'] = True
                    log(f"🎯 达到预期收益: {sym} 浮盈 {pnl_percent:.2f}% >= 预期 {expected_pct:.2f}%")
                    push_telegram(f"🎯 {sym} 达到预期收益 {expected_pct:.2f}%，激活跟踪止损")

            # 跟踪止损：仅在达到预期收益后激活
            if info.get('expected_met', False):
                trailing_stop = info.get('trailing_stop_pct', 1.0)

                if not info.get('trailing_activated', False):
                    info['trailing_activated'] = True
                    log(f"🔒 跟踪止损已激活: {sym}")
                    push_telegram(f"🔒 跟踪止损激活: {sym} 浮盈 {pnl_percent:.2f}%")

                if info.get('trailing_activated', False):
                    if info['side'] == 'long':
                        if current_price > info.get('highest_price', info['open_price']):
                            info['highest_price'] = current_price
                        peak = info['highest_price']
                        drawdown_pct = (peak - current_price) / peak * 100
                        if drawdown_pct >= trailing_stop:
                            log(f"📉 触发跟踪止损: {sym} 多单，最高价 {peak:.4f}，当前价 {current_price:.4f}，回撤 {drawdown_pct:.2f}% >= {trailing_stop}%")
                            self.close_position(sym, reason=f"跟踪止损（回撤 {drawdown_pct:.2f}%）")
                            closed_any = True
                            continue
                    else:
                        if current_price < info.get('lowest_price', info['open_price']):
                            info['lowest_price'] = current_price
                        trough = info['lowest_price']
                        bounce_pct = (current_price - trough) / trough * 100
                        if bounce_pct >= trailing_stop:
                            log(f"📈 触发跟踪止损: {sym} 空单，最低价 {trough:.4f}，当前价 {current_price:.4f}，反弹 {bounce_pct:.2f}% >= {trailing_stop}%")
                            self.close_position(sym, reason=f"跟踪止损（反弹 {bounce_pct:.2f}%）")
                            closed_any = True
                            continue

            # ATR 初始止损
            stop_price = info.get('stop_loss_price')
            if stop_price is not None:
                if (info['side'] == 'long' and current_price <= stop_price) or (info['side'] == 'short' and current_price >= stop_price):
                    log(f"💥 触发初始动态止损: {sym} 当前价 {current_price:.4f} 触及止损价 {stop_price:.4f}")
                    self.close_position(sym, reason=f"初始止损 {stop_price:.4f}")
                    closed_any = True
                    continue

            log(f"📉 检查持仓 {sym}: 盈亏 {pnl_percent:.2f}%, 预期 {expected_pct:.2f}%, 达到预期: {info.get('expected_met', False)}, 跟踪激活: {info.get('trailing_activated', False)}, 持仓时长 {hold_seconds/60:.1f}分钟")

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
            push_telegram(f"🔻 人工平仓（策略持仓）\n币种: {sym}\n方向: {side.upper()}\n开仓价: ${open_price:.4f}\n估算平仓价: ${current_price:.4f}\n估算盈亏: ${pnl_usdt:+.2f} ({pnl_percent:+.2f}%)\n注意：此盈亏为近似值")
            log(f"人工平仓检测: {sym} 策略持仓已不存在，清理记录")
            del self.strategy_positions[sym]
        if closed_by_manual:
            self._save_strategy_positions()
            self.last_positions = self.sync_positions()

    def check_manual_open(self):
        current_positions = self.sync_positions()
        newly_opened = []
        pending_symbols = set()
        for sig in self.pending_signals:
            pending_symbols.add(sig['raw_symbol'])
            pending_symbols.add(sig['ccxt_symbol'])
        for sym, side in current_positions.items():
            if sym not in self.last_positions and sym not in self.strategy_positions and sym not in pending_symbols:
                newly_opened.append((sym, side))
        self.last_positions = current_positions.copy()
        for sym, side in newly_opened:
            try:
                ticker = self.exchange.fetch_ticker(sym)
                price = ticker['last']
            except:
                price = 0
            push_telegram(f"🔵 检测到人工开仓\n币种: {sym}\n方向: {side.upper()}\n当前价格: ${price:.4f}\n注意：此仓位不会自动止盈止损，请自行管理")
            log(f"人工开仓检测: {sym} {side.upper()} @ {price}")

    def clear_pending_signals(self):
        self.pending_signals = []

# ==================== 10. 主程序 ====================
def main():
    # 写入 PID 文件（用于监控）
    with open("/tmp/trading_bot.pid", "w") as f:
        f.write(str(os.getpid()))
    
    trader = OKXTrader()
    last_pred = datetime.now() - timedelta(seconds=PREDICTION_INTERVAL)
    has_set_pending_this_cycle = False

    log("\n========== 全自动交易系统已启动 ==========")
    push_telegram(f"🤖 交易机器人启动\nK线: {BAR} | 预测: {HORIZON}根 ({HORIZON*3}分钟) | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮\n止盈: 动态止损+跟踪止损 | 最长持仓: {MAX_HOLD_SECONDS/60:.0f}分钟\n固定保证金: {MAX_SINGLE_TRADE_USDT} USDT/币\n流动性: 成交额≥{MIN_VOLUME_USDT/1_000_000:.0f}M, 市值≥{MIN_MARKET_CAP_USDT/1_000_000:.0f}M\n仓位模式: 逐仓 {LEVERAGE}x\n信号门槛: 得分≥95, 置信度≥{MIN_DIRECTION_CONFIDENCE}, R²≥{MIN_R_SQUARED}\n技术指标: RSI周期{RSI_PERIOD} 多单<{RSI_LONG_THRESHOLD} 空单>{RSI_SHORT_THRESHOLD}; MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})\n风控: 最多{MAX_CONCURRENT_POSITIONS}仓, 总保证金≤{MAX_TOTAL_MARGIN_RATIO*100}%权益\n开仓条件: 实体位置 或 价格有利移动 或 紧急动能信号\n多空对冲优化: ADX>{ADX_THRESHOLD}, 波动率>中位数, 多头连续{LONG_CONF_LOW_BARS}根低于{LONG_CONF_LOW_THRESHOLD}, 梯度差>{MIN_SCORE_GAP}, RSI<{RSI_SHORT_LIMIT}")

    while True:
        try:
            now = datetime.now()
            trader.sync_strategy_positions_with_exchange()
            trader.check_and_close_positions()
            trader.check_manual_close()
            trader.check_and_open_pending()
            trader.check_manual_open()

            if (now - last_pred).total_seconds() >= PREDICTION_INTERVAL:
                has_set_pending_this_cycle = False
                signals_dict = run_prediction_cycle()
                last_pred = now
                trader.clear_pending_signals()

                if signals_dict and not has_set_pending_this_cycle:
                    current_positions_count = len(trader.strategy_positions)
                    if current_positions_count >= MAX_CONCURRENT_POSITIONS:
                        push_telegram(f"⚠️ 当前已有 {current_positions_count} 个策略持仓，达到上限 {MAX_CONCURRENT_POSITIONS}，本次信号暂不开仓")
                    else:
                        available_balance = trader.get_available_balance()
                        open_amount = MAX_SINGLE_TRADE_USDT
                        if available_balance < open_amount + 5:
                            push_telegram(f"⚠️ 可用余额不足 {open_amount} USDT（可用: {available_balance:.2f}），无法开仓")
                        else:
                            signals_list = [(sym, sig, exp_ret) for sym, (sig, exp_ret) in signals_dict.items()]
                            trader.set_pending_signals(signals_list, open_amount)
                            has_set_pending_this_cycle = True
                            push_telegram(f"📋 已设置待开仓信号，将在价格满足条件时开仓，保证金: {open_amount:.2f} USDT/币")

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

    return trader

# ==================== 11. 入口 ====================
if __name__ == "__main__":
    trader_obj = None
    try:
        trader_obj = main()
    except KeyboardInterrupt:
        log("\n🛑 手动停止")
        for _ in range(3):
            if push_telegram("🛑 交易机器人正在停止..."):
                break
            time.sleep(1)
        if trader_obj is not None:
            trader_obj.close_all()
        for _ in range(3):
            if push_telegram("🛑 交易机器人已完全停止"):
                break
            time.sleep(1)
        time.sleep(1)
    except Exception as e:
        err_msg = f"❌ 机器人崩溃: {str(e)[:100]}"
        log(err_msg)
        for _ in range(3):
            if push_telegram(err_msg):
                break
            time.sleep(1)
        time.sleep(1)
