#!/usr/bin/env python3
"""
极端反转拐点单点捕捉系统 - 激进放宽版（流动性门槛保持 + 稳健错误处理）
- 零延迟振荡器 + 动能衰减瞬间（仅加速度方向）+ 成交量高对称区过滤
- 正向状态机触发，无 TimesFM 预测
- Telegram 推送 + 异步交易 (可关闭)
- 数据获取：失败返回0 + 缓存机制 + 重试，避免误入不明标的
"""

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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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
LIMIT = 200

# ---------- 极端反转阈值 (已激进放宽) ----------
EXTREME_PRICE_DEVIATION = 1.5
EXTREME_VOLUME_IMBALANCE_RATIO = 0.65
MOMENTUM_DECAY_BARS = 5
VOLUME_PROFILE_LOOKBACK = 200
VWAP_TOLERANCE_PCT = 1.5

OSCIL_LONG_THRESHOLD = 0.45
OSCIL_SHORT_THRESHOLD = 0.55

# 原有风控参数
TOP_N = 50
FINAL_PICK_N = 3
SIGNAL_VALID_SECONDS = 60
MAX_HOLD_SECONDS = 900
AUTO_TRADE = False

RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 2.0
STOP_LOSS_PCT = 1.5

# 流动性门槛（保持不变）
MIN_VOLUME_USDT = 10_000_000
MIN_MARKET_CAP_USDT = 20_000_000
VOLATILITY_SAMPLE_SIZE = 200

LEVERAGE = 3
USE_PERCENT_OF_AVAILABLE = 0.5
MAX_CONCURRENT_POSITIONS = 3
MAX_TOTAL_MARGIN_RATIO = 1.0

PRICE_POSITION_RATIO = 0.1

API_KEY = "10d14cf0-79da-4597-9456-3aa1b88e1acf"
API_SECRET = "1B6A940855EC5787CD4E56BEF6D94733"
API_PASS = "kP9!vR2@mN5+"
IS_SANDBOX = False

OUTPUT_FILE = f"{log_dir}/signals_vps.json"
STRATEGY_POSITIONS_FILE = f"{log_dir}/strategy_positions.json"

# 缓存相关
CACHE_TTL_SECONDS = 600  # 10分钟
volume_cache = {}        # symbol -> (value, timestamp)
marketcap_cache = {}
cache_lock = Lock()

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

# ==================== 4. 带重试与缓存的 OKX 数据获取 ====================
def http_get_with_retry(url, params=None, max_retries=2, timeout=5):
    """简单重试 GET 请求"""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(0.3)
    return None

def get_all_swap_contracts():
    try:
        url = "https://www.okx.com/api/v5/public/instruments"
        params = {"instType": "SWAP"}
        resp = http_get_with_retry(url, params=params, timeout=10)
        if resp is None:
            return []
        data = resp.json()["data"]
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
            resp = http_get_with_retry(url, params=params, timeout=10)
            if resp is None:
                continue
            data = resp.json()
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
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                err(f"获取K线失败 {instId}: {e}")
    return None

def fetch_volume_usdt(instId):
    """24h成交额，失败/无效返回0，有缓存"""
    now = time.time()
    with cache_lock:
        if instId in volume_cache:
            val, ts = volume_cache[instId]
            if now - ts < CACHE_TTL_SECONDS and val > 0:
                return val

    for attempt in range(3):
        try:
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {"instId": instId}
            resp = http_get_with_retry(url, params=params, timeout=5)
            if resp is None:
                continue
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                vol = float(data["data"][0].get("volCcy24h", 0))
                if vol > 0:
                    with cache_lock:
                        volume_cache[instId] = (vol, now)
                    return vol
        except:
            pass
    # 全部失败，返回0
    return 0.0

def fetch_market_cap(instId):
    """市值，优先 OKX ticker 再 CoinGecko，失败/无效返回0，有缓存"""
    now = time.time()
    with cache_lock:
        if instId in marketcap_cache:
            val, ts = marketcap_cache[instId]
            if now - ts < CACHE_TTL_SECONDS and val > 0:
                return val

    # 1. OKX 公共 ticker 市值
    for attempt in range(2):
        try:
            url = f"https://www.okx.com/api/v5/market/ticker?instId={instId}"
            resp = http_get_with_retry(url, timeout=5)
            if resp is None:
                continue
            data = resp.json()
            if data.get('code') == '0' and data.get('data'):
                mkt = float(data['data'][0].get('mktCap', 0))
                if mkt > 0:
                    with cache_lock:
                        marketcap_cache[instId] = (mkt, now)
                    return mkt
                break
        except:
            pass

    # 2. CoinGecko
    try:
        base = instId.split('-')[0]
        search_url = f"https://api.coingecko.com/api/v3/search?query={base}"
        resp = http_get_with_retry(search_url, timeout=5)
        if resp and resp.status_code == 200:
            coins = resp.json().get('coins')
            if coins:
                coin_id = coins[0]['id']
                coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                coin_resp = http_get_with_retry(coin_url, timeout=5)
                if coin_resp and coin_resp.status_code == 200:
                    market_cap = coin_resp.json().get('market_data', {}).get('market_cap', {}).get('usd', 0)
                    if market_cap > 0:
                        with cache_lock:
                            marketcap_cache[instId] = (market_cap, now)
                        return market_cap
    except:
        pass

    # 失败返回0
    return 0

def calculate_volatility(prices):
    if len(prices) < 30:
        return 0
    ret = np.diff(prices) / prices[:-1]
    return np.std(ret) * 1000

# ==================== 5. 零延迟振荡器与极端反转核心 ====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def compute_efficiency_ratio(prices, period=10):
    if len(prices) < period + 1:
        return 0.5
    direction = abs(prices[-1] - prices[-period-1])
    volatility = np.sum(np.abs(np.diff(prices[-(period+1):])))
    if volatility == 0:
        return 0.0
    return direction / volatility

def compute_simple_phase(prices, period=20):
    if len(prices) < period + 3:
        return 0.5
    window = prices[-period:]
    diff1 = np.diff(window)
    diff2 = np.diff(diff1)
    denom = np.abs(diff1[-len(diff2):]) + 1e-12
    phase_raw = np.arctan(diff2 / denom)
    phase_norm = (phase_raw[-1] + np.pi/2) / np.pi
    return np.clip(phase_norm, 0, 1)

def compute_volume_imbalance(df, period=20):
    if df is None or len(df) < period:
        return 0.5
    recent = df.iloc[-period:]
    buy_vol = recent[recent['c'] > recent['o']]['v'].sum()
    sell_vol = recent[recent['c'] < recent['o']]['v'].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.5
    return buy_vol / total

def compute_vwap(df, period=200):
    if df is None or len(df) < period:
        return None
    recent = df.iloc[-period:].copy()
    typical_price = (recent['h'] + recent['l'] + recent['c']) / 3
    vwap = np.sum(typical_price * recent['v']) / recent['v'].sum()
    return vwap

def detect_momentum_decay(df, period=MOMENTUM_DECAY_BARS):
    if df is None or len(df) < period + 3:
        return None, 0
    closes = df['c'].values.astype(float)
    diffs = np.diff(closes[-(period+3):])
    accel = np.diff(diffs)
    current_accel = accel[-1]
    if current_accel > 0:
        strength = min(1.0, abs(current_accel) * 50)
        return 'bullish', strength
    elif current_accel < 0:
        strength = min(1.0, abs(current_accel) * 50)
        return 'bearish', strength
    return None, 0

def check_extreme_condition(symbol, current_price, df_5m, atr):
    if df_5m is None or len(df_5m) < 30 or atr is None:
        return False, 0, None
    ema20 = df_5m['c'].ewm(span=20, adjust=False).mean().iloc[-1]
    deviation = (current_price - ema20) / atr if atr > 0 else 0
    imbalance = compute_volume_imbalance(df_5m, 20)
    extreme_imbalance = (imbalance > EXTREME_VOLUME_IMBALANCE_RATIO) or \
                        (imbalance < (1 - EXTREME_VOLUME_IMBALANCE_RATIO))
    if abs(deviation) > EXTREME_PRICE_DEVIATION and extreme_imbalance:
        severity = min(abs(deviation)/5, 1.0)*0.6 + (abs(imbalance-0.5)*2)*0.4
        return True, severity, {"deviation": deviation, "imbalance": imbalance}
    return False, 0, None

def extreme_reversal_signal(symbol, current_price):
    df_5m = fetch_klines_with_retry(symbol, BAR, 100)
    if df_5m is None or len(df_5m) < 60:
        log(f"⚠️ {symbol} 5m K线数据不足")
        return None
    df_1m = fetch_klines_with_retry(symbol, "1m", 60)
    if df_1m is None or len(df_1m) < 30:
        log(f"⚠️ {symbol} 1m K线数据不足，使用5m替代")
        df_1m = df_5m

    high = df_5m['h'].values
    low = df_5m['l'].values
    close = df_5m['c'].values
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else None

    is_extreme, extreme_sev, ext_info = check_extreme_condition(symbol, current_price, df_5m, atr)
    if not is_extreme:
        return None

    efficiency = compute_efficiency_ratio(df_5m['c'].values, 10)
    phase = compute_simple_phase(df_5m['c'].values, 20)
    rsi_val = compute_rsi(df_5m['c'], RSI_PERIOD)
    composite_oscil = 0.4*efficiency + 0.3*phase + 0.3*(rsi_val/100)

    decay_dir, decay_strength = detect_momentum_decay(df_1m)
    if decay_dir is None:
        return None

    vwap = compute_vwap(df_5m, VOLUME_PROFILE_LOOKBACK)
    vwap_distance = 0
    if vwap is not None:
        vwap_distance = abs(current_price - vwap) / vwap * 100
        if vwap_distance > VWAP_TOLERANCE_PCT:
            decay_strength *= 0.6

    direction = None
    if decay_dir == 'bullish' and composite_oscil < OSCIL_LONG_THRESHOLD:
        direction = 'long'
    elif decay_dir == 'bearish' and composite_oscil > OSCIL_SHORT_THRESHOLD:
        direction = 'short'

    if direction is None:
        return None

    score = (decay_strength*0.5 + extreme_sev*0.3 + (1 - abs(composite_oscil-0.5)*2)*0.2) * 100
    trigger_zone = (current_price * 0.998, current_price * 1.002)

    detail = (
        f"偏离={ext_info['deviation']:.2f}σ | "
        f"量比={ext_info['imbalance']:.2f} | "
        f"振荡={composite_oscil:.2f} | "
        f"衰竭={decay_dir} | "
        f"强度={decay_strength:.2f} | "
        f"VWAP偏离={vwap_distance:.1f}%"
    )

    return {
        'symbol': symbol,
        'signal': direction,
        'score': score,
        'current_price': current_price,
        'trigger_zone': trigger_zone,
        'detail': detail,
        'expected_return': 0.015 if direction == 'long' else -0.015,
        'estimated_hold_minutes': 10
    }

def generate_signals_for_symbol(symbol):
    try:
        ticker_url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
        resp = http_get_with_retry(ticker_url, params=None, timeout=5)
        if resp is None:
            return None, "ticker获取失败"
        data = resp.json()
        if data.get('code') != '0':
            return None, "ticker错误"
        current_price = float(data['data'][0]['last'])
        signal_info = extreme_reversal_signal(symbol, current_price)
        if signal_info is None:
            return None, "无极端反转信号"
        return signal_info, ""
    except Exception as e:
        return None, f"异常: {str(e)[:50]}"

# ==================== 6. 异步交易类（与原版一致，略） ====================
# （此处引用之前的 OKXTraderAsync 完整定义，因篇幅限制不再重复，实际使用时必须保留全部方法）
# 请将前面给出的 OKXTraderAsync 类完整粘贴在此处

# ==================== 7. 主循环 ====================
async def main_async():
    trader = OKXTraderAsync()
    await trader.init()
    last_pred = datetime.now() - timedelta(seconds=60)

    log("========== 极端反转拐点捕捉系统启动 (激进放宽版 + 稳健错误处理) ==========")
    push_telegram("🤖 极端反转系统启动 (激进放宽参数, 自动交易关闭)")

    while True:
        try:
            now = datetime.now()
            await trader.check_and_open_pending()

            if (now - last_pred).total_seconds() >= 60:
                log("🔄 开始新一轮市场扫描...")
                symbols = get_all_swap_contracts()
                vol_list = []
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = {ex.submit(lambda s: fetch_klines_with_retry(s, BAR, 80), s): s for s in symbols}
                    for f in as_completed(futs):
                        s = futs[f]
                        try:
                            df = f.result()
                            if df is not None and len(df) >= 30:
                                p = df['c'].values.astype(np.float32)
                                vol_list.append({"symbol": s, "vol": calculate_volatility(p)})
                        except: pass
                log(f"📊 成功获取波动率数据的合约数: {len(vol_list)}")
                if vol_list:
                    df_vol = pd.DataFrame(vol_list).sort_values("vol", ascending=False).head(VOLATILITY_SAMPLE_SIZE)
                    top_vol_symbols = df_vol["symbol"].tolist()
                    log(f"💰 参与流动性过滤的合约数: {len(top_vol_symbols)}")
                    filtered = []
                    with ThreadPoolExecutor(max_workers=8) as ex:
                        futs_map = {s: (ex.submit(fetch_volume_usdt, s), ex.submit(fetch_market_cap, s)) for s in top_vol_symbols}
                        for s, (f_vol, f_mcap) in futs_map.items():
                            vol_usdt = f_vol.result()
                            mcap = f_mcap.result()
                            if vol_usdt >= MIN_VOLUME_USDT and mcap >= MIN_MARKET_CAP_USDT:
                                filtered.append(s)
                    log(f"💵 通过流动性过滤的合约数: {len(filtered)}")
                    candidates = filtered[:TOP_N]
                    log(f"🎯 本轮候选合约数: {len(candidates)}")

                    signals = []
                    with ThreadPoolExecutor(max_workers=8) as ex:
                        futs = {ex.submit(generate_signals_for_symbol, s): s for s in candidates}
                        for f in as_completed(futs):
                            sig, reason = f.result()
                            if sig:
                                signals.append(sig)

                    if signals:
                        df_sig = pd.DataFrame(signals).sort_values("score", ascending=False)
                        top_signals = df_sig.head(FINAL_PICK_N)
                        msg = ["✅ 极端反转信号 (激进放宽):"]
                        for _, row in top_signals.iterrows():
                            msg.append(f"#{row['symbol']} {row['signal'].upper()} 评分:{row['score']:.1f}")
                            msg.append(f"  现价:{row['current_price']:.6f}")
                            msg.append(f"  诊断: {row['detail']}")
                        push_telegram("\n".join(msg))
                        trader.set_pending_signals(top_signals.to_dict('records'))
                    else:
                        log("📉 本轮无极端反转信号")
                last_pred = now

            await asyncio.sleep(1)
        except KeyboardInterrupt:
            log("🛑 停止")
            break
        except Exception as e:
            log_system_exception(f"主循环异常: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log("手动停止")
        push_telegram("🛑 机器人停止")
