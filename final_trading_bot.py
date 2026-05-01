#!/usr/bin/env python3
"""
极端反转拐点单点捕捉系统 (最优实践版)
- 零延迟振荡器 + 动能衰减瞬间 + 成交量高对称区过滤
- 正向状态机触发，不使用 TimesFM 预测
- Telegram 推送 + 异步交易 (可关闭)
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
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from okx_ws import OKXWebSocket      # 自定义WebSocket客户端, 需自行实现

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
LIMIT = 200                    # 用于指标计算的K线数量

# 极端反转策略权重参数
EXTREME_PRICE_DEVIATION = 3.0          # 价格偏离EMA的ATR倍数阈值
EXTREME_VOLUME_IMBALANCE_RATIO = 0.75  # 主动买卖量比极端阈值(>0.75或<0.25)
MOMENTUM_DECAY_BARS = 5                # 动能衰减检测窗口(1m K线)
MOMENTUM_ACCEL_THRESHOLD = 0.0005      # 加速度过零阈值
VOLUME_PROFILE_LOOKBACK = 200          # VWAP计算K线数
VWAP_TOLERANCE_PCT = 1.5               # 价格在VWAP附近的容忍度(%)

# 原有风控参数
TOP_N = 50
FINAL_PICK_N = 3
SIGNAL_VALID_SECONDS = 60
MAX_HOLD_SECONDS = 900
AUTO_TRADE = False                     # 关闭自动交易，仅推送信号

RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 2.0
STOP_LOSS_PCT = 1.5

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

# ==================== 4. OKX数据获取 ====================
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

# ==================== 5. 零延迟振荡器与极端反转核心 ====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def compute_efficiency_ratio(prices, period=10):
    """高效价格动量比：净变化/路径长度，越接近1趋势越强"""
    if len(prices) < period + 1:
        return 0.5
    direction = abs(prices[-1] - prices[-period-1])
    volatility = np.sum(np.abs(np.diff(prices[-(period+1):])))
    if volatility == 0:
        return 0.0
    return direction / volatility

def compute_simple_phase(prices, period=20):
    """简化瞬时相位角（0-1），0超卖，1超买"""
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
    """主动买卖量比：阳线成交量占比"""
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
    """滚动VWAP"""
    if df is None or len(df) < period:
        return None
    recent = df.iloc[-period:].copy()
    typical_price = (recent['h'] + recent['l'] + recent['c']) / 3
    vwap = np.sum(typical_price * recent['v']) / recent['v'].sum()
    return vwap

def detect_momentum_decay(df, period=MOMENTUM_DECAY_BARS):
    """
    动能衰减检测：加速度过零 + 量价背离
    返回 (direction, strength)
      direction: 'bullish' (下跌衰竭), 'bearish' (上涨衰竭), None
    """
    if df is None or len(df) < period + 3:
        return None, 0
    closes = df['c'].values.astype(float)
    volumes = df['v'].values.astype(float)
    
    # 价格加速度（二阶差分）
    diffs = np.diff(closes[-(period+3):])
    accel = np.diff(diffs)
    current_accel = accel[-1]
    prev_accel = accel[-2] if len(accel) >= 2 else 0
    
    # 近期最低/最高量价
    seg_closes = closes[-(period+1):-1]
    seg_volumes = volumes[-(period+1):-1]
    low_idx = np.argmin(seg_closes)
    high_idx = np.argmax(seg_closes)
    vol_at_low = seg_volumes[low_idx]
    vol_at_high = seg_volumes[high_idx]
    current_vol = volumes[-1]
    
    # 下跌衰竭（做多信号）
    if (current_accel > 0 and prev_accel <= 0) or \
       (closes[-1] <= closes[-2] and current_vol < vol_at_low * 0.7):
        strength = min(1.0, abs(current_accel)*100 + (1 - current_vol/max(vol_at_low,1)))
        return 'bullish', strength
    
    # 上涨衰竭（做空信号）
    if (current_accel < 0 and prev_accel >= 0) or \
       (closes[-1] >= closes[-2] and current_vol < vol_at_high * 0.7):
        strength = min(1.0, abs(current_accel)*100 + (1 - current_vol/max(vol_at_high,1)))
        return 'bearish', strength
    
    return None, 0

def check_extreme_condition(symbol, current_price, df_5m, atr):
    """极端形态双确检测"""
    if df_5m is None or len(df_5m) < 30 or atr is None:
        return False, 0
    ema20 = df_5m['c'].ewm(span=20, adjust=False).mean().iloc[-1]
    deviation = (current_price - ema20) / atr if atr > 0 else 0
    imbalance = compute_volume_imbalance(df_5m, 20)
    extreme_imbalance = imbalance > EXTREME_VOLUME_IMBALANCE_RATIO or imbalance < (1 - EXTREME_VOLUME_IMBALANCE_RATIO)
    if abs(deviation) > EXTREME_PRICE_DEVIATION and extreme_imbalance:
        severity = min(abs(deviation)/5, 1.0)*0.6 + (abs(imbalance-0.5)*2)*0.4
        return True, severity
    return False, 0

def extreme_reversal_signal(symbol, current_price):
    """
    极端拐点综合判定（最优实践正向推导）
    返回 dict 或 None
    """
    df_5m = fetch_klines_with_retry(symbol, BAR, 100)
    if df_5m is None or len(df_5m) < 60:
        return None
    df_1m = fetch_klines_with_retry(symbol, "1m", 60)
    if df_1m is None or len(df_1m) < 30:
        df_1m = df_5m  # 降级使用
    
    # ATR
    high = df_5m['h'].values
    low = df_5m['l'].values
    close = df_5m['c'].values
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else None
    
    # 极端形态
    is_extreme, extreme_sev = check_extreme_condition(symbol, current_price, df_5m, atr)
    if not is_extreme:
        return None
    
    # 多振荡器加权
    efficiency = compute_efficiency_ratio(df_5m['c'].values, 10)
    phase = compute_simple_phase(df_5m['c'].values, 20)
    rsi_val = compute_rsi(df_5m['c'], RSI_PERIOD)
    composite_oscil = 0.4*efficiency + 0.3*phase + 0.3*(rsi_val/100)
    
    # 动能衰减
    decay_dir, decay_strength = detect_momentum_decay(df_1m)
    if decay_dir is None:
        return None
    
    # VWAP过滤
    vwap = compute_vwap(df_5m, VOLUME_PROFILE_LOOKBACK)
    vwap_distance = 0
    if vwap is not None:
        vwap_distance = abs(current_price - vwap) / vwap * 100
        if vwap_distance > VWAP_TOLERANCE_PCT:
            decay_strength *= 0.6  # 偏离高成交量区削弱信号
    
    # 方向判定
    direction = None
    if decay_dir == 'bullish' and composite_oscil < 0.3:
        direction = 'long'
    elif decay_dir == 'bearish' and composite_oscil > 0.7:
        direction = 'short'
    
    if direction is None:
        return None
    
    score = (decay_strength*0.5 + extreme_sev*0.3 + (1 - abs(composite_oscil-0.5)*2)*0.2) * 100
    trigger_zone = (current_price * 0.998, current_price * 1.002)
    
    return {
        'symbol': symbol,
        'signal': direction,
        'score': score,
        'current_price': current_price,
        'trigger_zone': trigger_zone,
        'detail': f"极端反转 | 振荡器={composite_oscil:.2f} | 衰竭方向={decay_dir} | 极端度={extreme_sev:.2f} | VWAP偏离={vwap_distance:.1f}%",
        'expected_return': 0.015 if direction == 'long' else -0.015,
        'estimated_hold_minutes': 10
    }

# ==================== 6. 信号生成（无TimesFM） ====================
def generate_signals_for_symbol(symbol):
    """为单个合约生成极端反转信号"""
    try:
        ticker_url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
        ticker_data = requests.get(ticker_url, timeout=5).json()
        if ticker_data.get('code') != '0':
            return None, "ticker获取失败"
        current_price = float(ticker_data['data'][0]['last'])
        
        signal_info = extreme_reversal_signal(symbol, current_price)
        if signal_info is None:
            return None, "无极端反转信号"
        return signal_info, ""
    except Exception as e:
        return None, f"异常: {str(e)[:50]}"

# ==================== 7. 异步交易类（保留原有风控，增加极端信号处理） ====================
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
        await self.exchange.load_markets(reload=True)
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
        if not info: return
        stop_price = info.get('stop_loss_price')
        if stop_price is None: return
        side = info['side']
        if (side == 'long' and mark_price <= stop_price) or (side == 'short' and mark_price >= stop_price):
            log(f"💥 WebSocket触发止损: {symbol} 标记价{mark_price:.6f} 触及{stop_price:.6f}")
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
                if contracts == 0: continue
                pos_side = p.get('info', {}).get('posSide')
                side = pos_side if pos_side in ['long', 'short'] else ('long' if contracts > 0 else 'short')
                pos_map[p['symbol']] = side
            return pos_map
        except Exception as e:
            err(f"同步持仓失败: {e}")
            return {}

    def get_atr_sync(self, symbol):
        try:
            df = fetch_klines_with_retry(symbol, BAR, ATR_PERIOD+1)
            if df is None or len(df) < ATR_PERIOD+1: return None
            high, low, close = df['h'].values, df['l'].values, df['c'].values
            tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            return np.mean(tr[-ATR_PERIOD:])
        except Exception as e:
            err(f"计算ATR失败 {symbol}: {e}")
            return None

    async def open_position(self, symbol, side, expected_return, hold_minutes, signal_price=None, extreme_signal=False):
        async with self.position_lock:
            current_pos = await self.sync_positions()
            ccxt_symbol = self.exchange.market(symbol)['symbol']
            if ccxt_symbol in current_pos:
                log(f"⏸️ {symbol} 已有持仓，拒绝重复开仓")
                return False
            available = await self.get_available_balance()
            margin_usdt = available * USE_PERCENT_OF_AVAILABLE
            if margin_usdt < MIN_BALANCE_USDT:
                log(f"💰 保证金不足 {margin_usdt:.2f} < {MIN_BALANCE_USDT}")
                return False
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # 极端信号可跳过一些位置过滤，此处简化保留
            atr = self.get_atr_sync(symbol)
            if atr is not None:
                stop_loss_price = current_price - atr*ATR_MULTIPLIER if side == 'long' else current_price + atr*ATR_MULTIPLIER
            else:
                stop_loss_price = current_price*(1 - STOP_LOSS_PCT/100) if side == 'long' else current_price*(1 + STOP_LOSS_PCT/100)

            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1.0)
            amount = (margin_usdt * LEVERAGE) / (current_price * contract_size)
            amount = self.exchange.amount_to_precision(symbol, amount)
            amount = float(amount)

            try:
                await self.exchange.set_leverage(LEVERAGE, symbol, params={'mgnMode': 'isolated', 'posSide': side})
                order = await self.exchange.create_order(
                    symbol=symbol, type='market', side='buy' if side=='long' else 'sell',
                    amount=amount, params={'positionSide': side, 'tdMode': 'isolated'}
                )
                actual_price = order.get('average', current_price)
                actual_filled = order.get('filled', amount) or float(order.get('info', {}).get('filledSz', 0))
                if actual_filled == 0:
                    err(f"订单未成交: {order}")
                    return False

                info = {
                    'side': side,
                    'open_price': actual_price,
                    'open_time': time.time(),
                    'open_qty': actual_filled,
                    'open_margin': actual_filled * actual_price / LEVERAGE,
                    'stop_loss_price': stop_loss_price,
                    'highest_price': actual_price,
                    'lowest_price': actual_price,
                    'trailing_stop_pct': TRAILING_STOP_PCT,
                    'trailing_activated': False,
                    'expected_return': expected_return,
                    'max_hold_seconds': MAX_HOLD_SECONDS
                }
                self.strategy_positions[ccxt_symbol] = info
                self._save_strategy_positions()
                inst_id = self.exchange.market(ccxt_symbol)['id']
                await self.ws_client.subscribe_mark_price([inst_id])
                push_telegram(f"✅ 开仓成功\n{symbol} {side.upper()}\n价格: {actual_price:.6f}\n止损: {stop_loss_price:.6f}")
                log(f"✅ 开仓 {ccxt_symbol} {side} {actual_filled}张 @ {actual_price:.4f}")
                return True
            except Exception as e:
                err(f"开仓异常 {symbol}: {e}")
                return False

    async def close_position(self, symbol, reason=""):
        # ... 保留原有平仓实现，此处简化
        async with self.position_lock:
            info = self.strategy_positions.pop(symbol, None)
            if not info: return False
            try:
                positions = await self.exchange.fetch_positions([symbol])
                real_pos = next((p for p in positions if p['side']==info['side'] and float(p.get('contracts',0))>0), None)
                if not real_pos:
                    log(f"⚠️ 无实际持仓 {symbol}，清理记录")
                    return True
                amount = abs(float(real_pos['contracts']))
                close_price = float(real_pos.get('last') or real_pos.get('markPrice') or 0)
                if close_price == 0:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    close_price = ticker['last']
                await self.exchange.create_order(
                    symbol, 'market', 'sell' if info['side']=='long' else 'buy',
                    amount, params={'reduceOnly': True, 'tdMode': 'isolated'}
                )
                pnl = (close_price - info['open_price']) * info['open_qty'] if info['side']=='long' else (info['open_price'] - close_price) * info['open_qty']
                log(f"🔻 平仓 {symbol} 盈亏: {pnl:+.2f} | {reason}")
                push_telegram(f"🔻 平仓\n{symbol} {info['side'].upper()}\n盈亏: {pnl:+.2f} USDT\n原因: {reason}")
                self._save_strategy_positions()
                return True
            except Exception as e:
                err(f"平仓失败 {symbol}: {e}")
                return False

    def _save_strategy_positions(self):
        with open(STRATEGY_POSITIONS_FILE, 'w') as f:
            json.dump(self.strategy_positions, f, indent=2, default=str)

    def set_pending_signals(self, signals_list):
        self.pending_signals = []
        for sig in signals_list:
            self.pending_signals.append({
                'raw_symbol': sig['symbol'],
                'side': sig['signal'],
                'signal_price': sig['current_price'],
                'expected_return': sig['expected_return'],
                'expected_hold_minutes': sig['estimated_hold_minutes'],
                'timestamp': time.time(),
                'extreme': True,
                'trigger_zone': sig.get('trigger_zone')
            })
        log(f"📋 设置待开仓极端信号: {len(self.pending_signals)} 个")

    async def check_and_open_pending(self):
        if not AUTO_TRADE:
            if self.pending_signals:
                log(f"⏸️ 自动交易关闭，跳过 {len(self.pending_signals)} 个信号")
                self.pending_signals.clear()
            return
        if not self.pending_signals: return
        now = time.time()
        to_remove = []
        for idx, sig in enumerate(self.pending_signals):
            if now - sig['timestamp'] > SIGNAL_VALID_SECONDS:
                to_remove.append(idx)
                continue
            try:
                ticker = await self.exchange.fetch_ticker(sig['raw_symbol'])
                current_price = ticker['last']
            except:
                continue
            # 正向状态机：价格在触发区内立即开仓
            if sig.get('trigger_zone'):
                low_z, high_z = sig['trigger_zone']
                if not (low_z <= current_price <= high_z):
                    continue
            ccxt_symbol = self.exchange.market(sig['raw_symbol'])['symbol']
            if ccxt_symbol in await self.sync_positions():
                to_remove.append(idx)
                continue
            success = await self.open_position(
                sig['raw_symbol'], sig['side'], sig['expected_return'],
                sig['expected_hold_minutes'], signal_price=sig['signal_price'],
                extreme_signal=True
            )
            if success:
                to_remove.append(idx)
        for idx in sorted(to_remove, reverse=True):
            self.pending_signals.pop(idx)

# ==================== 8. 主循环 ====================
async def main_async():
    trader = OKXTraderAsync()
    await trader.init()
    last_pred = datetime.now() - timedelta(seconds=60)

    log("========== 极端反转拐点捕捉系统启动 | 杠杆3x | 仅推送信号 ==========")
    push_telegram("🤖 极端反转系统启动 (自动交易关闭)")

    while True:
        try:
            now = datetime.now()
            await trader.check_and_open_pending()
            # 可加入定期持仓检查等

            if (now - last_pred).total_seconds() >= 60:
                # 获取高波动候选
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
                if vol_list:
                    df_vol = pd.DataFrame(vol_list).sort_values("vol", ascending=False).head(VOLATILITY_SAMPLE_SIZE)
                    top_vol_symbols = df_vol["symbol"].tolist()
                    # 过滤成交额/市值
                    filtered = []
                    with ThreadPoolExecutor(max_workers=8) as ex:
                        futs_map = {s: (ex.submit(fetch_volume_usdt, s), ex.submit(fetch_market_cap, s)) for s in top_vol_symbols}
                        for s, (f_vol, f_mcap) in futs_map.items():
                            vol_usdt = f_vol.result()
                            mcap = f_mcap.result()
                            if vol_usdt >= MIN_VOLUME_USDT and mcap >= MIN_MARKET_CAP_USDT:
                                filtered.append(s)
                    candidates = filtered[:TOP_N]

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
                        msg = ["✅ 极端反转信号:"]
                        for _, row in top_signals.iterrows():
                            msg.append(f"#{row['symbol']} {row['signal'].upper()} 评分:{row['score']:.1f}")
                            msg.append(f"  价格:{row['current_price']:.6f} | {row['detail']}")
                        push_telegram("\n".join(msg))
                        trader.set_pending_signals(top_signals.to_dict('records'))
                    else:
                        push_telegram("📉 无极端反转信号")
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
