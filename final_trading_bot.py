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
log_dir = "/home/apt/timesfm_lab"
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
# 代理已移除，VPS 直连

BAR = "3m"
HIGHER_BAR = "15m"
LIMIT = 900
HORIZON = 7

TOP_N = 50
FINAL_PICK_N = 2
MIN_EXPECTED_RETURN = 0.0003
MIN_R_SQUARED = 0.5
MIN_DIRECTION_CONFIDENCE = 0.65

RSI_PERIOD = 9
MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 5
RSI_LONG_THRESHOLD = 35
RSI_SHORT_THRESHOLD = 65

OUTPUT_FILE = f"{log_dir}/signals_vps.json"
REPORT_FILE = f"{log_dir}/signals_vps_report.json"
STRATEGY_POSITIONS_FILE = f"{log_dir}/strategy_positions.json"

API_KEY = "10d14cf0-79da-4597-9456-3aa1b88e1acf"
API_SECRET = "1B6A940855EC5787CD4E56BEF6D94733"
API_PASS = "kP9!vR2@mN5+"
IS_SANDBOX = False

PREDICTION_INTERVAL = 1500
MIN_BALANCE_USDT = 10.0
MAX_SINGLE_TRADE_USDT = 10
MAX_MARGIN_MULTIPLIER = 2

TAKE_PROFIT_PCT = 5.0
STOP_LOSS_PCT = 1.0
MAX_HOLD_SECONDS = 20 * 60

MIN_VOLUME_USDT = 10_000_000
MIN_MARKET_CAP_USDT = 20_000_000
VOLATILITY_SAMPLE_SIZE = 200

LEVERAGE = 3
PRICE_POSITION_RATIO = 0.1
FAVORABLE_MOVE_PCT = 0.2

MAX_CONCURRENT_POSITIONS = 2
MAX_TOTAL_MARGIN_RATIO = 0.5

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TRAILING_STOP_PCT = 2.0

# ==================== 3. Telegram推送（无代理） ====================
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

# ==================== 5. OKX数据获取（无代理） ====================
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

def check_technical_indicators(symbol, side, current_price):
    try:
        df = fetch_klines_with_retry(symbol, BAR, 100)
        if df is None or len(df) < 60:
            return True, f"数据不足，跳过指标检查 (当前价格: {current_price:.6f})"
        closes = df['c']
        rsi = compute_rsi(closes, RSI_PERIOD)
        macd_line, signal_line, histogram, hist_prev = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

        MACD_HIST_EPSILON = 0.0005
        side_cn = "多单" if side == 'long' else "空单"

        if side == 'long':
            if rsi >= RSI_LONG_THRESHOLD:
                return False, f"{side_cn} RSI={rsi:.1f} ≥ {RSI_LONG_THRESHOLD}，不符合多单条件 (当前价格: {current_price:.6f})"
        else:
            if rsi <= RSI_SHORT_THRESHOLD:
                return False, f"{side_cn} RSI={rsi:.1f} ≤ {RSI_SHORT_THRESHOLD}，不符合空单条件 (当前价格: {current_price:.6f})"

        if side == 'long':
            if histogram <= -MACD_HIST_EPSILON:
                return False, f"{side_cn} MACD柱状线={histogram:.4f} ≤ -{MACD_HIST_EPSILON}，动能过负 (当前价格: {current_price:.6f})"
        else:
            if histogram >= MACD_HIST_EPSILON:
                return False, f"{side_cn} MACD柱状线={histogram:.4f} ≥ {MACD_HIST_EPSILON}，动能过正 (当前价格: {current_price:.6f})"

        if side == 'long':
            if macd_line <= 0 or signal_line <= 0:
                return False, f"{side_cn} 快慢线不在零轴上方 (MACD={macd_line:.4f}, Signal={signal_line:.4f}) (当前价格: {current_price:.6f})"
        else:
            if macd_line >= 0 or signal_line >= 0:
                return False, f"{side_cn} 快慢线不在零轴下方 (MACD={macd_line:.4f}, Signal={signal_line:.4f}) (当前价格: {current_price:.6f})"

        df_higher = fetch_klines_with_retry(symbol, HIGHER_BAR, 100)
        if df_higher is not None and len(df_higher) >= 30:
            closes_higher = df_higher['c']
            macd_higher, signal_higher, _, _ = compute_macd(closes_higher, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            if side == 'long':
                if macd_higher <= signal_higher:
                    return False, f"{side_cn} 15分钟MACD死叉 (MACD={macd_higher:.4f} ≤ Signal={signal_higher:.4f})，方向不符 (当前价格: {current_price:.6f})"
            else:
                if macd_higher >= signal_higher:
                    return False, f"{side_cn} 15分钟MACD金叉 (MACD={macd_higher:.4f} ≥ Signal={signal_higher:.4f})，方向不符 (当前价格: {current_price:.6f})"

        desc = f"RSI={rsi:.1f}, MACD={macd_line:.4f}, Signal={signal_line:.4f}, Hist={histogram:.4f}"
        return True, f"技术指标通过: {desc} (当前价格: {current_price:.6f})"
    except Exception as e:
        err(f"技术指标计算异常 {symbol}: {e}")
        return True, f"指标计算异常，跳过检查 (当前价格: {current_price:.6f})"

# ==================== 7. 预测评分 ====================
def predict_and_score(instId):
    try:
        df = fetch_klines_with_retry(instId, BAR, LIMIT)
        if df is None or len(df) < 50:
            return None, "数据不足"
        ts = df['c'].values.astype(np.float32)
        current_price = float(ts[-1])

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

        direction_confidence = 0.7 * consistency + 0.3 * max(0.0, min(1.0, r_squared))

        signal_side = "long" if expected_return > 0 else "short"
        side_text = "多单" if signal_side == "long" else "空单"

        if abs(expected_return) < MIN_EXPECTED_RETURN:
            return None, f"{side_text}预期收益 {expected_return*100:.2f}% (绝对值) < {MIN_EXPECTED_RETURN*100:.2f}% (当前价格: {current_price:.6f})"
        if r_squared < MIN_R_SQUARED:
            return None, f"{side_text}R² {r_squared:.3f} < {MIN_R_SQUARED} (当前价格: {current_price:.6f})"
        if direction_confidence < MIN_DIRECTION_CONFIDENCE:
            return None, f"{side_text}方向置信度 {direction_confidence:.3f} < {MIN_DIRECTION_CONFIDENCE} (当前价格: {current_price:.6f})"

        tech_ok, tech_msg = check_technical_indicators(instId, signal_side, current_price)
        if not tech_ok:
            return None, tech_msg

        score = abs(expected_return) * 100 * 0.4 + r_squared * 0.3 + consistency * 0.3

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
            "signal": signal_side,
            "expected_return": expected_return,
            "r_squared": r_squared,
            "consistency": consistency,
            "direction_confidence": direction_confidence,
            "score": score,
            "last_price": current_price,
            "price_info": price_info,
            "tech_msg": tech_msg
        }
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
    log("📈 开始专业评分（含高级技术指标过滤）...")
    log("-" * 80)

    valid = []
    filtered_reasons = []
    for s in candidates:
        res, reason = predict_and_score(s)
        if res:
            valid.append(res)
            log(f"  [{len(valid)}] {res['symbol']}")
            log(f"      预期涨跌: {res['expected_return']*100:+.2f}%")
            log(f"      R²: {res['r_squared']:.2f} | 一致性: {res['consistency']:.2f} | 方向置信度: {res['direction_confidence']:.2f} | 得分: {res['score']:.4f}")
            log(f"      技术指标: {res['tech_msg']}")
        else:
            filtered_reasons.append(f"{s}: {reason}")
            log(f"  {s} 过滤: {reason}")

    if not valid:
        log("❌ 无符合条件信号")
        summary = "❌ 本轮无高质量交易信号\n\n候选池及过滤原因：\n" + "\n".join(filtered_reasons)
        push_telegram(summary)
        return {}

    df_results = pd.DataFrame(valid).sort_values("score", ascending=False)
    top = df_results.head(FINAL_PICK_N)

    msg = ["✅ 高质量交易信号（已通过技术指标确认）："]
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

    output_dict = {row['symbol']: row['signal'] for _, row in top.iterrows()}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    return output_dict

# ==================== 9. 交易模块（修复版，无代理） ====================
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
                    'trailing_activated': info.get('trailing_activated', False)
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
                    'trailing_activated': info.get('trailing_activated', False)
                }
            log(f"📂 已加载 {len(self.strategy_positions)} 个策略持仓记录")
        except Exception as e:
            err(f"加载策略持仓失败: {e}")

    def _init(self):
        log("🚀 初始化OKX交易客户端...")
        ex = ccxt.okx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASS,
            "enableRateLimit": True,
            "timeout": 30000,
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
                        }
                        self._save_strategy_positions()
                        log(f"🔄 接管孤儿持仓: {sym} {side.upper()} 已纳入管理")
                        push_telegram(f"🔄 策略接管持仓: {sym} {side.upper()}，现由程序自动管理")
                        self.pending_signals = [sig for sig in self.pending_signals if sig['ccxt_symbol'] != sym and sig['raw_symbol'] != sym]
        except Exception as e:
            err(f"同步策略持仓异常: {e}")

    def set_leverage(self, symbol, leverage=LEVERAGE):
        """使用签名请求设置杠杆，彻底避免参数错误"""
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

    def open_position(self, symbol, side, base_margin_usdt, ignore_price_position=False):
        current_positions = self.sync_positions()
        ccxt_symbol = self.exchange.market(symbol)['symbol']
        if ccxt_symbol in current_positions:
            log(f"⏸️ {symbol} 已有持仓 {current_positions[ccxt_symbol]}，拒绝重复开仓")
            return False

        if not ignore_price_position:
            ok, msg = self.check_price_position_entity(symbol, side)
            if not ok:
                log(f"⏸️ 跳过开仓 {symbol} {side}: {msg}")
                return False
        else:
            log(f"🚀 忽略实体位置检查，因有利移动触发开仓 {symbol} {side}")

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
            # 设置杠杆
            self.set_leverage(symbol)
            # 下单
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

            # 解析成交信息
            actual_open_price = order.get('average', price)
            actual_filled = order.get('filled', amount)
            if actual_filled == 0:
                info = order.get('info', {})
                actual_filled = float(info.get('filledSz', 0))
                actual_open_price = float(info.get('avgPx', price))
            if actual_filled == 0:
                err(f"订单未成交: {order}")
                # 尝试补救
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
                                stop_loss_price = actual_open_price - atr * ATR_MULTIPLIER
                            else:
                                stop_loss_price = actual_open_price + atr * ATR_MULTIPLIER
                        else:
                            if side == 'long':
                                stop_loss_price = actual_open_price * (1 - STOP_LOSS_PCT/100)
                            else:
                                stop_loss_price = actual_open_price * (1 + STOP_LOSS_PCT/100)
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
                            'trailing_activated': False
                        }
                        self._save_strategy_positions()
                        log(f"✅ 补救记录持仓 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_margin:.2f} | 止损: {stop_loss_price:.4f}")
                        push_telegram(f"✅ 策略开仓成功（补救）\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_margin:.2f}\n动态止损: ${stop_loss_price:.4f}")
                        return True
                return False

            actual_nominal = actual_filled * actual_open_price
            actual_used_margin = actual_nominal / LEVERAGE

            # 计算动态止损
            atr = self.get_atr(symbol)
            if atr is not None:
                if side == 'long':
                    stop_loss_price = actual_open_price - atr * ATR_MULTIPLIER
                else:
                    stop_loss_price = actual_open_price + atr * ATR_MULTIPLIER
                log(f"ATR={atr:.4f}, 动态止损价={stop_loss_price:.4f}")
            else:
                if side == 'long':
                    stop_loss_price = actual_open_price * (1 - STOP_LOSS_PCT/100)
                else:
                    stop_loss_price = actual_open_price * (1 + STOP_LOSS_PCT/100)
                log(f"ATR计算失败，使用固定止损 {STOP_LOSS_PCT}% -> {stop_loss_price:.4f}")

            self.strategy_positions[ccxt_symbol] = {
                'side': side,
                'open_price': actual_open_price,
                'open_time': time.time(),
                'open_qty': actual_filled,
                'open_margin': actual_used_margin,
                'open_nominal': actual_nominal,
                'stop_loss_price': stop_loss_price,
                'trailing_activated': False
            }
            self._save_strategy_positions()
            log(f"✅ 开仓成功 {ccxt_symbol} {side.upper()} {actual_filled} 张 @ {actual_open_price:.4f} | 保证金: ${actual_used_margin:.2f} | 止损: {stop_loss_price:.4f}")
            msg = f"✅ 策略开仓成功\n币种: {ccxt_symbol}\n方向: {side.upper()}\n数量: {actual_filled} 张\n价格: ${actual_open_price:.4f}\n保证金(逐仓): ${actual_used_margin:.2f}\n动态止损: ${stop_loss_price:.4f}"
            if is_adjusted:
                msg += f"\n⚠️ 保证金已从 {base_margin_usdt} 自动上调至 {adjusted_margin:.2f} USDT（满足最小张数）"
            for _ in range(3):
                if push_telegram(msg):
                    break
                time.sleep(1)
            return True
        except Exception as e:
            err(f"开仓失败 {symbol} {side}: {e}")
            # 尝试补救
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
                                stop_loss_price = actual_open_price - atr * ATR_MULTIPLIER
                            else:
                                stop_loss_price = actual_open_price + atr * ATR_MULTIPLIER
                        else:
                            if side == 'long':
                                stop_loss_price = actual_open_price * (1 - STOP_LOSS_PCT/100)
                            else:
                                stop_loss_price = actual_open_price * (1 + STOP_LOSS_PCT/100)
                        self.strategy_positions[ccxt_symbol] = {
                            'side': side,
                            'open_price': actual_open_price,
                            'open_time': time.time(),
                            'open_qty': actual_filled,
                            'open_margin': actual_margin,
                            'open_nominal': actual_filled * actual_open_price,
                            'stop_loss_price': stop_loss_price,
                            'trailing_activated': False
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

            # 更新跟踪价格极值
            if info['side'] == 'long':
                if current_price > info.get('highest_price', info['open_price']):
                    info['highest_price'] = current_price
                # 计算从最高价回撤的幅度
                peak = info['highest_price']
                drawdown_pct = (peak - current_price) / peak * 100
                # 触发跟踪止损条件：回撤 >= TRAILING_STOP_PCT
                if drawdown_pct >= TRAILING_STOP_PCT:
                    log(f"📉 触发跟踪止损: {sym} 多单，最高价 {peak:.4f}，当前价 {current_price:.4f}，回撤 {drawdown_pct:.2f}% >= {TRAILING_STOP_PCT}%")
                    self.close_position(sym, reason=f"跟踪止损（回撤 {drawdown_pct:.2f}%）")
                    closed_any = True
                    continue
            else:  # short
                if current_price < info.get('lowest_price', info['open_price']):
                    info['lowest_price'] = current_price
                # 计算从最低价反弹的幅度
                trough = info['lowest_price']
                bounce_pct = (current_price - trough) / trough * 100
                if bounce_pct >= TRAILING_STOP_PCT:
                    log(f"📈 触发跟踪止损: {sym} 空单，最低价 {trough:.4f}，当前价 {current_price:.4f}，反弹 {bounce_pct:.2f}% >= {TRAILING_STOP_PCT}%")
                    self.close_position(sym, reason=f"跟踪止损（反弹 {bounce_pct:.2f}%）")
                    closed_any = True
                    continue

            # 检查 ATR 初始止损（如果还没被跟踪止损处理）
            stop_price = info.get('stop_loss_price')
            if stop_price is not None:
                if (info['side'] == 'long' and current_price <= stop_price) or (info['side'] == 'short' and current_price >= stop_price):
                    log(f"💥 触发初始动态止损: {sym} 当前价 {current_price:.4f} 触及止损价 {stop_price:.4f}")
                    self.close_position(sym, reason=f"初始止损 {stop_price:.4f}")
                    closed_any = True
                    continue

            # 可选：打印当前状态
            pnl_percent = float(pos.get('percentage', 0))
            if abs(pnl_percent) < 1:
                pnl_percent = pnl_percent * 100
            hold_seconds = time.time() - info['open_time']
            log(f"📉 检查持仓 {sym}: 盈亏 {pnl_percent:.2f}%, 持仓时长 {hold_seconds/60:.1f}分钟")

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

    def set_pending_signals(self, signals, margin_amount):
        self.pending_signals = []
        for raw_symbol, side in signals.items():
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
                    'signal_price': signal_price
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
                success = self.open_position(raw_symbol, side, margin, ignore_price_position=True)
                if success:
                    to_remove.append(idx)
                else:
                    log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
            else:
                ok, msg = self.check_price_position_entity(raw_symbol, side)
                if ok:
                    log(f"🚀 待开仓信号价格满足实体位置条件: {raw_symbol} {side.upper()}")
                    success = self.open_position(raw_symbol, side, margin, ignore_price_position=False)
                    if success:
                        to_remove.append(idx)
                    else:
                        log(f"⚠️ 开仓失败 {raw_symbol} {side.upper()}，保留在待开仓队列中")
                else:
                    log(f"⏸️ 待开仓信号 {raw_symbol} {side.upper()} 价格不满足: {msg}")
        for idx in sorted(to_remove, reverse=True):
            self.pending_signals.pop(idx)

    def clear_pending_signals(self):
        self.pending_signals = []

# ==================== 10. 主程序 ====================
def main():
    trader = OKXTrader()
    last_pred = datetime.now() - timedelta(seconds=PREDICTION_INTERVAL)
    has_set_pending_this_cycle = False

    log("\n========== 全自动交易系统已启动 ==========")
    push_telegram(f"🤖 交易机器人启动\nK线: {BAR} | 预测: {HORIZON}根 ({HORIZON*3}分钟) | 每{PREDICTION_INTERVAL/60:.1f}分钟一轮\n止盈: +{TAKE_PROFIT_PCT}% | 动态止损: ATR({ATR_PERIOD})×{ATR_MULTIPLIER} | 最长持仓: {MAX_HOLD_SECONDS/60:.0f}分钟\n固定保证金: {MAX_SINGLE_TRADE_USDT} USDT/币\n流动性: 成交额≥{MIN_VOLUME_USDT/1_000_000:.0f}M, 市值≥{MIN_MARKET_CAP_USDT/1_000_000:.0f}M\n仓位模式: 逐仓 {LEVERAGE}x\n信号门槛: 置信度≥{MIN_DIRECTION_CONFIDENCE}, R²≥{MIN_R_SQUARED}, 预期收益≥{MIN_EXPECTED_RETURN*100:.1f}%\n技术指标: RSI周期{RSI_PERIOD} 多单<{RSI_LONG_THRESHOLD} 空单>{RSI_SHORT_THRESHOLD}; MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) 柱状图+零轴+15分钟验证\n风控: 最多{MAX_CONCURRENT_POSITIONS}仓, 总保证金≤{MAX_TOTAL_MARGIN_RATIO*100}%权益\n开仓条件: 实体位置(底部/顶部10%) 或 价格有利移动>{FAVORABLE_MOVE_PCT}%")

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

                signals = run_prediction_cycle()
                last_pred = now

                trader.clear_pending_signals()

                if signals and not has_set_pending_this_cycle:
                    current_positions_count = len(trader.strategy_positions)
                    if current_positions_count >= MAX_CONCURRENT_POSITIONS:
                        push_telegram(f"⚠️ 当前已有 {current_positions_count} 个策略持仓，达到上限 {MAX_CONCURRENT_POSITIONS}，本次信号暂不开仓")
                    else:
                        available_balance = trader.get_available_balance()
                        open_amount = MAX_SINGLE_TRADE_USDT
                        if available_balance < open_amount + 5:
                            push_telegram(f"⚠️ 可用余额不足 {open_amount} USDT（可用: {available_balance:.2f}），无法开仓")
                        else:
                            trader.set_pending_signals(signals, open_amount)
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
            time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n🛑 手动停止")
        if 'trader' in globals():
            trader.close_all()
        push_telegram("🛑 交易机器人已停止")
    except Exception as e:
        err(f"致命错误: {traceback.format_exc()}")
        push_telegram(f"❌ 机器人崩溃: {str(e)[:100]}")
