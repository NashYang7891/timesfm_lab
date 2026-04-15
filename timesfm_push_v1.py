import numpy as np
import pandas as pd
import requests
import torch
import warnings
from datetime import datetime
import json
import time
import traceback
import sys
import timesfm

# WSL 兼容性：强制标准输出为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# ==================== 【1. 核心配置】 ====================
# 📡 Telegram 推送配置 (请替换为你的 Bot Token 和 Chat ID)
TG_BOT_TOKEN = "8722422674:AAGrKmRurQ2G__j-Vxbh5451v0e9_u97CQY"  # 例: 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TG_CHAT_ID = "5372217316"              # 例: 123456789 (个人ID或群ID加负号如 -1001234567890)
# 若服务器在国内，Telegram API 通常也需要走代理。若不需要则留 None
TG_PROXIES = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"} 

# 📊 策略参数 (已按你的要求调整)
BAR = "3m"            # 拉取 3分钟 K线
LIMIT = 512           # 历史 512 根
HORIZON = 10          # 预测未来 10 根 (3m * 10 = 30分钟)
TOP_N = 50            # 波动率初筛数量
FINAL_PICK_N = 2      # 最终推送数量
MIN_EXPECTED_RETURN = 0.0008
MIN_R_SQUARED = 0.55
MIN_DIRECTION_CONFIDENCE = 0.58
OUTPUT_FILE = "signals_vps.json"       # 保留供 VPS 交易机器人读取
REPORT_FILE = "signals_vps_report.json"

OKX_PROXIES = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ==================== 【2. Telegram 推送】 ====================
def push_telegram(content):
    if TG_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("⚠️ 未配置 TG_BOT_TOKEN，跳过推送")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": content,
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, proxies=TG_PROXIES, timeout=10)
        ret = res.json()
        if ret.get("ok"):
            print("✅ Telegram 推送成功")
            return True
        else:
            print(f"⚠️ Telegram 推送失败: {ret.get('description')}")
            return False
    except Exception as e:
        print(f"⚠️ Telegram 推送异常: {e}")
        return False

# ==================== 【3. 模型加载】 ====================
print("🚀 加载 TimesFM 模型...")
torch.set_float32_matmul_precision("high")

def _build_timesfm_model():
    errors = []
    try:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        try:
            fc = timesfm.ForecastConfig(max_context=8192, max_horizon=512)
            m.compile(forecast_config=fc)
        except Exception: pass
        return m
    except Exception as e:
        errors.append(f"TimesFM_2p5_200M_torch 路径失败: {e}")
    try:
        m = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu" if torch.cuda.is_available() else "cpu", per_core_batch_size=32, horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.5-200m-pytorch"),
        )
        return m
    except Exception as e:
        errors.append(f"TimesFm 经典接口失败: {e}")
    raise RuntimeError("TimesFM 初始化失败:\n- " + "\n- ".join(errors))

model = _build_timesfm_model()
print("✅ 模型加载完成")

# ==================== 【4. 网络请求 (OKX)】 ====================
def get_all_swap_contracts():
    try:
        url = "https://www.okx.com/api/v5/public/instruments"
        res = requests.get(url, params={"instType": "SWAP"}, proxies=OKX_PROXIES, headers=HEADERS, timeout=15).json()
        if res.get("code") != "0": return []
        return [i["instId"] for i in res.get("data", []) if i.get("settleCcy") == "USDT" and i.get("state") == "live"]
    except Exception:
        return []

def fetch_klines(instId, limit):
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        r = requests.get(url, params={"instId": instId, "bar": BAR, "limit": limit},
                         proxies=OKX_PROXIES, headers=HEADERS, timeout=15).json()
        if r.get("code") != "0" or not r.get("data"): return None
        df = pd.DataFrame(r['data'], columns=['ts','o','h','l','c','vol','volCcy','volCcyQuote','confirm'])
        # OKX 返回 [最新, ..., 最旧]
        closes = pd.to_numeric(df['c'], errors='coerce').values.astype(np.float32)
        return closes
    except Exception:
        return None

def calculate_volatility(prices):
    if len(prices) < 30: return 0.0
    ret = np.diff(prices) / prices[:-1]
    return float(np.std(ret) * 1000)

# ==================== 【5. 核心预测与评分】 ====================
def predict_and_score(instId):
    try:
        ts_okx = fetch_klines(instId, LIMIT)
        if ts_okx is None or len(ts_okx) < 50: return None
        
        current_price = float(ts_okx[0])  # ✅ 索引0为最新价
        ts_model = ts_okx[::-1]           # 🤖 模型需正序输入

        with torch.no_grad():
            try:
                point, _ = model.forecast(horizon=HORIZON, inputs=[ts_model])
                forecast_values = np.array(point[0], dtype=np.float32)
            except TypeError:
                out = model.forecast(inputs=[ts_model], freq=[0])
                forecast_values = np.array(out[0][0] if isinstance(out[0], (list, tuple)) else out[0], dtype=np.float32)
            except Exception:
                out = model.forecast(inputs=[ts_model])
                forecast_values = np.array(out[0][0] if isinstance(out[0], (list, tuple)) else out[0], dtype=np.float32)

        final_forecast = forecast_values[-1]
        expected_return = (final_forecast - current_price) / current_price

        steps = np.arange(len(forecast_values))
        z = np.polyfit(steps, forecast_values, 1)
        p = np.poly1d(z)
        y_average = np.mean(forecast_values)
        ss_res = np.sum((forecast_values - p(steps)) ** 2)
        ss_tot = np.sum((forecast_values - y_average) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        diffs = np.diff(np.insert(forecast_values, 0, current_price))
        direction = 1 if expected_return > 0 else -1
        consistency = np.sum(np.sign(diffs) == direction) / len(diffs)
        direction_confidence = 0.7 * consistency + 0.3 * max(0.0, min(1.0, r_squared))

        if (abs(expected_return) < MIN_EXPECTED_RETURN or r_squared < MIN_R_SQUARED or direction_confidence < MIN_DIRECTION_CONFIDENCE):
            return None

        score = abs(expected_return) * 100 * 0.4 + r_squared * 0.3 + consistency * 0.3
        signal = "LONG" if expected_return > 0 else "SHORT"
        expected_pct = abs(expected_return) * 100
        tp_pct = max(np.floor(expected_pct), 0.5)
        sl_pct = max(round(tp_pct * 0.5, 2), 0.3)

        tp_price = current_price * (1 + tp_pct / 100) if signal == "LONG" else current_price * (1 - tp_pct / 100)
        sl_price = current_price * (1 - sl_pct / 100) if signal == "LONG" else current_price * (1 + sl_pct / 100)

        return {
            "symbol": instId, "signal": signal, "expected_return": float(expected_return),
            "r_squared": float(r_squared), "consistency": float(consistency),
            "direction_confidence": float(direction_confidence), "score": float(score),
            "last_price": float(current_price), "tp_pct": float(tp_pct), "sl_pct": float(sl_pct),
            "tp_price": float(tp_price), "sl_price": float(sl_price),
        }
    except Exception:
        return None

# ==================== 【6. 主流程】 ====================
def run_cycle():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"⏰ [{now_str}] 策略：{BAR}周期 | 预测未来{HORIZON}根({HORIZON*3}分钟) | 循环:10分钟")
    print(f"{'='*60}")

    symbols = get_all_swap_contracts()
    if not symbols:
        print("❌ 未获取到合约列表，跳过本轮")
        return
    print(f"✅ 合约总数：{len(symbols)} | 开始波动率筛选...")

    vol_list = []
    for s in symbols:
        try:
            time.sleep(0.15)
            p = fetch_klines(s, 80)
            if p is not None:
                vol_list.append({"symbol": s, "vol": calculate_volatility(p)})
        except Exception:
            continue

    if not vol_list:
        print("⚠️ 波动率数据不足，跳过本轮")
        return

    df_vol = pd.DataFrame(vol_list).sort_values("vol", ascending=False).head(TOP_N)
    candidate_list = df_vol["symbol"].tolist()
    print(f"🏆 候选池：波动率Top{len(candidate_list)}\n")

    print("📈 开始专业评分...")
    valid_candidates = []
    for s in candidate_list:
        res = predict_and_score(s)
        if res:
            valid_candidates.append(res)
            print(f"  [{len(valid_candidates)}] {res['symbol']} | 得分: {res['score']:.4f}")

    # 构建推送消息 (严格对齐你要求的格式)
    if not valid_candidates:
        msg = "ℹ️ 本轮市场波动/置信度不足，无高胜率信号。"
        print(f"\n{msg}")
        push_telegram(msg)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return

    df_results = pd.DataFrame(valid_candidates).sort_values("score", ascending=False).reset_index(drop=True)
    top_picks = df_results.head(FINAL_PICK_N)

    # 拼接消息
    msg_lines = [f"🏆 最终入选前{len(top_picks)}名："]
    for i, row in top_picks.iterrows():
        price_fmt = ".4f" if row['last_price'] < 10 else ".2f"
        line = (
            f"  #{i+1} {row['symbol']} | {row['signal']} | "
            f"预:{row['expected_return']*100:+.2f}% | R²:{row['r_squared']:.2f} | "
            f"置信:{row['direction_confidence']:.2f} | 得分:{row['score']:.4f}\n"
            f"      💰 现价:{row['last_price']:{price_fmt}} | 止盈:{row['tp_price']:{price_fmt}} | 止损:{row['sl_price']:{price_fmt}}"
        )
        print(line)
        msg_lines.append(line)

    final_msg = "\n".join(msg_lines)
    print(f"\n📤 正在推送至 Telegram...")
    push_telegram(final_msg)

    # 💾 保留 JSON 供 VPS 交易机器人同步读取
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(top_picks[["symbol", "signal", "last_price", "tp_price", "sl_price"]].to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(top_picks.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print("💾 结果已保存至 signals_vps.json & report.json")

# ==================== 【7. 主循环 (10分钟)】 ====================
if __name__ == "__main__":
    print("🌐 启动自动扫描脚本... (按 Ctrl+C 停止)")
    try:
        while True:
            run_cycle()
            print(f"\n⏳ 等待下一轮扫描 (600秒)...")
            time.sleep(600) 
    except KeyboardInterrupt:
        print("\n🛑 用户手动终止脚本，安全退出。")
    except Exception as e:
        print(f"💥 未知崩溃: {e}")
        traceback.print_exc()
