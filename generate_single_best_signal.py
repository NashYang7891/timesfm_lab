import numpy as np
import requests
import torch
import json
import warnings
from datetime import datetime
from timesfm import TimesFM

warnings.filterwarnings('ignore')

# ================= 全局配置 =================
BAR = "1H"
LIMIT = 168
VOL_CALC_LIMIT = 120
HORIZON = 12
TOP_N = 30
OUTPUT_FILE = "signals_vps.json"

# ================= 评分算法 =================
def calculate_signal_quality(forecast_values, current_price):
    expected_return = (forecast_values[-1] - current_price) / current_price
    steps = np.arange(len(forecast_values))
    y = forecast_values
    z = np.polyfit(steps, y, 1)
    p = np.poly1d(z)
    y_hat = p(steps)
    y_bar = np.mean(y)
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y_bar)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    score = abs(expected_return) * r_squared
    return score, expected_return, r_squared

# ================= OKX API =================
def get_all_swap_contracts():
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SWAP"}
    try:
        data = requests.get(url, params=params, timeout=10).json()["data"]
        return [item["instId"] for item in data if item["settleCcy"] == "USDT" and item["state"] == "live"]
    except:
        return []

def fetch_klines(instId, limit):
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {"instId": instId, "bar": BAR, "limit": limit}
    try:
        data = requests.get(url, params=params, timeout=10).json()["data"]
        return np.array([float(x[4]) for x in data[::-1]])
    except:
        return None

# ================= 主程序 =================
def main():
    print(f"[{datetime.now()}] 初始化 TimesFM 模型...")

    # ======================
    # 强制 GPU（不可 fallback）
    # ======================
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到 CUDA GPU，程序退出")
    
    device = "cuda:0"
    print(f"✅ 强制使用 GPU: {torch.cuda.get_device_name(0)}")

    # ======================
    # 官方标准初始化（必须这样写）
    # ======================
    tfm = TimesFM(
        context_len=LIMIT,
        horizon_len=HORIZON,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend=device  # 直接在这里指定GPU，最稳定
    )
    tfm.load_from_checkpoint("google/timesfm-2.0-500m-pytorch")

    # 1. 扫描波动率
    symbols = get_all_swap_contracts()
    vol_list = []
    print(f"正在扫描 {len(symbols)} 个合约...")

    for s in symbols:
        prices = fetch_klines(s, VOL_CALC_LIMIT)
        if prices is not None and len(prices) >= 20:
            ret = np.diff(np.log(prices))
            vol = np.std(ret)
            vol_list.append({
                "instId": s,
                "vol": vol,
                "current_price": prices[-1],
                "history": prices
            })

    top_symbols = sorted(vol_list, key=lambda x: x["vol"], reverse=True)[:TOP_N]

    # 2. 预测
    candidate_signals = []
    for item in top_symbols:
        try:
            ts_data = item["history"][-LIMIT:]
            forecast_input = [ts_data]
            
            # 官方标准预测接口
            point_forecast, _, _ = tfm.forecast(forecast_input)
            forecast_path = point_forecast[0]

            score, ret_pct, r2 = calculate_signal_quality(forecast_path, item["current_price"])

            candidate_signals.append({
                "symbol": item["instId"],
                "side": "long" if ret_pct > 0 else "short",
                "score": score,
                "expected_return": ret_pct,
                "smoothness": r2
            })
        except Exception as e:
            print(f"⚠️ {item['instId']} 预测失败: {str(e)}")
            continue

    # 3. 输出信号
    if candidate_signals:
        best = max(candidate_signals, key=lambda x: x["score"])
        output = {
            "strategy": "TimesFM_GPU",
            "symbol": best["symbol"],
            "action": best["side"],
            "confidence": round(best["score"]*100, 4),
            "expected_12h_return": f"{round(best['expected_return']*100, 2)}%",
            "path_smooth_r2": round(best["smoothness"], 2),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"✅ 信号已生成：{best['symbol']} {best['side']}")
    else:
        print("❌ 无有效信号")

if __name__ == "__main__":
    main()
