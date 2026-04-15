import numpy as np
import pandas as pd
import requests
import torch
import timesfm
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ================= 全局配置 =================
BAR = "1H"                  # 1小时周期
LIMIT = 168                 # 历史长度缩短至168根（最近7天），提高灵敏度
VOL_CALC_LIMIT = 120        # 波动率计算：最近5天
HORIZON = 12                # 预测未来12小时
TOP_N = 30                  # 筛选波动率前30
OUTPUT_FILE = "signals_vps.json"

# ================= 核心评分算法 =================
def calculate_signal_quality(forecast_values, current_price):
    """
    分析 TimesFM 预测路径的质量
    score = 预期收益率 * 路径平滑度 (R-squared)
    """
    # 1. 计算总预期收益率
    expected_return = (forecast_values[-1] - current_price) / current_price
    
    # 2. 计算路径平滑度 (R-squared)
    # 衡量预测路径是否为单边趋势，避免选出上下乱跳的震荡币
    steps = np.arange(len(forecast_values))
    y = forecast_values
    
    # 线性回归拟合
    z = np.polyfit(steps, y, 1)
    p = np.poly1d(z)
    y_hat = p(steps)
    y_bar = np.mean(y)
    
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y_bar)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 3. 综合评分 (绝对值收益 * 平滑度)
    score = abs(expected_return) * r_squared
    return score, expected_return, r_squared

# ================= 工具函数 =================
def get_all_swap_contracts():
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SWAP"}
    try:
        data = requests.get(url, params=params, timeout=10).json()["data"]
        return [item["instId"] for item in data if item["settleCcy"] == "USDT" and item["state"] == "live"]
    except: return []

def fetch_klines(instId, limit):
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {"instId": instId, "bar": BAR, "limit": limit}
    try:
        data = requests.get(url, params=params, timeout=10).json()["data"]
        # OKX返回是 [ts, open, high, low, close, vol...]，close在索引4
        return np.array([float(x[4]) for x in data[::-1]]) # 转为正序
    except: return None

# ================= 主程序 =================
def main():
    print(f"[{datetime.now()}] 正在初始化 TimesFM 模型...")
    tfm = timesfm.TimesFm(
        context_len=LIMIT,
        horizon_len=HORIZON,
        input_patch_size=32,
        output_patch_size=128,
        num_layers=20,
        model_dims=1280,
        backend="gpu", # 如果没GPU请改为cpu
    )
    tfm.load_from_checkpoint("google/timesfm-2.0-500m-pytorch") # 请确保路径正确

    # 1. 获取全市场币种并计算波动率筛选 Top N
    symbols = get_all_swap_contracts()
    vol_list = []
    print(f"正在扫描 {len(symbols)} 个合约的波动率...")
    
    for s in symbols:
        prices = fetch_klines(s, VOL_CALC_LIMIT)
        if prices is not None and len(prices) > 10:
            vol = np.std(np.diff(np.log(prices))) # 对数收益率标准差
            vol_list.append({"instId": s, "vol": vol, "current_price": prices[-1], "history": prices})
    
    top_symbols = sorted(vol_list, key=lambda x: x["vol"], reverse=True)[:TOP_N]

    # 2. 对 Top 30 进行 TimesFM 推理并评分
    candidate_signals = []
    print(f"对波动率前 {TOP_N} 进行 TimesFM 趋势推理...")

    for item in top_symbols:
        # 准备数据，TimesFM 要求输入是 List[np.array]
        forecast_input = [item["history"][-LIMIT:]] 
        # 执行推理
        point_forecast, _, _ = tfm.forecast(forecast_input)
        forecast_path = point_forecast[0] # 取出预测数组
        
        # 计算质量评分
        score, ret_pct, r2 = calculate_signal_quality(forecast_path, item["current_price"])
        
        candidate_signals.append({
            "symbol": item["instId"],
            "side": "long" if ret_pct > 0 else "short",
            "score": score,
            "expected_return": ret_pct,
            "smoothness": r2
        })

    # 3. 选出最强且最稳的一个币种
    if candidate_signals:
        best_signal = max(candidate_signals, key=lambda x: x["score"])
        
        # 格式化输出给 AlgoSE 机器人
        final_json = {
            "strategy_name": "TimesFM_Single_Best",
            "target_symbol": best_signal["symbol"],
            "action": best_signal["side"],
            "confidence": round(best_signal["score"] * 100, 4), # 放大分数便于观察
            "metrics": {
                "expected_return_12h": f"{round(best_signal['expected_return']*100, 2)}%",
                "path_r2": round(best_signal["smoothness"], 2),
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # 写入 VPS 同步文件
        with open(OUTPUT_FILE, "w") as f:
            json.dump(final_json, f, indent=4)
        
        print(f"✅ 最佳信号已锁定: {best_signal['symbol']} ({best_signal['side']}) Score: {final_json['confidence']}")
    else:
        print("❌ 未能生成有效信号")

if __name__ == "__main__":
    main()
