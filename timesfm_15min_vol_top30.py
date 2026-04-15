import numpy as np
import pandas as pd
import requests
import torch
import warnings
import time
import json
from datetime import datetime

# 屏蔽无关警告
warnings.filterwarnings('ignore')

# ==================== 【1. 核心配置区域】 ====================
# 企业微信参数
CORP_ID = "wwc4b0ff26e668aeb0"      
SECRET = "lB--gK5xsoZBLYgICXMOgVaju9dzunEkFrF_YBDOUqI"  
AGENT_ID = 1000002          

# 代理配置 (WSL 访问必需)
PROXIES = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

# 浏览器伪装
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# ==================== 【2. 还原策略参数】 ====================
BAR = "15m"                 
LIMIT = 160                 # 还原至您之前的回溯长度
HORIZON = 12
TOP_N_VOL = 50              # 候选池
FINAL_PICK_N = 2            # 最终入选数量
MIN_EXPECTED_RETURN = 0.0008  # 还原至 0.08%
MIN_R_SQUARED = 0.55
MIN_DIRECTION_CONFIDENCE = 0.55 # 还原至之前的置信度要求
OUTPUT_FILE = "/home/apt/timesfm_lab/signals_vps.json"
BASE_URL = "https://okx.cab" 

# ==================== 【3. 模型加载】 ====================
print("🚀 Loading TimesFM Model...")
torch.set_float32_matmul_precision("high")

def _build_model():
    try:
        import timesfm
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    except:
        return torch.hub.load('google/timesfm-2.5-200m-pytorch', 'from_pretrained', source='github')

model = _build_model()
print("✅ Model Ready.")

# ==================== 【4. 功能函数】 ====================
def push_wecom(content):
    """推送至企业微信"""
    try:
        t_url = f"https://qq.com{CORP_ID}&corpsecret={SECRET}"
        token_res = requests.get(t_url, proxies=PROXIES, headers=HEADERS, timeout=10).json()
        token = token_res.get("access_token")
        if not token: return
        
        send_url = f"https://qq.com{token}"
        payload = {
            "touser": "@all", "msgtype": "text", "agentid": AGENT_ID,
            "text": {"content": f"🚀 TimesFM 15m Signal\n----------------\n{content}"}
        }
        requests.post(send_url, json=payload, proxies=PROXIES, headers=HEADERS, timeout=10)
        print("✅ 信号已同步推送到企业微信")
    except Exception as e: print(f"Push Error: {e}")

def get_symbols():
    url = f"{BASE_URL}/api/v5/public/instruments"
    try:
        res = requests.get(url, params={"instType": "SWAP"}, proxies=PROXIES, headers=HEADERS, timeout=12)
        if res.status_code != 200: return []
        data = res.json().get("data", [])
        return [i["instId"] for i in data if i["settleCcy"] == "USDT" and i["state"] == "live"]
    except: return []

def fetch_data(instId, limit):
    """获取K线，严格保持与您原脚本一致的排序逻辑"""
    try:
        url = f"{BASE_URL}/api/v5/market/history-candles"
        params = {"instId": instId, "bar": BAR, "limit": str(limit)}
        res = requests.get(url, params=params, proxies=PROXIES, headers=HEADERS, timeout=8)
        if res.status_code != 200: return None
        data = res.json().get("data", [])
        if not data: return None
        
        # 还原排序逻辑：将逆序数据转为时间戳升序（旧 -> 新）
        df = pd.DataFrame(data, columns=["ts","o","h","l","c","v","vc","cv","confirm"])
        df["c"] = df["c"].astype(float)
        df["ts"] = df["ts"].astype(int)
        df = df.sort_values("ts").reset_index(drop=True)
        return df["c"].values.astype(np.float32)
    except: return None

def get_volatility(prices):
    if prices is None or len(prices) < 30: return 0
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 1000

def predict_and_score(instId):
    try:
        ts = fetch_data(instId, LIMIT)
        if ts is None or len(ts) < 50: return None
        current_price = float(ts[-1])

        with torch.no_grad():
            point, _ = model.forecast(horizon=HORIZON, inputs=[ts])
            f_vals = np.array(point, dtype=np.float32).flatten()
        
        exp_ret = (f_vals[-1] - current_price) / current_price
        
        # R2 路径平滑度
        steps = np.arange(len(f_vals))
        z = np.polyfit(steps, f_vals, 1)
        p = np.poly1d(z)
        r2 = 1 - (np.sum((f_vals - p(steps))**2) / np.sum((f_vals - np.mean(f_vals))**2))
        
        # 趋势一致性
        direction = 1 if exp_ret > 0 else -1
        diffs = np.diff(np.insert(f_vals, 0, current_price))
        consist = np.sum(np.sign(diffs) == direction) / len(diffs)
        conf = 0.7 * consist + 0.3 * max(0.0, min(1.0, r2))

        # 还原过滤门槛
        if abs(exp_ret) < MIN_EXPECTED_RETURN or conf < MIN_DIRECTION_CONFIDENCE:
            return None

        tp = max(np.floor(abs(exp_ret) * 100), 0.5)
        sl = max(round(tp * 0.5, 2), 0.3)
        score = abs(exp_ret) * 100 * 0.4 + r2 * 0.3 + consist * 0.3

        return {"symbol": instId, "side": "LONG" if exp_ret > 0 else "SHORT", "ret": exp_ret * 100, "r2": r2, "score": score, "tp": float(tp), "sl": float(sl)}
    except: return None

# ==================== 【5. 运行循环】 ====================
def run_cycle():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- New Cycle Starting ---")
    all_symbols = get_symbols()
    if not all_symbols:
        print("❌ No symbols fetched. Check Proxy.")
        return
        
    vols = []
    print(f"Scanning {len(all_symbols)} symbols...")
    for s in all_symbols:
        p = fetch_data(s, 80)
        vols.append({"symbol": s, "vol": get_volatility(p)})
    
    candidates = pd.DataFrame(vols).sort_values("vol", ascending=False).head(TOP_N_VOL)["symbol"].tolist()
    
    signals = []
    for s in candidates:
        res = predict_and_score(s)
        if res:
            signals.append(res)
            print(f"  [SIGNAL] {res['symbol']} Score: {res['score']:.4f}")

    if signals:
        top_picks = pd.DataFrame(signals).sort_values("score", ascending=False).head(FINAL_PICK_N)
        report_msg = ""
        json_vps = {}
        for i, row in top_picks.iterrows():
            line = f"#{i+1} {row['symbol']} | {row['side']} | Exp: {row['ret']:+.2f}% | TP:{row['tp']}% SL:{row['sl']}%"
            report_msg += line + "\n"
            json_vps[row['symbol']] = {"side": row['side'], "tp": row['tp'], "sl": row['sl']}
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(json_vps, f, indent=2, ensure_ascii=False)
            
        print(f"\nFinal Picks:\n{report_msg}")
        push_wecom(report_msg)
    else:
        print("No valid signals this round.")

if __name__ == "__main__":
    while True:
        try:
            run_cycle()
        except Exception as e:
            print(f"Runtime Error: {e}")
        print("\nWaiting 15 minutes...")
        time.sleep(900)
