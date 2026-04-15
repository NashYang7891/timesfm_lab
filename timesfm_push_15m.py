import numpy as np
import pandas as pd
import requests
import torch
import warnings
import time
import json
import subprocess
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 【1. 微信推送：IP直连稳定版】 ====================
def push_wecom(content):
    corpid = "wwc4b0ff26e668aeb0"
    secret = "lB--gK5xsoZBLYgICXMOgVaju9dzunEkFrF_YBDOUqI"
    agentid = 1000002
    proxy = "http://127.0.0.1:10809"
    # 企业微信 API 的直连 IP
    token_url = f"https://101.91.57{corpid}&corpsecret={secret}"
    try:
        # 1. 获取 Token
        token_cmd = ['curl', '-s', '-k', '-x', proxy, '-H', 'Host: ://qq.com', token_url]
        token_res = subprocess.check_output(token_cmd).decode('utf-8')
        token = json.loads(token_res).get("access_token")
        if not token: return

        # 2. 发送消息
        send_url = f"https://101.91.57{token}"
        msg_json = json.dumps({
            "touser": "@all", "msgtype": "text", "agentid": agentid,
            "text": {"content": f"🚀 TimesFM 信号推送\n----------------\n{content}"}
        }, ensure_ascii=False)
        send_cmd = ['curl', '-s', '-k', '-x', proxy, '-H', 'Host: ://qq.com', '-H', 'Content-Type: application/json', '-d', msg_json, send_url]
        subprocess.check_output(send_cmd)
        print("✅ 信号已推送到企业微信")
    except: pass

# ==================== 【2. 策略参数：还原原始最稳配置】 ====================
BAR = "5m"
LIMIT = 512
HORIZON = 12
TOP_N = 50
FINAL_PICK_N = 2
MIN_EXPECTED_RETURN = 0.0008
PROXIES = {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}

# ==================== 【3. 模型加载：本地模式】 ====================
print("🚀 加载 TimesFM 模型 (本地加载模式)...")
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
print("✅ 模型加载完成")

# ==================== 【4. 数据获取与预测】 ====================
def get_all_swap_contracts():
    # 换回你原本最稳的域名
    url = "https://okx.com"
    try:
        res = requests.get(url, params={"instType": "SWAP"}, proxies=PROXIES, timeout=10).json()
        return [item["instId"] for item in res['data'] if item["settleCcy"] == "USDT" and item["state"] == "live"]
    except: return []

def fetch_data(instId):
    try:
        url = "https://okx.com"
        res = requests.get(url, params={"instId": instId, "bar": BAR, "limit": LIMIT}, proxies=PROXIES, timeout=10).json()
        df = pd.DataFrame(res['data'], columns=["ts","o","h","l","c","v","vc","cv","confirm"])
        df["c"] = df["c"].astype(float)
        df["ts"] = df["ts"].astype(int)
        return df.sort_values("ts").reset_index(drop=True)["c"].values.astype(np.float32)
    except: return None

def run_cycle():
    now_t = datetime.now().strftime('%H:%M:%S')
    print(f"\n[{now_t}] --- Cycle Start ---")
    symbols = get_all_swap_contracts()
    if not symbols: 
        print("❌ 获取合约失败")
        return
    print(f"Scanning {len(symbols)} symbols...")
    
    # 预测前 50 个
    signals = []
    for s in symbols[:TOP_N]:
        ts = fetch_data(s)
        if ts is None or len(ts) < 50: continue
        curr_p = float(ts[-1])
        with torch.no_grad():
            point, _ = model.forecast(horizon=HORIZON, inputs=[ts])
            f_vals = np.array(point, dtype=np.float32).flatten()
        
        ret = (f_vals[-1] - curr_p) / curr_p
        if abs(ret) < MIN_EXPECTED_RETURN: continue
        
        tp_pct = max(np.floor(abs(ret)*100), 0.5)
        sl_pct = max(round(tp_pct*0.5, 2), 0.3)
        side = "LONG" if ret > 0 else "SHORT"
        tp_p = curr_p * (1 + tp_pct/100) if side == "LONG" else curr_p * (1 - tp_pct/100)
        sl_p = curr_p * (1 - sl_pct/100) if side == "LONG" else curr_p * (1 + sl_pct/100)
        
        res = {"symbol": s, "side": side, "ret": ret*100, "tp_p": tp_p, "sl_p": sl_p, "tp_pct": tp_pct, "sl_pct": sl_pct, "score": abs(ret)}
        signals.append(res)
        print(f"  [+] {s} Score: {abs(ret):.4f}")

    if signals:
        top_picks = pd.DataFrame(signals).sort_values("score", ascending=False).head(FINAL_PICK_N)
        report = ""
        for _, row in top_picks.iterrows():
            line = f"#{row['symbol']} | {row['side']} | 预:{row['ret']:+.2f}% | 盈:{row['tp_p']:.4f}({row['tp_pct']}%) | 损:{row['sl_p']:.4f}({row['sl_pct']}%)"
            print(line)
            report += line + "\n"
        push_wecom(report)

if __name__ == "__main__":
    while True:
        try: run_cycle()
        except Exception as e: print(f"Error: {e}")
        time.sleep(900)
