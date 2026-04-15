import numpy as np
import pandas as pd
import requests
import torch
import warnings
import time
import json
import subprocess
from datetime import datetime

warnings.filterwarnings('ignore')

# 检查 TimesFM 环境
try:
    import timesfm
except Exception as e:
    raise RuntimeError("未安装 timesfm 环境。")

# ==================== 【1. 企业微信推送：原生系统调用版】 ====================
def push_wecom_native(content):
    """
    使用系统级 curl 直接推送，绕过 Python requests 的域名解析 Bug
    """
    corpid = "wwc4b0ff26e668aeb0"
    secret = "lB--gK5xsoZBLYgICXMOgVaju9dzunEkFrF_YBDOUqI"
    agentid = 1000002
    proxy = "http://127.0.0.1:10809"
    
    # 1. 获取 Token (使用 IP 直连 101.91.57.58 绕过域名劫持)
    token_url = f"https://101.91.57{corpid}&corpsecret={secret}"
    try:
        tk_cmd = ['curl', '-s', '-k', '-x', proxy, '-H', 'Host: ://qq.com', token_url]
        tk_res = subprocess.check_output(tk_cmd).decode('utf-8')
        tk = json.loads(tk_res).get("access_token")
        if not tk: return

        # 2. 发送消息
        sd_url = f"https://101.91.57{tk}"
        msg_json = json.dumps({
            "touser": "@all", "msgtype": "text", "agentid": agentid, 
            "text": {"content": f"🚀 TimesFM 信号推送\n----------------\n{content}"}
        }, ensure_ascii=False)
        
        sd_cmd = ['curl', '-s', '-k', '-x', proxy, '-H', 'Host: ://qq.com', 
                  '-H', 'Content-Type: application/json', '-d', msg_json, sd_url]
        subprocess.check_output(sd_cmd)
        print("✅ 信号已成功推送到企业微信")
    except:
        print("⚠️ 微信推送尝试失败，但不影响预测结果落盘")

# ==================== 【2. 策略核心参数】 ====================
BAR = "5m"
LIMIT = 512                 # 回溯 512 根 K 线
HORIZON = 12                # 预测未来 1 小时
TOP_N = 50                  # 候选池
FINAL_PICK_N = 2            # 最终输出前2
MIN_EXPECTED_RETURN = 0.0008 # 0.08% 门槛
MIN_R_SQUARED = 0.55
MIN_DIRECTION_CONFIDENCE = 0.58
OUTPUT_FILE = "signals_vps.json"

# ==================== 【3. 模型加载】 ====================
print("🚀 正在加载 TimesFM 预测引擎...")
torch.set_float32_matmul_precision("high")

def _build_timesfm_model():
    try:
        # 优先尝试新接口
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        return m
    except:
        # 方案B：经典接口
        return timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu" if torch.cuda.is_available() else "cpu", per_core_batch_size=32, horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.5-200m-pytorch")
        )

model = _build_timesfm_model()
print("✅ 模型引擎准备就绪")

# ==================== 【4. 工具函数】 ====================
def get_all_swap_contracts():
    """获取 OKX 所有 USDT 结算合约"""
    url = "https://okx.com"
    try:
        res = requests.get(url, params={"instType": "SWAP"}, timeout=10).json()
        return [i["instId"] for i in res["data"] if i["settleCcy"] == "USDT" and i["state"] == "live"]
    except: return []

def fetch_klines(instId, limit):
    """获取 K 线并执行原始排序逻辑"""
    try:
        url = "https://okx.com"
        res = requests.get(url, params={"instId": instId, "bar": BAR, "limit": str(limit)}, timeout=5).json()
        df = pd.DataFrame(res["data"], columns=["ts", "o", "h", "l", "c", "v", "vc", "cv", "confirm"])
        df["c"] = df["c"].astype(float)
        df["ts"] = df["ts"].astype(int)
        df = df.sort_values("ts").reset_index(drop=True)
        return df["c"].values.astype(np.float32)
    except: return None

def calculate_volatility(prices):
    if len(prices) < 30: return 0
    return np.std(np.diff(prices) / prices[:-1]) * 1000

# ==================== 【5. 核心预测评分 (保留方案A/B投票逻辑)】 ====================
def predict_and_score(instId):
    try:
        ts = fetch_klines(instId, LIMIT)
        if ts is None or len(ts) < 50: return None
        curr_p = float(ts[-1])

        with torch.no_grad():
            try:
                # 尝试新接口返回
                point, _ = model.forecast(horizon=HORIZON, inputs=[ts])
                f_vals = np.array(point, dtype=np.float32).flatten()
            except:
                # 尝试经典接口返回
                out = model.forecast(inputs=[ts], freq=[0])
                f_vals = np.array(out[0] if isinstance(out, tuple) else out, dtype=np.float32).flatten()

        exp_ret = (f_vals[-1] - curr_p) / curr_p
        
        # 路径分析与一致性
        steps = np.arange(len(f_vals))
        z = np.polyfit(steps, f_vals, 1)
        p = np.poly1d(z)
        r2 = 1 - (np.sum((f_vals - p(steps))**2) / np.sum((f_vals - np.mean(f_vals))**2))
        
        direction = 1 if exp_ret > 0 else -1
        diffs = np.diff(np.insert(f_vals, 0, curr_p))
        consist = np.sum(np.sign(diffs) == direction) / len(diffs)
        conf = 0.7 * consist + 0.3 * max(0.0, min(1.0, r2))

        if abs(exp_ret) < MIN_EXPECTED_RETURN or r2 < MIN_R_SQUARED or conf < MIN_DIRECTION_CONFIDENCE:
            return None

        # --- 止盈止损价格换算 ---
        tp_pct = max(np.floor(abs(exp_ret)*100), 0.5)
        sl_pct = max(round(tp_pct*0.5, 2), 0.3)
        tp_p = curr_p * (1 + tp_pct/100) if exp_ret > 0 else curr_p * (1 - tp_pct/100)
        sl_p = curr_p * (1 - sl_pct/100) if exp_ret > 0 else curr_p * (1 + sl_pct/100)

        score = abs(exp_ret) * 100 * 0.4 + r2 * 0.3 + consist * 0.3
        return {
            "symbol": instId, "side": "LONG" if exp_ret > 0 else "SHORT",
            "ret": exp_ret * 100, "score": score,
            "tp_p": tp_p, "sl_p": sl_p, "tp_pct": tp_pct, "sl_pct": sl_pct
        }
    except: return None

# ==================== 【6. 主入口逻辑】 ====================
if __name__ == "__main__":
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 周期任务启动...")
            symbols = get_all_swap_contracts()
            print(f"✅ OKX 数据连通，获取合约：{len(symbols)} 个")

            vols = []
            for s in symbols:
                p = fetch_klines(s, 80)
                if p is not None:
                    vols.append({"symbol": s, "vol": calculate_volatility(p)})

            df_vol = pd.DataFrame(vols).sort_values("vol", ascending=False).head(TOP_N)
            candidates = df_vol["symbol"].tolist()
            print(f"🏆 波动率初筛完成，候选池：{len(candidates)} 个")

            valid_candidates = []
            for s in candidates:
                res = predict_and_score(s)
                if res:
                    valid_candidates.append(res)
                    print(f"  [+] 信号发现: {res['symbol']} 得分: {res['score']:.4f}")

            if valid_candidates:
                top_picks = pd.DataFrame(valid_candidates).sort_values("score", ascending=False).head(FINAL_PICK_N)
                
                print("\n" + "="*20 + " 本轮 Top 2 信号汇总 " + "="*20)
                push_msg = ""
                output_dict = {}
                for i, row in top_picks.iterrows():
                    line = (f"#{i+1} {row['symbol']} | {row['side']} | "
                            f"预期:{row['ret']:+.2f}% | 止盈:{row['tp_p']:.4f} | 止损:{row['sl_p']:.4f}")
                    print(line)
                    push_msg += line + "\n"
                    output_dict[row['symbol']] = row['side'].lower()

                # 保存到信号文件
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(output_dict, f, indent=2, ensure_ascii=False)
                
                # 原生系统调用推送
                push_wecom_native(push_msg)
            else:
                print("❌ 本轮扫描结束，未发现符合阈值的信号")

        except Exception as e:
            print(f"运行异常: {e}")
        
        print(f"\n等待 15 分钟后进行下一轮预测...")
        time.sleep(900)
