import numpy as np
import pandas as pd
import requests
import torch
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# TimesFM 兼容导入（不同版本包名/接口可能不同）
try:
    import timesfm  # type: ignore
except Exception as e:
    raise RuntimeError(
        "未安装 timesfm 或环境不兼容。请先安装/升级 timesfm 后再运行。"
    ) from e

# ==================== 【终极参数】 ====================
# 5m 输入，预测未来 12 根（约 1 小时）
BAR = "5m"
LIMIT = 512
HORIZON = 12
TOP_N = 50  # 候选池
FINAL_PICK_N = 2  # 最终输出前 N 个合约给机器人观察/执行
MIN_EXPECTED_RETURN = 0.0008  # 0.08%
MIN_R_SQUARED = 0.55
# 方向置信度阈值：先用 0.58（信号少则每次下调 0.02，假信号多则上调 0.02）
MIN_DIRECTION_CONFIDENCE = 0.58
OUTPUT_FILE = "signals_vps.json"
REPORT_FILE = "signals_vps_report.json"

# ==================== 模型配置 ====================
print("🚀 加载 TimesFM 模型...")
torch.set_float32_matmul_precision("high")


def _build_timesfm_model():
    """兼容不同 timesfm 版本的模型构造接口。"""
    errors = []

    # 方案A：新接口（你当前脚本使用的接口）
    try:
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        try:
            fc = timesfm.ForecastConfig(max_context=8192, max_horizon=512)
            m.compile(forecast_config=fc)
        except Exception:
            # 某些版本不需要 compile 或参数名不同
            pass
        return m
    except Exception as e:
        errors.append(f"TimesFM_2p5_200M_torch 路径失败: {e}")

    # 方案B：经典接口（较常见）
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
print("✅ 模型加载完成")

# ==================== 工具函数 ====================
def get_all_swap_contracts():
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SWAP"}
    data = requests.get(url, params=params, timeout=10).json()["data"]
    symbols = []
    for item in data:
        if item["settleCcy"] == "USDT" and item["state"] == "live":
            symbols.append(item["instId"])
    return symbols


def fetch_klines(instId, limit):
    try:
        url = "https://www.okx.com/api/v5/market/history-candles"
        params = {"instId": instId, "bar": BAR, "limit": str(limit)}
        data = requests.get(url, params=params, timeout=5).json()["data"]
        df = pd.DataFrame(data, columns=["ts", "o", "h", "l", "c", "v", "vc", "cv", "confirm"])
        df["c"] = df["c"].astype(float)
        df["ts"] = df["ts"].astype(int)
        df = df.sort_values("ts").reset_index(drop=True)
        return df["c"].values.astype(np.float32)
    except Exception:
        return None


def calculate_volatility(prices):
    if len(prices) < 30:
        return 0
    ret = np.diff(prices) / prices[:-1]
    return np.std(ret) * 1000


# ==================== 【核心】专业预测 + 综合评分 ====================
def predict_and_score(instId):
    try:
        ts = fetch_klines(instId, LIMIT)
        if ts is None or len(ts) < 50:
            return None

        current_price = float(ts[-1])

        # 兼容不同 timesfm 版本的 forecast 返回格式
        with torch.no_grad():
            try:
                point, _ = model.forecast(horizon=HORIZON, inputs=[ts])
                forecast_values = np.array(point[0], dtype=np.float32)
            except TypeError:
                # 经典接口通常是 forecast(inputs, freq=None)
                out = model.forecast(inputs=[ts], freq=[0])
                if isinstance(out, tuple):
                    # (forecast, extra)
                    forecast_values = np.array(out[0][0], dtype=np.float32)
                else:
                    forecast_values = np.array(out[0], dtype=np.float32)
            except Exception:
                # 再兜底一次
                out = model.forecast(inputs=[ts])
                if isinstance(out, tuple):
                    forecast_values = np.array(out[0][0], dtype=np.float32)
                else:
                    forecast_values = np.array(out[0], dtype=np.float32)

        # 1. 总预期涨跌幅
        final_forecast = forecast_values[-1]
        expected_return = (final_forecast - current_price) / current_price

        # 2. 路径平滑度 (R-squared)
        steps = np.arange(len(forecast_values))
        z = np.polyfit(steps, forecast_values, 1)
        p = np.poly1d(z)
        y_average = np.mean(forecast_values)
        ss_res = np.sum((forecast_values - p(steps)) ** 2)
        ss_tot = np.sum((forecast_values - y_average) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 3. 趋势一致性
        diffs = np.diff(np.insert(forecast_values, 0, current_price))
        direction = 1 if expected_return > 0 else -1
        consistency = np.sum(np.sign(diffs) == direction) / len(diffs)

        # 4. 方向置信度（0~1）
        direction_confidence = 0.7 * consistency + 0.3 * max(0.0, min(1.0, r_squared))

        # 基础过滤
        if (
            abs(expected_return) < MIN_EXPECTED_RETURN
            or r_squared < MIN_R_SQUARED
            or direction_confidence < MIN_DIRECTION_CONFIDENCE
        ):
            return None

        # 综合得分
        score = (
            abs(expected_return) * 100 * 0.4
            + r_squared * 0.3
            + consistency * 0.3
        )

        signal = "long" if expected_return > 0 else "short"

        expected_pct = abs(expected_return) * 100
        take_profit_pct = max(np.floor(expected_pct), 0.5)
        stop_loss_pct = max(round(take_profit_pct * 0.5, 2), 0.3)

        return {
            "symbol": instId,
            "signal": signal,
            "expected_return": expected_return,
            "r_squared": r_squared,
            "consistency": consistency,
            "direction_confidence": direction_confidence,
            "score": score,
            "take_profit_pct": float(take_profit_pct),
            "stop_loss_pct": float(stop_loss_pct),
        }

    except Exception:
        return None


# ==================== 主流程 ====================
if __name__ == "__main__":
    print(f"\n⏰ 终极策略：输出综合得分最高的前{FINAL_PICK_N}个币")
    print("📊 评分权重：预期涨跌(40%) + 平滑度(30%) + 一致性(30%)\n")

    symbols = get_all_swap_contracts()
    print(f"✅ 合约总数：{len(symbols)}")

    vol_list = []
    for s in symbols:
        p = fetch_klines(s, 80)
        if p is not None:
            vol = calculate_volatility(p)
            vol_list.append({"symbol": s, "vol": vol})

    df_vol = pd.DataFrame(vol_list).sort_values("vol", ascending=False).head(TOP_N)
    candidate_list = df_vol["symbol"].tolist()
    print(f"🏆 候选池：波动率Top{TOP_N}\n")

    print("📈 开始专业评分...")
    print("-" * 80)

    valid_candidates = []
    for s in candidate_list:
        res = predict_and_score(s)
        if res:
            valid_candidates.append(res)
            print(f"  [{len(valid_candidates)}] {res['symbol']}")
            print(
                f"      涨跌: {res['expected_return']*100:+.2f}% | R²: {res['r_squared']:.2f} | "
                f"一致性: {res['consistency']:.2f} | 方向置信度: {res['direction_confidence']:.2f} | 得分: {res['score']:.4f}"
            )

    if len(valid_candidates) == 0:
        print("\n" + "=" * 80)
        print("❌ 没有符合条件的币")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
    else:
        df_results = pd.DataFrame(valid_candidates).sort_values("score", ascending=False).reset_index(drop=True)
        top_picks = df_results.head(FINAL_PICK_N)

        print("\n" + "=" * 80)
        print(f"🏆 最终入选前{len(top_picks)}名：")

        output_dict = {}
        report_rows = []
        for i, row in top_picks.iterrows():
            print(
                f"  #{i+1} {row['symbol']} | {row['signal'].upper()} | "
                f"预期涨跌: {row['expected_return']*100:+.2f}% | "
                f"R²: {row['r_squared']:.2f} | 一致性: {row['consistency']:.2f} | "
                f"方向置信度: {row['directi
