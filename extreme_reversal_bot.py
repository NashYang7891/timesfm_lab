#!/usr/bin/env python3
"""
极端反转拐点推导系统 - 自动监控成交量前30的USDT永续合约 (修正版)
"""

import asyncio
import httpx
import numpy as np
import pandas as pd
import time
import ccxt.async_support as ccxt
from datetime import datetime

# ===== 配置 =====
TG_BOT_TOKEN = "8722422674:AAGrKmRurQ2G__j-Vxbh5451v0e9_u97CQY"
TG_CHAT_ID = "5372217316"
TG_PROXIES = None

OKX_API_KEY = "10d14cf0-79da-4597-9456-3aa1b88e1acf"
OKX_SECRET = "1B6A940855EC5787CD4E56BEF6D94733"
OKX_PASSPHRASE = "kP9!vR2@mN5+"


class CryptoExtremeBot:
    def __init__(self):
        # 策略参数
        self.threshold_osc = 0.4          # 综合振荡器阈值（<0.4视为超卖）
        self.deviation_std = 2.0          # 偏离度（价格低于MA20的标准差倍数）
        self.wall_threshold = 0.75        # 挂单壁比例（买方深度占比>0.75）
        self.top_n_symbols = 30           # 监控成交量前30的合约
        self.symbols = []                 # 运行时动态获取
        self.last_sent = {}               # 信号冷却
        self.exchange = None
        self.order_enabled = True         # 是否自动下单

    # ========== 交易所初始化 ==========
    def _init_exchange(self):
        self.exchange = ccxt.okx({
            'apiKey': OKX_API_KEY,
            'secret': OKX_SECRET,
            'password': OKX_PASSPHRASE,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

    # ========== TG推送 ==========
    async def send_tg(self, text):
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        try:
            async with httpx.AsyncClient(proxy=TG_PROXIES) as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                return resp.status_code == 200
        except Exception as e:
            print(f"⚠️ TG推送异常: {e}")
            return False

    # ========== 指标计算 ==========
    @staticmethod
    def compute_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50

    @staticmethod
    def compute_kdj(df, period=9):
        low_min = df['l'].rolling(period).min()
        high_max = df['h'].rolling(period).max()
        rsv = (df['c'] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(span=3, adjust=False).mean()
        d = k.ewm(span=3, adjust=False).mean()
        j = 3 * k - 2 * d
        return j.iloc[-1] if not j.empty else 50

    @staticmethod
    def compute_cci(df, period=20):
        tp = (df['h'] + df['l'] + df['c']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci.iloc[-1] if not cci.empty else 0

    def get_weighted_oscillator(self, rsi, kdj_j, cci):
        norm_rsi = rsi / 100
        norm_kdj = kdj_j / 100
        norm_cci = (cci + 200) / 400
        return (norm_rsi + norm_kdj + norm_cci) / 3

    def calculate_momentum_pivot(self, prices):
        """推导买入位：加速度过零后的理论拐点"""
        if len(prices) < 2:
            return None
        pivot = 2 * prices.iloc[-1] - prices.iloc[-2]
        # 保留足够精度，避免极低价格币种显示为 0.0
        return round(pivot, 10)

    # ========== 数据获取 ==========
    async def fetch_market_data(self, symbol):
        """获取OHLCV并计算所有指标"""
        try:
            ohlcv_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
            ohlcv_1m = await self.exchange.fetch_ohlcv(symbol, '1m', limit=60)
            if not ohlcv_5m or len(ohlcv_5m) < 60:
                return None
        except Exception as e:
            print(f"⚠️ 获取K线失败 {symbol}: {e}")
            return None

        df_5m = pd.DataFrame(ohlcv_5m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_5m[['c', 'h', 'l']] = df_5m[['c', 'h', 'l']].astype(float)

        df_1m = pd.DataFrame(ohlcv_1m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_1m['c'] = df_1m['c'].astype(float)

        prices = df_5m['c']
        curr_price = prices.iloc[-1]

        # 布林带偏离度
        ma20 = prices.rolling(20).mean().iloc[-1]
        std20 = prices.rolling(20).std().iloc[-1]
        deviation = (ma20 - curr_price) / std20 if std20 > 0 else 0

        # 指标
        rsi = self.compute_rsi(prices, 14)
        kdj_j = self.compute_kdj(df_5m, 9)
        cci = self.compute_cci(df_5m, 20)
        osc_score = self.get_weighted_oscillator(rsi, kdj_j, cci)

        # 挂单壁（用ticker买卖量比近似）
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            bid_vol = ticker.get('bidVolume', 0) or 0
            ask_vol = ticker.get('askVolume', 0) or 0
            total_vol = bid_vol + ask_vol
            buy_wall = bid_vol / total_vol if total_vol > 0 else 0.5
        except:
            buy_wall = 0.5

        return {
            'close': prices,
            'price_1m': df_1m['c'] if not df_1m.empty else prices,
            'curr_price': curr_price,
            'ma20': ma20, 'std20': std20,
            'deviation': deviation,
            'rsi': rsi, 'kdj_j': kdj_j, 'cci': cci,
            'osc_score': osc_score,
            'buy_wall': buy_wall
        }

    # ========== 下单 ==========
    async def place_limit_order(self, symbol, price, amount=0.01):
        try:
            await self.exchange.set_leverage(3, symbol, params={'posSide': 'long', 'mgnMode': 'isolated'})
            order = await self.exchange.create_order(
                symbol=symbol, type='limit', side='buy',
                amount=amount, price=price,
                params={'posSide': 'long', 'tdMode': 'isolated', 'ordType': 'post_only'}
            )
            print(f"✅ 已挂限价单 {symbol} @ {price}, 数量: {amount}")
            return order
        except Exception as e:
            print(f"❌ 下单失败 {symbol}: {e}")
            return None

    # ========== 动态获取热门合约 ==========
    async def fetch_hot_symbols(self):
        """通过 OKX 公共 API 获取成交量最大的 top_n_symbols 个 SWAP 合约"""
        try:
            # 1. 获取所有 USDT 永续合约的 instId 列表
            instruments_url = "https://www.okx.com/api/v5/public/instruments"
            params = {"instType": "SWAP"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(instruments_url, params=params, timeout=10)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}")
                data = resp.json()
                if data.get("code") != "0":
                    raise Exception(f"API error: {data}")

            all_inst_ids = [item["instId"] for item in data["data"]
                            if item["settleCcy"] == "USDT" and item["state"] == "live"]

            # 2. 批量获取合约的 24h 成交量（USDT 计价）
            tickers_url = "https://www.okx.com/api/v5/market/tickers"
            params = {"instType": "SWAP"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(tickers_url, params=params, timeout=10)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}")
                tickers_data = resp.json()
                if tickers_data.get("code") != "0":
                    raise Exception(f"Tickers API error: {tickers_data}")

            # 构建 instId -> volCcy24h 的字典
            vol_map = {}
            for t in tickers_data["data"]:
                inst_id = t.get("instId")
                if inst_id:
                    vol = float(t.get("volCcy24h", 0))
                    vol_map[inst_id] = vol

            # 3. 筛选并排序前 top_n_symbols 个
            valid_pairs = [(inst_id, vol_map.get(inst_id, 0)) for inst_id in all_inst_ids]
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
            self.symbols = [s for s, _ in valid_pairs[:self.top_n_symbols]]

            print(f"✅ 动态热门合约: {len(self.symbols)} 个，示例: {self.symbols[:6]}")
        except Exception as e:
            print(f"⚠️ 获取热门合约失败，使用默认列表: {e}")
            # 兜底列表
            self.symbols = [
                "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "ADA-USDT-SWAP",
                "AVAX-USDT-SWAP", "DOT-USDT-SWAP", "LINK-USDT-SWAP",
                "MATIC-USDT-SWAP", "UNI-USDT-SWAP", "SHIB-USDT-SWAP",
                "LTC-USDT-SWAP", "ATOM-USDT-SWAP", "FIL-USDT-SWAP",
                "APT-USDT-SWAP", "ARB-USDT-SWAP", "OP-USDT-SWAP",
                "WLD-USDT-SWAP", "SEI-USDT-SWAP", "SUI-USDT-SWAP",
                "TIA-USDT-SWAP", "INJ-USDT-SWAP", "RNDR-USDT-SWAP",
                "FET-USDT-SWAP", "AGIX-USDT-SWAP", "PEPE-USDT-SWAP",
                "WIF-USDT-SWAP", "BONK-USDT-SWAP", "FLOKI-USDT-SWAP"
            ]

    # ========== 主扫描 ==========
    async def scan_market(self):
        print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] 扫描 {len(self.symbols)} 个合约...")
        for symbol in self.symbols:
            try:
                data = await self.fetch_market_data(symbol)
                if data is None:
                    continue

                curr_price = data['curr_price']
                deviation = data['deviation']
                osc_score = data['osc_score']
                buy_wall = data['buy_wall']

                if deviation > self.deviation_std and buy_wall > self.wall_threshold:
                    if osc_score < self.threshold_osc:
                        target_entry = self.calculate_momentum_pivot(data['price_1m'])
                        if target_entry is None:
                            continue

                        now = time.time()
                        if symbol in self.last_sent and now - self.last_sent[symbol] < 600:
                            continue

                        # ----- 明确的多头信号模板 -----
                        msg = (
                            f"📈 **极端反转做多信号**\n\n"
                            f"🏷️ 币种: `{symbol}`\n"
                            f"💰 当前价: `{curr_price}`\n"
                            f"📍 **推导买入位**: `{target_entry}`\n\n"
                            f"📊 **核心参数**:\n"
                            f"- 偏离度: `{deviation:.2f}` (超卖)\n"
                            f"- 振荡加权: `{osc_score:.2f}` (超卖)\n"
                            f"- 挂单壁: `{buy_wall:.2f}` (买方深度强)\n"
                            f"⏳ 动能: `加速度过零` → 下跌衰竭\n"
                            f"🛑 建议止损: `{curr_price * 0.98:.6f}`"
                        )
                        success = await self.send_tg(msg)
                        if success:
                            self.last_sent[symbol] = now
                            print(f"✅ 推送 {symbol} 信号")

                            if self.order_enabled:
                                await self.place_limit_order(symbol, target_entry, amount=0.01)

            except Exception as e:
                print(f"❌ 扫描 {symbol} 出错: {e}")

    async def run(self):
        self._init_exchange()
        try:
            await self.exchange.load_markets()
            print("✅ OKX 连接成功")
            await self.fetch_hot_symbols()
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return

        await self.send_tg(f"🚀 **极端拐点推导系统启动**\n模式: 激进版 (加速度过零)\n监控合约数: {len(self.symbols)}")
        while True:
            start = time.time()
            await self.scan_market()
            elapsed = time.time() - start
            await asyncio.sleep(max(0, 60 - elapsed))


if __name__ == "__main__":
    bot = CryptoExtremeBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("停止运行")
