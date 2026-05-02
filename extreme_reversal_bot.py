#!/usr/bin/env python3
"""
极端反转拐点推导系统 - 黄金平衡版
参数优化 + 分层过滤 + 时段风控 + 成交量门槛 20M
"""

import asyncio
import httpx
import numpy as np
import pandas as pd
import time
import ccxt.async_support as ccxt
from datetime import datetime, timezone, timedelta

# ===== 配置 =====
TG_BOT_TOKEN = "8722422674:AAGrKmRurQ2G__j-Vxbh5451v0e9_u97CQY"
TG_CHAT_ID = "5372217316"
TG_PROXIES = None

OKX_API_KEY = "10d14cf0-79da-4597-9456-3aa1b88e1acf"
OKX_SECRET = "1B6A940855EC5787CD4E56BEF6D94733"
OKX_PASSPHRASE = "kP9!vR2@mN5+"


class CryptoExtremeBot:
    def __init__(self):
        # ---------- 黄金平衡版参数 ----------
        self.deviation_std = 2.5           # 偏离度：从2.8降至2.5，兼顾极端与频率
        self.threshold_osc = 0.28          # 综合振荡器：从0.2放宽至0.28，避免过于僵化
        self.wall_threshold = 0.75         # 挂单壁比例保持不变
        self.volume_spike_ratio = 1.5      # 成交量喷发：从2.0x降至1.5x，宽松但仍有主力介入痕迹
        self.accel_threshold = 0.0         # 加速度：取消固定缓冲，改为简单过零即发
        self.min_15m_rsi = 40              # 15分钟RSI上限：从35上调至40，避免错过早期反弹

        # ---------- 流动性门槛 ----------
        self.MIN_VOLUME_USDT = 20_000_000  # 24h成交额 ≥ 2000万 USDT

        # ---------- 系统控制 ----------
        self.top_n_symbols = 30
        self.symbols = []
        self.last_sent = {}
        self.exchange = None
        self.order_enabled = True          # 是否自动下单

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

    # ========== 时段权重与禁区判断 ==========
    def get_time_multiplier(self):
        now_utc = datetime.now(timezone.utc).hour
        if 20 <= now_utc <= 23:           # 低流动性时段
            return 1.3
        if 13 <= now_utc <= 16:           # 美盘剧烈波动时段
            return 1.2
        return 1.0

    def is_in_quiet_period(self):
        now = datetime.now(timezone.utc)
        # 日线换线 (00:00-00:05)
        if now.hour == 0 and now.minute < 5:
            return True
        # 费率结算前后5分钟
        for funding_hour in [0, 8, 16]:
            start = now.replace(hour=funding_hour, minute=0, second=0, microsecond=0) - timedelta(minutes=5)
            end = now.replace(hour=funding_hour, minute=0, second=0, microsecond=0) + timedelta(minutes=5)
            if start <= now <= end:
                return True
        return False

    async def is_btc_waterfall(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC-USDT-SWAP', '1h', limit=1)
            if ohlcv and len(ohlcv) > 0:
                change = (ohlcv[0][4] - ohlcv[0][1]) / ohlcv[0][1]
                if change < -0.05:  # 1小时跌幅超5%
                    print(f"⚠️ BTC 1小时跌幅 {change*100:.2f}%，全局暂停")
                    return True
        except Exception as e:
            print(f"⚠️ 获取BTC瀑布信息失败: {e}")
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
        if len(prices) < 2:
            return None
        pivot = 2 * prices.iloc[-1] - prices.iloc[-2]
        return round(pivot, 10)

    # ========== 数据获取 ==========
    async def fetch_market_data(self, symbol):
        try:
            ohlcv_1m, ohlcv_5m, ohlcv_15m = await asyncio.gather(
                self.exchange.fetch_ohlcv(symbol, '1m', limit=100),
                self.exchange.fetch_ohlcv(symbol, '5m', limit=100),
                self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            )
        except Exception as e:
            print(f"⚠️ 获取K线失败 {symbol}: {e}")
            return None

        if not ohlcv_5m or len(ohlcv_5m) < 60 or not ohlcv_1m or len(ohlcv_1m) < 30:
            return None

        def to_df(data, cols=['ts','o','h','l','c','v']):
            df = pd.DataFrame(data, columns=cols)
            for col in ['c','h','l','v']:
                df[col] = df[col].astype(float)
            return df

        df_5m = to_df(ohlcv_5m)
        df_1m = to_df(ohlcv_1m)
        df_15m = to_df(ohlcv_15m) if ohlcv_15m and len(ohlcv_15m) >= 50 else None

        prices_5m = df_5m['c']
        curr_price = prices_5m.iloc[-1]

        ma20 = prices_5m.rolling(20).mean().iloc[-1]
        std20 = prices_5m.rolling(20).std().iloc[-1]
        deviation = (ma20 - curr_price) / std20 if std20 > 0 else 0

        rsi_5m = self.compute_rsi(prices_5m, 14)
        kdj_j = self.compute_kdj(df_5m, 9)
        cci = self.compute_cci(df_5m, 20)
        osc_score = self.get_weighted_oscillator(rsi_5m, kdj_j, cci)

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            bid_vol = ticker.get('bidVolume', 0) or 0
            ask_vol = ticker.get('askVolume', 0) or 0
            total_vol = bid_vol + ask_vol
            buy_wall = bid_vol / total_vol if total_vol > 0 else 0.5
        except:
            buy_wall = 0.5

        rsi_15m = None
        if df_15m is not None and len(df_15m) >= 30:
            rsi_15m = self.compute_rsi(df_15m['c'], 14)

        return {
            'curr_price': curr_price,
            'deviation': deviation,
            'osc_score': osc_score,
            'buy_wall': buy_wall,
            'price_1m': df_1m['c'],
            'volume_1m': df_1m['v'],
            'rsi_15m': rsi_15m
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
        try:
            instruments_url = "https://www.okx.com/api/v5/public/instruments"
            tickers_url = "https://www.okx.com/api/v5/market/tickers"
            params = {"instType": "SWAP"}

            async with httpx.AsyncClient() as client:
                resp = await client.get(instruments_url, params=params, timeout=10)
                if resp.status_code != 200: raise Exception(f"HTTP {resp.status_code}")
                data = resp.json()
                if data.get("code") != "0": raise Exception(f"API error: {data}")
                all_inst_ids = [item["instId"] for item in data["data"]
                                if item["settleCcy"] == "USDT" and item["state"] == "live"]

                resp = await client.get(tickers_url, params=params, timeout=10)
                if resp.status_code != 200: raise Exception(f"HTTP {resp.status_code}")
                tickers_data = resp.json()
                if tickers_data.get("code") != "0": raise Exception(f"Tickers API error: {tickers_data}")

            vol_map = {}
            for t in tickers_data["data"]:
                inst_id = t.get("instId")
                if inst_id:
                    vol_map[inst_id] = float(t.get("volCcy24h", 0))

            valid_pairs = [(inst_id, vol_map.get(inst_id, 0)) for inst_id in all_inst_ids
                           if vol_map.get(inst_id, 0) >= self.MIN_VOLUME_USDT]
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
            self.symbols = [s for s, _ in valid_pairs[:self.top_n_symbols]]
            print(f"✅ 动态热门合约: {len(self.symbols)} 个 (≥ {self.MIN_VOLUME_USDT/1e6:.0f}M USDT)")
        except Exception as e:
            print(f"⚠️ 获取热门合约失败，使用默认列表: {e}")
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
        if self.is_in_quiet_period():
            print(f"⏰ [{datetime.now().strftime('%H:%M:%S')}] 静默期，跳过")
            return
        if await self.is_btc_waterfall():
            return

        time_mult = self.get_time_multiplier()
        effective_dev = self.deviation_std * time_mult

        print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] 扫描 {len(self.symbols)} 个合约 "
              f"(时段系数:{time_mult}, 偏离度门限:{effective_dev:.2f})")

        for symbol in self.symbols:
            try:
                data = await self.fetch_market_data(symbol)
                if data is None:
                    continue

                # 1. 空间过滤 (动态)
                if data['deviation'] < effective_dev:
                    continue

                # 2. 振荡器
                if data['osc_score'] > self.threshold_osc:
                    continue

                # 3. 挂单壁
                if data['buy_wall'] < self.wall_threshold:
                    continue

                # 4. 15分钟趋势
                if data['rsi_15m'] is None or data['rsi_15m'] > self.min_15m_rsi:
                    continue

                # 5. 成交量喷发
                vol_1m = data['volume_1m']
                if len(vol_1m) < 21:
                    continue
                current_vol = vol_1m.iloc[-1]
                avg_vol = vol_1m.iloc[-21:-1].mean()
                if current_vol < self.volume_spike_ratio * avg_vol:
                    continue

                # 6. 加速度过零 (取消固定缓冲)
                prices_1m = data['price_1m']
                if len(prices_1m) < 5:
                    continue
                diffs = np.diff(prices_1m.values)
                accel = np.diff(diffs)
                if len(accel) < 2:
                    continue
                curr_acc = accel[-1]
                prev_acc = accel[-2]
                if not (curr_acc > 0 and prev_acc <= 0):   # 简单的由负转正
                    continue

                # 信号生成
                target_entry = self.calculate_momentum_pivot(prices_1m)
                if target_entry is None:
                    continue

                now = time.time()
                if symbol in self.last_sent and now - self.last_sent[symbol] < 3600:
                    continue

                msg = (
                    f"📈 **极端反转做多信号 (平衡版)**\n\n"
                    f"🏷️ 币种: `{symbol}`\n"
                    f"💰 当前价: `{data['curr_price']}`\n"
                    f"📍 **推导买入位**: `{target_entry}`\n\n"
                    f"📊 **通过条件**:\n"
                    f"- 动态偏离度: `{data['deviation']:.2f}` (≥{effective_dev:.2f})\n"
                    f"- 振荡加权: `{data['osc_score']:.2f}` (≤{self.threshold_osc})\n"
                    f"- 挂单壁: `{data['buy_wall']:.2f}` (≥{self.wall_threshold})\n"
                    f"- 15分钟RSI: `{data['rsi_15m']:.1f}` (<{self.min_15m_rsi})\n"
                    f"- 成交量: `{current_vol/avg_vol:.1f}x` (≥{self.volume_spike_ratio})\n"
                    f"- 加速度过零: `{curr_acc:.6f}` (prev≤0, curr>0)\n\n"
                    f"🛑 建议止损: `{data['curr_price'] * 0.98:.6f}`"
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

        await self.send_tg(f"🚀 **黄金平衡版反转系统启动**\n"
                           f"偏离度>{self.deviation_std}σ(动态) | 振荡器<{self.threshold_osc} | "
                           f"巨量>{self.volume_spike_ratio}x | 加速度>0 | 15mRSI<{self.min_15m_rsi}\n"
                           f"最低24h成交额: {self.MIN_VOLUME_USDT/1e6}M USDT")
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
