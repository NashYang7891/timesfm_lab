#!/usr/bin/env python3
"""
极端反转拐点推导系统 - 多空对称版
支持 做多（下跌衰竭） 与 做空（上涨衰竭），参数可独立调节
"""

import asyncio, httpx, numpy as np, pandas as pd, time, ccxt.async_support as ccxt
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
        # ---------- 多空独立参数 ----------
        # 做多条件（下跌衰竭）
        self.long_deviation = 2.5          # 价格低于MA20的σ倍数
        self.long_osc = 0.28              # 综合振荡器上限（超卖）
        self.long_wall = 0.75             # 买方深度占比下限
        self.long_15m_rsi_max = 40        # 15分钟RSI必须低于此值
        # 做空条件（上涨衰竭）
        self.short_deviation = 2.5         # 价格高于MA20的σ倍数
        self.short_osc = 0.72             # 综合振荡器下限（超买，1-0.28）
        self.short_wall = 0.25            # 卖方深度占比下限（即买方占比<0.25）
        self.short_15m_rsi_min = 60       # 15分钟RSI必须高于此值
        # 共同条件
        self.volume_spike_ratio = 1.5      # 成交量倍率
        self.accel_threshold = 0.0         # 加速度过零即发（取消硬缓冲）
        # 流动性门槛
        self.MIN_VOLUME_USDT = 20_000_000
        self.top_n_symbols = 30
        self.symbols = []
        self.last_sent = {}
        self.exchange = None
        self.order_enabled = True

    def _init_exchange(self):
        self.exchange = ccxt.okx({
            'apiKey': OKX_API_KEY, 'secret': OKX_SECRET, 'password': OKX_PASSPHRASE,
            'enableRateLimit': True, 'options': {'defaultType': 'swap'}
        })

    async def send_tg(self, text):
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        try:
            async with httpx.AsyncClient(proxy=TG_PROXIES) as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                return resp.status_code == 200
        except Exception as e:
            print(f"TG推送异常: {e}")
            return False

    # 时段权重、静默期、瀑布检测（同前，略加调整以支持做空）
    def get_time_multiplier(self):
        now = datetime.now(timezone.utc).hour
        if 20 <= now <= 23: return 1.3
        if 13 <= now <= 16: return 1.2
        return 1.0

    def is_in_quiet_period(self):
        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute < 5: return True
        for h in [0,8,16]:
            start = now.replace(hour=h, minute=0, second=0) - timedelta(minutes=5)
            end = now.replace(hour=h, minute=0, second=0) + timedelta(minutes=5)
            if start <= now <= end: return True
        return False

    async def is_btc_waterfall(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC-USDT-SWAP', '1h', limit=1)
            if ohlcv and len(ohlcv) > 0:
                chg = (ohlcv[0][4] - ohlcv[0][1]) / ohlcv[0][1]
                if chg < -0.05: return True   # 暴跌暂停（做多危险，做空同样可能极端）
        except: pass
        return False

    # ---------- 指标计算（同前）----------
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

    def calculate_momentum_pivot(self, prices, direction='long'):
        if len(prices) < 2: return None
        pivot = 2 * prices.iloc[-1] - prices.iloc[-2]
        return round(pivot, 10)

    # ---------- 数据获取 ----------
    async def fetch_market_data(self, symbol):
        try:
            ohlcv_1m, ohlcv_5m, ohlcv_15m = await asyncio.gather(
                self.exchange.fetch_ohlcv(symbol, '1m', limit=100),
                self.exchange.fetch_ohlcv(symbol, '5m', limit=100),
                self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            )
        except Exception as e:
            return None
        if not ohlcv_5m or len(ohlcv_5m) < 60 or not ohlcv_1m or len(ohlcv_1m) < 30:
            return None

        def to_df(data, cols=['ts','o','h','l','c','v']):
            df = pd.DataFrame(data, columns=cols)
            for col in ['c','h','l','v']: df[col] = df[col].astype(float)
            return df
        df_5m = to_df(ohlcv_5m)
        df_1m = to_df(ohlcv_1m)
        df_15m = to_df(ohlcv_15m) if ohlcv_15m and len(ohlcv_15m) >= 50 else None

        prices_5m = df_5m['c']
        curr_price = prices_5m.iloc[-1]
        ma20 = prices_5m.rolling(20).mean().iloc[-1]
        std20 = prices_5m.rolling(20).std().iloc[-1]
        deviation = (ma20 - curr_price) / std20 if std20 > 0 else 0   # >0 超跌, <0 超涨

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
        except: buy_wall = 0.5

        rsi_15m = None
        if df_15m is not None and len(df_15m) >= 30:
            rsi_15m = self.compute_rsi(df_15m['c'], 14)

        return {
            'curr_price': curr_price, 'deviation': deviation, 'osc_score': osc_score,
            'buy_wall': buy_wall, 'price_1m': df_1m['c'], 'volume_1m': df_1m['v'],
            'rsi_15m': rsi_15m
        }

    # ---------- 下单（支持做空）----------
    async def place_limit_order(self, symbol, side, price, amount=0.01):
        try:
            pos_side = 'long' if side == 'buy' else 'short'
            await self.exchange.set_leverage(3, symbol, params={'posSide': pos_side, 'mgnMode': 'isolated'})
            order = await self.exchange.create_order(
                symbol=symbol, type='limit', side=side,
                amount=amount, price=price,
                params={'posSide': pos_side, 'tdMode': 'isolated', 'ordType': 'post_only'}
            )
            print(f"✅ 已挂{'多' if side=='buy' else '空'}单 {symbol} @ {price}")
            return order
        except Exception as e:
            print(f"❌ 下单失败 {symbol}: {e}")

    # ---------- 动态合约筛选 ----------
    async def fetch_hot_symbols(self):
        try:
            instruments_url = "https://www.okx.com/api/v5/public/instruments"
            tickers_url = "https://www.okx.com/api/v5/market/tickers"
            params = {"instType": "SWAP"}
            async with httpx.AsyncClient() as client:
                resp = await client.get(instruments_url, params=params, timeout=10)
                if resp.status_code != 200: raise Exception("instruments fail")
                data = resp.json()
                all_ids = [i["instId"] for i in data["data"] if i["settleCcy"]=="USDT" and i["state"]=="live"]
                resp = await client.get(tickers_url, params=params, timeout=10)
                if resp.status_code != 200: raise Exception("tickers fail")
                tickers_data = resp.json()
            vol_map = {t["instId"]: float(t.get("volCcy24h",0)) for t in tickers_data["data"] if "instId" in t}
            valid = [(sid, vol_map.get(sid,0)) for sid in all_ids if vol_map.get(sid,0) >= self.MIN_VOLUME_USDT]
            valid.sort(key=lambda x: x[1], reverse=True)
            self.symbols = [s for s,_ in valid[:self.top_n_symbols]]
            print(f"✅ 动态热门合约: {len(self.symbols)} 个 (≥{self.MIN_VOLUME_USDT/1e6}M)")
        except Exception as e:
            print(f"⚠️ 获取失败: {e}")
            self.symbols = ["BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","XRP-USDT-SWAP","DOGE-USDT-SWAP"]

    # ---------- 主扫描（多空对称）----------
    async def scan_market(self):
        if self.is_in_quiet_period():
            print(f"[{datetime.now():%H:%M:%S}] 静默期")
            return
        if await self.is_btc_waterfall():
            return

        time_mult = self.get_time_multiplier()
        print(f"[{datetime.now():%H:%M:%S}] 扫描 {len(self.symbols)} 个合约 (时段系数:{time_mult})")

        for symbol in self.symbols:
            try:
                data = await self.fetch_market_data(symbol)
                if data is None: continue

                dev = data['deviation']
                osc = data['osc_score']
                bw = data['buy_wall']
                rsi15 = data['rsi_15m']
                vol = data['volume_1m']
                prices = data['price_1m']

                # 成交量条件
                if len(vol) < 21: continue
                cur_vol = vol.iloc[-1]
                avg_vol = vol.iloc[-21:-1].mean()
                if cur_vol < self.volume_spike_ratio * avg_vol: continue

                # 加速度过零
                if len(prices) < 5: continue
                diffs = np.diff(prices.values)
                accel = np.diff(diffs)
                if len(accel) < 2: continue
                curr_acc = accel[-1]
                prev_acc = accel[-2]

                # 判断场景
                signal = None
                direction = None
                # 做多场景
                long_dev_eff = self.long_deviation * time_mult
                if (dev > long_dev_eff and osc < self.long_osc and bw > self.long_wall and
                    rsi15 is not None and rsi15 < self.long_15m_rsi_max and
                    curr_acc > 0 and prev_acc <= 0):
                    signal = 'long'
                    direction = 'buy'
                # 做空场景
                short_dev_eff = self.short_deviation * time_mult
                if (dev < -short_dev_eff and osc > self.short_osc and bw < self.short_wall and
                    rsi15 is not None and rsi15 > self.short_15m_rsi_min and
                    curr_acc < 0 and prev_acc >= 0):
                    signal = 'short'
                    direction = 'sell'

                if signal is None: continue

                target = self.calculate_momentum_pivot(prices, signal)
                if target is None: continue

                now = time.time()
                if symbol in self.last_sent and now - self.last_sent[symbol] < 3600: continue

                # 构建消息
                emoji = "📈" if signal == 'long' else "📉"
                side_cn = "做多" if signal == 'long' else "做空"
                msg = (
                    f"{emoji} **极端反转{side_cn}信号**\n\n"
                    f"🏷️ 币种: `{symbol}`\n"
                    f"💰 当前价: `{data['curr_price']}`\n"
                    f"📍 **{'买入' if signal=='long' else '卖出'}位**: `{target}`\n\n"
                    f"📊 关键参数:\n"
                    f"- 偏离度: `{dev:.2f}`\n"
                    f"- 振荡器: `{osc:.2f}`\n"
                    f"- 挂单壁: `{bw:.2f}`\n"
                    f"- 15mRSI: `{rsi15:.1f}`\n"
                    f"- 成交量: `{cur_vol/avg_vol:.1f}x`\n"
                    f"- 加速度: `{curr_acc:.6f}`\n"
                    f"🛑 建议止损: `{data['curr_price'] * (0.98 if signal=='long' else 1.02):.6f}`"
                )
                if await self.send_tg(msg):
                    self.last_sent[symbol] = now
                    print(f"✅ 推送 {symbol} {side_cn}")
                    if self.order_enabled:
                        await self.place_limit_order(symbol, direction, target, 0.01)

            except Exception as e:
                print(f"❌ {symbol}: {e}")

    async def run(self):
        self._init_exchange()
        await self.exchange.load_markets()
        await self.fetch_hot_symbols()
        await self.send_tg(f"🚀 多空对称系统启动 | 做多偏离>{self.long_deviation}σ 做空偏离<-{self.short_deviation}σ | 成交量>{self.volume_spike_ratio}x")
        while True:
            start = time.time()
            await self.scan_market()
            await asyncio.sleep(max(0, 60 - (time.time() - start)))

if __name__ == "__main__":
    bot = CryptoExtremeBot()
    try: asyncio.run(bot.run())
    except KeyboardInterrupt: print("停止")
