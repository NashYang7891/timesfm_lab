import asyncio
import json
import logging
import websockets

logger = logging.getLogger(__name__)

class OKXWebSocket:
    def __init__(self, on_mark_price_callback):
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.callback = on_mark_price_callback
        self.keep_running = True
        self.subscribed_symbols = set()
        self.ws = None

    async def subscribe_mark_price(self, symbols):
        """订阅一个或多个币种的标记价格（格式：BTC-USDT-SWAP）"""
        new_symbols = [s for s in symbols if s not in self.subscribed_symbols]
        if not new_symbols:
            return
        args = [{"channel": "mark-price", "instId": sym} for sym in new_symbols]
        subscribe_msg = {"op": "subscribe", "args": args}
        if self.ws:
            await self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.update(new_symbols)
            logger.info(f"已订阅标记价格: {new_symbols}")

    async def run(self):
        """持续运行，自动重连"""
        while self.keep_running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    if self.subscribed_symbols:
                        await self.subscribe_mark_price(list(self.subscribed_symbols))
                    async for message in ws:
                        data = json.loads(message)
                        if "arg" in data and data["arg"]["channel"] == "mark-price":
                            inst_id = data["arg"]["instId"]
                            mark_price = float(data["data"][0]["markPx"])
                            await self.callback(inst_id, mark_price)
            except (websockets.ConnectionClosed, Exception) as e:
                logger.warning(f"WebSocket 断开，5秒后重连: {e}")
                await asyncio.sleep(5)
            finally:
                self.ws = None

    async def close(self):
        self.keep_running = False
        if self.ws:
            await self.ws.close()
