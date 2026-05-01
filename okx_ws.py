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
        self._ping_task = None

    async def _send_ping(self):
        """每 20 秒发送一次应用层 ping，保持连接活跃"""
        while self.ws and self.ws.open:
            try:
                await self.ws.send("ping")
                await asyncio.sleep(20)
            except Exception:
                break

    async def subscribe_mark_price(self, symbols):
        """订阅一个或多个币种的标记价格"""
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
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,      # 关闭协议层 ping，改用应用层
                    ping_timeout=None,
                    close_timeout=5
                ) as ws:
                    self.ws = ws
                    # 启动应用层心跳
                    self._ping_task = asyncio.create_task(self._send_ping())
                    # 重新订阅已有币种
                    if self.subscribed_symbols:
                        await self.subscribe_mark_price(list(self.subscribed_symbols))
                    async for message in ws:
                        if message == "pong":
                            continue
                        data = json.loads(message)
                        if "arg" in data and data["arg"]["channel"] == "mark-price":
                            inst_id = data["arg"]["instId"]
                            mark_price = float(data["data"][0]["markPx"])
                            await self.callback(inst_id, mark_price)
            except (websockets.ConnectionClosed, Exception) as e:
                logger.warning(f"WebSocket 断开，5秒后重连: {e}")
                await asyncio.sleep(5)
            finally:
                if self._ping_task:
                    self._ping_task.cancel()
                self.ws = None

    async def close(self):
        self.keep_running = False
        if self._ping_task:
            self._ping_task.cancel()
        if self.ws:
            await self.ws.close()
        logger.info("WebSocket 已关闭")
