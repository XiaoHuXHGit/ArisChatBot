import asyncio
import websockets
import json
import logging
from NetworkModule.MessageProcess import MessageProcess

logger = logging.getLogger(__name__)


class WebSocketServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self._server = None
        self._running = False

    def broadcast_message(self, message: str):
        """向所有连接的客户端广播消息"""
        if not self.connected_clients:
            return
        message_to_send = f"[Broadcast] {message}"
        tasks = [asyncio.create_task(client.send(message_to_send)) for client in self.connected_clients]
        asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))
        logger.info(f"向 {len(self.connected_clients)} 个客户端广播消息: {message}")

    async def handle_message(self, websocket, message_text: str):
        """处理客户端发来的消息"""
        try:
            data = json.loads(message_text)
            message_process = MessageProcess(data)
            send_message = await message_process.process()
            if send_message:
                await self.handle_send_message(websocket, send_message)
        except json.JSONDecodeError:
            error_response = {"type": "error", "message": "Invalid JSON received"}
            await websocket.send(json.dumps(error_response))
            logger.warning(f"客户端 {websocket.remote_address} 发送了无效 JSON: {message_text}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {websocket.remote_address} 在消息处理过程中断开连接。")
        except Exception as e:
            logger.error(f"客户端 {websocket.remote_address} 出现错误: {str(e)}")
            error_response = {"type": "error", "message": f"服务器错误: {str(e)}"}
            try:
                await websocket.send(json.dumps(error_response))
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"无法向 {websocket.remote_address} 发送错误响应，客户端已断开连接。")

    async def handle_send_message(self, websocket, data: list[dict]):
        """
        向特定的客户端发送消息
        """
        for message in data:
            await websocket.send(json.dumps(message))

    async def handle_client(self, websocket):
        """处理单个客户端连接"""
        logger.info(f"新客户端已连接: {websocket.remote_address}")
        self.connected_clients.add(websocket)

        try:
            async for message in websocket:
                logger.info(f"从 {websocket.remote_address} 收到消息: {message}")
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {websocket.remote_address} 已断开连接。")
        except Exception as e:
            logger.error(f"客户端 {websocket.remote_address} 出现错误: {str(e)}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"清理了客户端 {websocket.remote_address} 的连接")

    async def start(self):
        """启动 WebSocket 服务器"""
        if self._running:
            logger.warning("服务器已在运行中")
            return None

        self._server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            # ping_interval=20,
            # ping_timeout=10
        )
        self._running = True
        logger.info(f"WebSocket 服务器已在 ws://{self.host}:{self.port} 启动")

    async def stop(self):
        """停止 WebSocket 服务器并清理连接"""
        if not self._running or self._server is None:
            logger.warning("服务器未运行，无需停止")
            return None

        # 关闭所有客户端连接
        close_tasks = [client.close() for client in self.connected_clients]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            logger.info(f"已关闭 {len(close_tasks)} 个客户端连接")

        # 关闭服务器
        self._server.close()
        await self._server.wait_closed()
        self._running = False
        logger.info("WebSocket 服务器已停止")


# 示例用法（仅当直接运行此模块时）
if __name__ == "__main__":
    from configs import ConfigManager

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config = ConfigManager("websocket_config")
    server_host = config.websocket_config.server_host
    server_port = config.websocket_config.server_port

    server = WebSocketServer(server_host, server_port)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        loop.run_until_complete(server.stop())
    finally:
        loop.close()
