"""运动准备前预约推理连接，防止并发客户端污染在线记忆。"""

import asyncio

from openpi_client import msgpack_numpy
import pytest
from websockets.exceptions import ConnectionClosedOK

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class Socket:
    remote_address = ("127.0.0.1", 1000)

    def __init__(self, requests):
        self.requests = list(requests)
        self.sent = []
        self.closed = False
        self.before_receive = None

    async def send(self, value):
        self.sent.append(value)

    async def recv(self):
        if self.before_receive is not None:
            callback, self.before_receive = self.before_receive, None
            await callback()
        if self.requests:
            return msgpack_numpy.packb(self.requests.pop(0))
        raise ConnectionClosedOK(None, None)

    async def close(self, **kwargs):
        self.closed = True


def test_claim_prevents_other_connection_from_running_policy_and_releases():
    class Policy:
        def __init__(self):
            self.calls = []

        def infer(self, observation):
            self.calls.append(observation)
            return {"actions": []}

    async def run():
        policy = Policy()
        server = WebsocketPolicyServer(policy, single_client=True)
        first = Socket([{"operation": "claim"}, {"episode_id": "first"}])
        second = Socket([{"operation": "claim"}])
        original_send = first.send

        async def first_send(value):
            await original_send(value)
            if isinstance(value, bytes) and msgpack_numpy.unpackb(value).get("claimed"):
                with pytest.raises(RuntimeError, match="已有客户端"):
                    await server._handler(second)
                assert server._inference_owner is first

        first.send = first_send
        await server._handler(first)
        assert policy.calls == [{"episode_id": "first"}]
        assert server._inference_owner is None
        assert second.closed
        next_socket = Socket([{"operation": "claim"}, {"episode_id": "next"}])
        await server._handler(next_socket)
        assert policy.calls[-1] == {"episode_id": "next"}
        assert server._inference_owner is None

    asyncio.run(run())
