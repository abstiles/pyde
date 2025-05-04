import asyncio
from collections.abc import Callable
from datetime import timedelta
from enum import auto, Enum
from functools import cache
from importlib import resources
from importlib.abc import Traversable
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Self

from websockets.asyncio.server import Server, ServerConnection, serve

DEFAULT_RETRY_DELAY = timedelta(milliseconds=100)
DEFAULT_RETRY_COUNT = 30


@cache
def js_path() -> Traversable:
    from . import js  # pylint: disable=all # pyright: ignore
    return resources.files(js)


@cache
def depends_js() -> str:
    module = js_path() / 'vendored'/ 'diff-dom' / 'module.js'
    return module.read_text()


# The client JS keeps a live websocket connection to the LiveReloadServer,
# waiting for a signal to reload. The message sent determines whether to do
# a full reload, a refresh of the CSS, or a patched update of the DOM.
# When the client receives a "page" message, it fetches the new version of the
# page and diff it with the old. If the head has changed, reload the page.
# Otherwise, patch the DOM in place with the changes. If we cannot reconnect
# within the configured number of retries, we take that as a signal that the
# watch server is shut down, so we stop retrying.
def client_js() -> str:
    module = js_path() / 'reload-client.js'
    return module.read_text()


async def deliver_messages(connection: ServerConnection, messages: list[str]) -> None:
    for message in messages:
        await connection.send(message)
    await connection.close()


def reload_listener(server: Server, recv_port: Connection) -> Callable[[], None]:
    def send_reload_message() -> None:
        reload_types = recv_port.recv_bytes().decode('utf-8').split(',')
        for connection in server.connections:
            asyncio.create_task(deliver_messages(connection, reload_types))
    return send_reload_message


def launch(
    instance: 'LiveReloadServer', recv_port: Connection,
) -> None:
    async def main() -> None:
        loop = asyncio.get_event_loop()
        async with instance.get_server() as server:
            loop.add_reader(recv_port.fileno(), reload_listener(server, recv_port))
            await server.serve_forever()
    asyncio.run(main())


class LiveReloadServer:
    """A simple websocket server that sends reload messages"""

    _send_port: Connection

    class ReloadType(Enum):
        @staticmethod
        def _generate_next_value_(name: str, *_: object, **__: object) -> bytes:
            return name.lower().encode('utf-8')
        FULL = auto()
        PAGE = auto()
        CSS = auto()

    def __init__(
        self,
        address: str='',
        port: int=8001,
        reconnect_delay: timedelta = DEFAULT_RETRY_DELAY,
        retry_count: int = DEFAULT_RETRY_COUNT,
    ):
        self._address = address
        self._port = port
        self._retry_ms = reconnect_delay.total_seconds() * 1000
        self._retry_count = retry_count
        self._process: Process | None = None

    def client_js(self) -> str:
        return (
            '<script id="pyde-livereload-client" type="module">'
            + depends_js()
            + client_js().format(
                address=self._address or 'localhost',
                port=str(self._port),
                retry_ms=self._retry_ms,
                retry_count=self._retry_count,
            ) + '</script>'
        )

    @staticmethod
    async def message(websocket: ServerConnection) -> None:
        """Ignore messages"""
        async for _ in websocket:
            pass

    def get_server(self) -> serve:
        return serve(self.message, self._address or '0.0.0.0', self._port)

    def start(self) -> Self:
        if not self._process:
            recv_port, self._send_port = Pipe(False)
            # It would almost certainly be faster to use a thread than a whole
            # process, but for some reason I have not been able to get the
            # websocket service to properly shut down across threads. At least
            # processes give me a very simple kill switch.
            self._process = Process(target=launch, args=(self, recv_port))
            self._process.daemon = True
            self._process.start()
        return self

    def stop(self) -> None:
        if self._process:
            self._send_port.close()
            self._process.terminate()
            self._process = None

    def reload(self, *types: ReloadType) -> None:
        if not self._process:
            return
        selected = set(types)
        if not selected:
            type = self.ReloadType.PAGE.value
        elif self.ReloadType.FULL in selected:
            type = self.ReloadType.FULL.value
        else:
            type = b','.join(t.value for t in selected)
        self._send_port.send_bytes(type)
