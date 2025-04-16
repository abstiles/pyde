import asyncio
from collections.abc import Callable
from datetime import timedelta
from importlib import resources
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Self

from websockets.asyncio.server import Server, ServerConnection, serve

from . import vendored


DEFAULT_RETRY_DELAY = timedelta(milliseconds=100)
DEFAULT_RETRY_COUNT = 30

# The client JS keeps a live websocket connection to the LiveReloadServer,
# waiting for a signal to reload. The signal itself is simple: any message at
# all. When we receive a message, fetch the new version of the page and diff
# it with the old. If the head has changed, reload the page. Otherwise, patch
# the DOM in place with the changes. If we cannot reconnect within the
# configured number of retries, we take that as a signal that the watch server
# is shut down, so we stop retrying.

DIFF_DOM_MODULE = resources.files(vendored) / 'diff-dom' / 'module.js'
DEPENDS_JS = DIFF_DOM_MODULE.read_text()
CLIENT_JS = '''
const dd = new DiffDOM()
const removeThis = doc => doc.getElementById('pyde-livereload-client').remove();
removeThis(document);
const htmlParser = new DOMParser();
const socketUrl = 'ws://{address}:{port}';
const retryDelayMilliseconds = {retry_ms};
const maxAttempts = {retry_count};
let socket = new WebSocket(socketUrl);
let connected = false;
socket.addEventListener('open', () => {{ connected = true; }});
setTimeout(() => {{
	if (connected) return;
	console.error('Failed to connect to livereload service.');
	socket.onclose = () => null;
	socket.close();
}}, retryDelayMilliseconds * maxAttempts);
let attempts = 0;
const reloadIfCanConnect = () => {{
	attempts++;
	if (attempts > maxAttempts) {{
		console.error('Could not reconnect to dev server.');
		return;
	}}
	socket = new WebSocket(socketUrl);
	socket.addEventListener('message', listener);
	socket.addEventListener('error', () => {{
		setTimeout(reloadIfCanConnect, retryDelayMilliseconds);
	}});
	socket.addEventListener('open', () => {{
		attempts = 0;
	}});
}};
const listener = async (event) => {{
	console.log(event.data);
	await fetch(window.location)
		.then(response => response.text())
		.then(update => {{
			const newPage = htmlParser.parseFromString(update, 'text/html');
			removeThis(newPage);
			const newHead = newPage.documentElement.childNodes[0];
			const oldHead = document.documentElement.childNodes[0];
			const newBody = newPage.documentElement.childNodes[2];
			const oldBody = document.documentElement.childNodes[2];
			const tempNodes = [
				document.documentElement.appendChild(newHead),
				document.documentElement.appendChild(newBody),
			];
			let diff = dd.diff(oldHead, newHead);
			if (diff.length > 0) {{
				console.log("Head changed, doing full page reload.");
				location.reload();
			}}
			dd.apply(oldHead, diff);
			diff = dd.diff(oldBody, newBody);
			dd.apply(oldBody, diff);
			for (const tempNode of tempNodes) {{
				tempNode.remove();
			}}
			reloadIfCanConnect();
		}}).catch(reason => console.error(reason))
	;
}};
socket.addEventListener('message', listener);
'''


def reload_listener(server: Server, recv_port: Connection) -> Callable[[], None]:
    def send_reload_message() -> None:
        message = recv_port.recv_bytes().decode('utf-8')
        for connection in server.connections:
            asyncio.create_task(connection.send(message))
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
    _send_port: Connection

    """
    A simple websocket server that ignores messages

    This is useful to send a very simple signal to a client by strategically
    disconnecting. When the server process dies, the client Javascript attempts
    to reconnect, and upon reconnecting, it reloads the page. The reload method
    of this class simply terminates the existing server and starts a new one.
    """
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
            + DEPENDS_JS
            + CLIENT_JS.format(
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

    def reload(self) -> None:
        if self._process:
            # Any message at all functions as the reload signal.
            self._send_port.send_bytes(b'Reloading...')
