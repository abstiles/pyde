from multiprocessing import Process
from typing import Self

from websockets.sync.server import Server, ServerConnection, serve

CLIENT_JS = '''
<script>
(() => {{
	const socketUrl = 'ws://{address}:{port}';
	let socket = new WebSocket(socketUrl);
	socket.addEventListener('close', () => {{
		// Then the server has been turned off,
		// either due to file-change-triggered reboot,
		// or to truly being turned off.

		// Attempt to re-establish a connection until it works,
		// failing after a few seconds (at that point things are likely
		// turned off/permanantly broken instead of rebooting)
		const interAttemptTimeoutMilliseconds = 100;
		const maxDisconnectedTimeMilliseconds = 3000;
		const maxAttempts = Math.round(
			maxDisconnectedTimeMilliseconds / interAttemptTimeoutMilliseconds,
		);
		let attempts = 0;
		const reloadIfCanConnect = () => {{
			attempts++;
			if (attempts > maxAttempts) {{
				console.error('Could not reconnect to dev server.');
				return;
			}}
			socket = new WebSocket(socketUrl);
			socket.addEventListener('error', () => {{
				setTimeout(reloadIfCanConnect, interAttemptTimeoutMilliseconds);
			}});
			socket.addEventListener('open', () => {{
				console.log('Reloading');
				location.reload();
			}});
		}};
		reloadIfCanConnect();
	}});
}})();
</script>
'''


def launch(instance: 'LiveReloadServer') -> None:
    with instance.get_server() as server:
        server.serve_forever()


class LiveReloadServer:
    """
    A simple websocket server that ignores messages

    This is useful to send a very simple signal to a client by strategically
    disconnecting. When the server process dies, the client Javascript attempts
    to reconnect, and upon reconnecting, it reloads the page. The reload method
    of this class simply terminates the existing server and starts a new one.
    """
    def __init__(self, address: str='', port: int=8001):
        self._address = address
        self._port = port
        self._process: Process | None = None

    def client_js(self) -> str:
        return CLIENT_JS.format(
            address=self._address or 'localhost',
            port=str(self._port),
        )

    @staticmethod
    def message(websocket: ServerConnection) -> None:
        """Ignore messages"""
        for _ in websocket:
            pass

    def get_server(self) -> Server:
        return serve(self.message, self._address or '0.0.0.0', self._port)

    def start(self) -> Self:
        if not self._process:
            # It would almost certainly be faster to use a thread than a whole
            # process, but for some reason I have not been able to get the
            # websocket service to properly shut down across threads. At least
            # processes give me a very simple kill switch.
            self._process = Process(target=launch, args=(self,))
            self._process.daemon = True
            self._process.start()
        return self

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            self._process = None

    def reload(self) -> None:
        self.stop()
        self.start()
