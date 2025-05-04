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
const reconnect = () => {{
	attempts++;
	if (attempts > maxAttempts) {{
		console.error('Could not reconnect to dev server.');
		return;
	}}
	socket = new WebSocket(socketUrl);
	socket.addEventListener('message', listener);
	socket.addEventListener('close', () => {{
		if (socket.readyState != 0) {{
			reconnect();
		}} else {{
			setTimeout(reconnect, retryDelayMilliseconds);
		}}
	}});
	socket.addEventListener('open', () => {{
		attempts = 0;
	}});
}};
const reloadCss = () => {{
	const links = [...document.getElementsByTagName("link")];
	for (const link of links) {{
		if (link.rel === "stylesheet") {{
			const newCss = link.cloneNode();
			link.insertAdjacentElement('afterend', newCss);
			newCss.onload = () => link.remove();
		}}
	}}
}}
const listener = async (event) => {{
	console.log(event.data)
	if (event.data === 'full') {{
		location.reload();
	}} else if (event.data === 'css') {{
		reloadCss();
	}} else {{
		await fetch(location)
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
					console.log(JSON.stringify(diff));
					console.log("Head changed, doing full page reload.");
					location.reload();
				}}
				dd.apply(oldHead, diff);
				diff = dd.diff(oldBody, newBody);
				dd.apply(oldBody, diff);
				for (const tempNode of tempNodes) {{
					tempNode.remove();
				}}
			}}).catch(reason => console.error(reason))
		;
	}}
}}
socket.addEventListener('message', listener);
socket.addEventListener('close', reconnect);
