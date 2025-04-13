from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from typing import Any, Self


# Taken and lightly modified from https://stackoverflow.com/a/66558182/648855
class AnimateLoader:
    def __init__(self, desc: str="Loading...", end: str="Done!", fps: float=10):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            fps (float, optional): Animation frames per second. Default 10.
        """
        self.desc = desc
        self.end = end
        self.sleep_time = 1 / max(fps, 0.1)

        self._thread: Thread | None = None
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = True

    def start(self) -> Self:
        if not self._thread:
            self._thread = Thread(target=self._animate, daemon=True)
            self.done = False
            self._thread.start()
        return self

    def _animate(self) -> None:
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.sleep_time)

    def __enter__(self) -> None:
        self.start()

    def terminate_thread(self) -> None:
        self.done = True
        self._thread = None

    def stop(self) -> None:
        self.terminate_thread()
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols + "\r", end="", flush=True)
        print(f"\r{self.end}", end="", flush=True)

    def __exit__(self, exc_type: type[Exception] | None, *_: Any) -> None:
        if exc_type is None:
            self.stop()
        else:
            self.terminate_thread()
            print()
