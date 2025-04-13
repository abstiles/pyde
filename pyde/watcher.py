import time
from collections.abc import Collection
from pathlib import Path
from typing import Any, Protocol, Self

from watchdog.events import (
    EVENT_TYPE_CREATED,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_MODIFIED,
    EVENT_TYPE_MOVED,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from . import pathex
from .path import LocalPath


class Listener(Protocol):
    def update(self, path: LocalPath, /) -> None: ...
    def delete(self, path: LocalPath, /) -> None: ...


class NullListener:
    @staticmethod
    def update(path: LocalPath, /) -> None:
        pass
    @staticmethod
    def delete(path: LocalPath, /) -> None:
        pass


class SourceWatcher(FileSystemEventHandler):
    _listener: Listener

    def __init__(
        self,
        source_dir: LocalPath | Path | str,
        *,
        excluded: Collection[LocalPath | Path | str],
        included: Collection[LocalPath | Path | str],
    ):
        if not isinstance(source_dir, LocalPath):
            source_dir = LocalPath(source_dir)
        self._source_dir = source_dir.absolute()
        self._excluded = [*map(pathex.compile, map(str, excluded))]
        self._included = [*map(pathex.compile, map(str, included))]
        self._observer: BaseObserver | None = None
        self._listener = NullListener

    def register(self, listener: Listener) -> None:
        self._listener = listener

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def start(self) -> Self:
        if self._observer:
            raise RuntimeError(f'{type(self).__name__} already started.')
        self._observer = Observer()
        self._observer.schedule(
            self,
            str(self._source_dir),
            recursive=True,
            event_filter=[
                FileCreatedEvent, FileModifiedEvent,
                FileDeletedEvent, FileMovedEvent,
            ],
        )
        self._observer.start()
        return self

    def stop(self) -> None:
        if not self._observer:
            raise RuntimeError(f'{type(self).__name__} already stopped.')
        self._observer.stop()
        self._observer.join()
        self._observer = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type == EVENT_TYPE_MOVED and self.matches(event.dest_path):
            self.update(event.dest_path)
        if not self.matches(event.src_path):
            return
        if event.event_type in (EVENT_TYPE_CREATED, EVENT_TYPE_MODIFIED):
            self.update(event.src_path)
        elif event.event_type in (EVENT_TYPE_MOVED, EVENT_TYPE_DELETED):
            self.delete(event.src_path)

    def update(self, path: str | bytes) -> None:
        if isinstance(path, bytes):
            path = path.decode(errors='ignore')
        relative_path = LocalPath(path).relative_to(self._source_dir)
        self._listener.update(relative_path)

    def delete(self, path: str | bytes) -> None:
        if isinstance(path, bytes):
            path = path.decode(errors='ignore')
        relative_path = Path(path).relative_to(self._source_dir)
        print(f'Delete {relative_path!r}')

    def matches(self, path: str | bytes) -> bool:
        return (
            any(include.match(path) for include in self._included)
            or not any(exclude.match(path) for exclude in self._excluded)
        )


def eternal_watcher(
    path: LocalPath=LocalPath('.'),
    *,
    excluded: Collection[LocalPath]=(),
    included: Collection[LocalPath]=(),
) -> None:
    """Watches forever"""
    with SourceWatcher(path, excluded=excluded, included=included):
        while True:
            time.sleep(1)
