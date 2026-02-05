import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable, Awaitable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileDeletedEvent

logger = logging.getLogger(__name__)


class _WavFileHandler(FileSystemEventHandler):
    """Handles file system events, filtering for .wav files."""

    def __init__(
        self,
        on_new_wav: Callable[[Path], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float = 1.0,
        on_deleted_wav: Callable[[Path], Awaitable[None]] | None = None,
    ):
        self.on_new_wav = on_new_wav
        self.on_deleted_wav = on_deleted_wav
        self.loop = loop
        self.debounce_seconds = debounce_seconds
        self._pending: dict[Path, asyncio.TimerHandle] = {}
        self._lock = threading.Lock()

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Only process .wav files
        if path.suffix.lower() != '.wav':
            return

        logger.debug(f"Detected new file: {path}")

        # Debounce: cancel any pending callback for this path
        with self._lock:
            if path in self._pending:
                self._pending[path].cancel()

            # Schedule callback after debounce period
            def schedule_callback():
                with self._lock:
                    self._pending.pop(path, None)
                asyncio.run_coroutine_threadsafe(self.on_new_wav(path), self.loop)

            handle = self.loop.call_later(self.debounce_seconds, schedule_callback)
            self._pending[path] = handle

    def on_deleted(self, event: FileDeletedEvent):
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Only process .wav files
        if path.suffix.lower() != '.wav':
            return

        if self.on_deleted_wav is None:
            return

        logger.debug(f"Detected deleted file: {path}")

        # No debounce for deletion - immediate callback
        asyncio.run_coroutine_threadsafe(self.on_deleted_wav(path), self.loop)


class FileWatcher:
    """
    Watches a directory for new .wav files using watchdog.

    Debounces rapid events to handle file copies in progress.
    """

    def __init__(
        self,
        watch_dir: Path,
        on_new_wav: Callable[[Path], Awaitable[None]],
        debounce_seconds: float = 1.0,
        on_deleted_wav: Callable[[Path], Awaitable[None]] | None = None,
    ):
        self.watch_dir = watch_dir
        self.on_new_wav = on_new_wav
        self.on_deleted_wav = on_deleted_wav
        self.debounce_seconds = debounce_seconds
        self._observer: Observer | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start watching the directory."""
        handler = _WavFileHandler(
            self.on_new_wav, loop, self.debounce_seconds, self.on_deleted_wav
        )
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        logger.info(f"Started watching {self.watch_dir} for new .wav files")

    def stop(self) -> None:
        """Stop watching the directory."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("Stopped file watcher")
