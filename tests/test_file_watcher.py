import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from longecho.file_watcher import FileWatcher


@pytest.fixture
def mock_callback():
    return AsyncMock()


def test_file_watcher_initialization(tmp_path, mock_callback):
    """FileWatcher should initialize with directory and callback"""
    watcher = FileWatcher(tmp_path, mock_callback)

    assert watcher.watch_dir == tmp_path
    assert watcher.on_new_wav == mock_callback


def test_file_watcher_filters_non_wav(tmp_path, mock_callback):
    """FileWatcher should ignore non-wav files"""
    watcher = FileWatcher(tmp_path, mock_callback)
    loop = asyncio.new_event_loop()
    watcher.start(loop)

    try:
        # Create a non-wav file
        (tmp_path / "test.txt").write_text("hello")
        time.sleep(0.5)  # Wait for debounce

        mock_callback.assert_not_called()
    finally:
        watcher.stop()
        loop.close()


def test_file_watcher_detects_wav(tmp_path, mock_callback):
    """FileWatcher should detect new wav files"""
    watcher = FileWatcher(tmp_path, mock_callback, debounce_seconds=0.1)
    loop = asyncio.new_event_loop()
    watcher.start(loop)

    try:
        # Create a wav file
        wav_path = tmp_path / "test.wav"
        wav_path.write_text("dummy wav")

        # Wait for debounce + processing, running the loop to process callbacks
        end_time = time.time() + 0.5
        while time.time() < end_time:
            loop.run_until_complete(asyncio.sleep(0.05))

        # Callback should have been called with the wav path
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        assert call_args[0] == wav_path
    finally:
        watcher.stop()
        loop.close()


def test_file_watcher_ignores_pkl(tmp_path, mock_callback):
    """FileWatcher should ignore pkl files"""
    watcher = FileWatcher(tmp_path, mock_callback, debounce_seconds=0.1)
    loop = asyncio.new_event_loop()
    watcher.start(loop)

    try:
        (tmp_path / "test.pkl").write_text("cached data")
        time.sleep(0.3)

        mock_callback.assert_not_called()
    finally:
        watcher.stop()
        loop.close()


def test_file_watcher_detects_deletion(tmp_path, mock_callback):
    """FileWatcher should detect deleted wav files"""
    delete_callback = AsyncMock()
    watcher = FileWatcher(
        tmp_path, mock_callback, debounce_seconds=0.1, on_deleted_wav=delete_callback
    )
    loop = asyncio.new_event_loop()

    # Create a wav file first
    wav_path = tmp_path / "todelete.wav"
    wav_path.write_text("dummy wav")

    watcher.start(loop)

    try:
        # Delete the file
        wav_path.unlink()

        # Wait for callback (no debounce on deletion)
        end_time = time.time() + 0.5
        while time.time() < end_time:
            loop.run_until_complete(asyncio.sleep(0.05))

        # Delete callback should have been called
        delete_callback.assert_called_once()
        call_args = delete_callback.call_args[0]
        assert call_args[0] == wav_path
    finally:
        watcher.stop()
        loop.close()
