import pytest
from pathlib import Path


@pytest.fixture
def tmp_voice_dir(tmp_path):
    """Create temporary voice directory"""
    voice_dir = tmp_path / "voice_library"
    voice_dir.mkdir()
    return voice_dir
