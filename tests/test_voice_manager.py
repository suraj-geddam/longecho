import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock the inference module before importing voice_manager
sys.modules['inference'] = Mock()

from longecho.voice_manager import VoiceManager


@pytest.fixture
def mock_models():
    """Create mock Echo models"""
    fish_ae = Mock()
    pca_state = Mock()
    return fish_ae, pca_state


def test_voice_manager_initialization(tmp_path, mock_models):
    """Should initialize with empty voice library"""
    fish_ae, pca_state = mock_models
    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)

    assert vm.voice_dir == tmp_path
    assert len(vm.get_voice_names()) == 0


def test_load_voices_with_cache(tmp_path, mock_models):
    """Should load cached voice data when pkl exists"""
    fish_ae, pca_state = mock_models

    # Create dummy wav and pkl files
    wav_file = tmp_path / "test.wav"
    pkl_file = tmp_path / "test.pkl"

    wav_file.write_text("dummy")

    # Create mock cached data
    cached_data = (
        torch.randn(1, 100, 80),  # speaker_latent
        torch.ones(1, 100, dtype=torch.bool)  # speaker_mask
    )

    import pickle
    import time
    with open(pkl_file, 'wb') as f:
        pickle.dump(cached_data, f)

    # Ensure pkl file has a newer timestamp than wav file
    # by setting wav file mtime to 1 second in the past
    import os
    wav_stat = os.stat(wav_file)
    os.utime(wav_file, (wav_stat.st_atime, wav_stat.st_mtime - 1))

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)
    vm.load_voices()

    assert "test" in vm.get_voice_names()
    latent, mask = vm.get_voice("test")
    assert latent.shape == cached_data[0].shape


def test_get_voice_not_found(tmp_path, mock_models):
    """Should raise error for non-existent voice"""
    fish_ae, pca_state = mock_models
    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)

    with pytest.raises(KeyError):
        vm.get_voice("nonexistent")


def test_add_voice_processes_new_wav(tmp_path, mock_models):
    """Should process a new wav file and add it to voices"""
    fish_ae, pca_state = mock_models

    # Create a wav file
    wav_file = tmp_path / "newvoice.wav"
    wav_file.write_text("dummy wav content")

    # Mock the preprocessing
    mock_latent = torch.randn(1, 100, 80)
    mock_mask = torch.ones(1, 100, dtype=torch.bool)

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)

    with patch.object(vm, '_preprocess_voice', return_value=(mock_latent, mock_mask)):
        voice_name = vm.add_voice(wav_file)

    assert voice_name == "newvoice"
    assert "newvoice" in vm.get_voice_names()


def test_add_voice_skips_already_loaded(tmp_path, mock_models):
    """Should skip processing if voice is already loaded"""
    fish_ae, pca_state = mock_models

    wav_file = tmp_path / "existing.wav"
    wav_file.write_text("dummy")

    mock_latent = torch.randn(1, 100, 80)
    mock_mask = torch.ones(1, 100, dtype=torch.bool)

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)
    vm.voices["existing"] = (mock_latent, mock_mask)

    # Should return None when already loaded
    result = vm.add_voice(wav_file)

    assert result is None


def test_add_voice_creates_cache(tmp_path, mock_models):
    """Should create pkl cache file after processing"""
    fish_ae, pca_state = mock_models

    wav_file = tmp_path / "cacheme.wav"
    wav_file.write_text("dummy")
    pkl_file = tmp_path / "cacheme.pkl"

    mock_latent = torch.randn(1, 100, 80)
    mock_mask = torch.ones(1, 100, dtype=torch.bool)

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)

    with patch.object(vm, '_preprocess_voice', return_value=(mock_latent, mock_mask)):
        vm.add_voice(wav_file)

    assert pkl_file.exists()


def test_remove_voice_removes_loaded(tmp_path, mock_models):
    """Should remove a loaded voice"""
    fish_ae, pca_state = mock_models

    mock_latent = torch.randn(1, 100, 80)
    mock_mask = torch.ones(1, 100, dtype=torch.bool)

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)
    vm.voices["myvoice"] = (mock_latent, mock_mask)

    result = vm.remove_voice("myvoice")

    assert result is True
    assert "myvoice" not in vm.get_voice_names()


def test_remove_voice_returns_false_if_not_found(tmp_path, mock_models):
    """Should return False if voice not loaded"""
    fish_ae, pca_state = mock_models

    vm = VoiceManager(fish_ae, pca_state, voice_dir=tmp_path)

    result = vm.remove_voice("nonexistent")

    assert result is False
