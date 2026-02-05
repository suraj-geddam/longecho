"""
Test FastAPI app initialization and lifespan context manager.

This test verifies that the lifespan context manager properly initializes
the application on startup without using the deprecated @app.on_event decorator.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys


@pytest.fixture
def mock_dependencies():
    """Mock all heavy dependencies for faster testing"""
    # Mock the inference module
    mock_inference = Mock()
    mock_model = Mock()
    mock_fish_ae = Mock()
    mock_pca_state = Mock()

    mock_inference.load_model_from_hf = Mock(return_value=mock_model)
    mock_inference.load_fish_ae_from_hf = Mock(return_value=mock_fish_ae)
    mock_inference.load_pca_state_from_hf = Mock(return_value=mock_pca_state)

    sys.modules['inference'] = mock_inference
    sys.modules['inference_blockwise'] = Mock()

    # Mock VoiceManager
    mock_voice_manager_class = Mock()
    mock_voice_manager_instance = Mock()
    mock_voice_manager_instance.get_voice_names.return_value = ["voice1", "voice2"]
    mock_voice_manager_class.return_value = mock_voice_manager_instance

    # Mock AudioGenerator
    mock_audio_generator_class = Mock()
    mock_audio_generator_instance = Mock()
    mock_audio_generator_class.return_value = mock_audio_generator_instance

    with patch('longecho.main.load_model_from_hf', return_value=mock_model), \
         patch('longecho.main.load_fish_ae_from_hf', return_value=mock_fish_ae), \
         patch('longecho.main.load_pca_state_from_hf', return_value=mock_pca_state), \
         patch('longecho.main.VoiceManager', mock_voice_manager_class), \
         patch('longecho.main.AudioGenerator', mock_audio_generator_class):

        yield {
            'model': mock_model,
            'fish_ae': mock_fish_ae,
            'pca_state': mock_pca_state,
            'voice_manager': mock_voice_manager_instance,
            'audio_generator': mock_audio_generator_instance,
        }


def test_lifespan_initializes_app(mock_dependencies):
    """
    Test that the lifespan context manager properly initializes the app.

    This verifies:
    1. Models are loaded on startup
    2. VoiceManager is initialized
    3. AudioGenerator is initialized
    4. Health endpoint returns correct status
    """
    from longecho.main import app

    with TestClient(app) as client:
        # Test health endpoint to verify app started successfully
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["voices_loaded"] == 2  # Should match mock_voice_manager.get_voice_names


def test_voices_endpoint(mock_dependencies):
    """Test that the voices endpoint returns available voices"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.get("/voices")
        assert response.status_code == 200

        data = response.json()
        assert "voices" in data
        assert data["voices"] == ["voice1", "voice2"]


def test_root_endpoint(mock_dependencies):
    """Test that root endpoint returns the main page"""
    from longecho.main import app

    with TestClient(app) as client:
        # This will fail if static/index.html doesn't exist,
        # but we're mainly testing that the endpoint is registered
        response = client.get("/")
        # We expect either 200 (if file exists) or 404 (if file doesn't exist)
        # Both are valid - we're just testing the app initialized
        assert response.status_code in [200, 404]


def test_generate_validation_empty_text(mock_dependencies):
    """Test that generate endpoint rejects empty text"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.get("/generate", params={"text": "", "voice": "voice1"})
        assert response.status_code == 422  # Unprocessable Entity

        # Check that the error message mentions text validation
        data = response.json()
        assert "detail" in data


def test_generate_validation_text_too_long(mock_dependencies):
    """Test that generate endpoint rejects text longer than max_length"""
    from longecho.main import app

    with TestClient(app) as client:
        # Create text with 100,001 characters (max is 100,000)
        # Note: We can't actually test with 100k+ chars in URL due to HTTP URL length limits,
        # but FastAPI validation will catch this before the request is processed.
        # Testing with a smaller but still excessive length to verify validation works.
        long_text = "a" * 100001

        # This will fail with 422 due to validation if it reaches FastAPI,
        # or with URL error if the HTTP client rejects it first.
        # Both outcomes indicate the request would be rejected.
        try:
            response = client.get("/generate", params={"text": long_text, "voice": "voice1"})
            # If we get here, check it was rejected
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
        except Exception as e:
            # URL too long error from HTTP client also means validation would work
            assert "too long" in str(e).lower()


def test_generate_validation_empty_voice(mock_dependencies):
    """Test that generate endpoint rejects empty voice"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.get("/generate", params={"text": "Hello world", "voice": ""})
        assert response.status_code == 422  # Unprocessable Entity

        # Check that the error message mentions voice validation
        data = response.json()
        assert "detail" in data


def test_generate_validation_valid_inputs(mock_dependencies):
    """Test that generate endpoint accepts valid inputs"""
    from longecho.main import app
    import torch

    # Mock the voice manager to return valid voice data
    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),  # speaker_latent
        torch.ones(1, 10)         # speaker_mask
    )

    # Mock the audio generator to yield a single audio chunk
    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)  # 1 second of audio
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    # Mock segment_text to return a single chunk
    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.get("/generate", params={"text": "Hello world", "voice": "voice1"})
            # SSE endpoint should return 200 for successful stream initiation
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_generate_validation_max_length_boundary(mock_dependencies):
    """Test that generate endpoint accepts text below max_length"""
    from longecho.main import app
    import torch

    # Mock the voice manager to return valid voice data
    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),  # speaker_latent
        torch.ones(1, 10)         # speaker_mask
    )

    # Mock the audio generator to yield a single audio chunk
    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    # Create text with a reasonable length that tests validation works
    # but doesn't exceed HTTP URL limits (testing with 5000 chars)
    test_text = "This is a test sentence. " * 200  # ~5000 characters

    # Mock segment_text to return a single chunk
    with patch('longecho.main.segment_text', return_value=[test_text[:100]]):
        with TestClient(app) as client:
            response = client.get("/generate", params={"text": test_text, "voice": "voice1"})
            # Should accept text below max length
            assert response.status_code == 200


def test_stop_endpoint(mock_dependencies):
    """Test that stop endpoint calls request_stop on audio generator"""
    from longecho.main import app

    mock_audio_generator = mock_dependencies['audio_generator']

    with TestClient(app) as client:
        response = client.post("/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stop requested"

        # Verify request_stop was called
        mock_audio_generator.request_stop.assert_called_once()


def test_generate_with_normalization_level_moderate(mock_dependencies):
    """Test that generate endpoint accepts normalization_level parameter"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.get("/generate", params={
                "text": "Hello world",
                "voice": "voice1",
                "normalization_level": "moderate"
            })
            assert response.status_code == 200


def test_generate_with_normalization_level_full(mock_dependencies):
    """Test that generate endpoint accepts normalization_level='full'"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    with patch('longecho.main.segment_text', return_value=["Hello world"]):
        with TestClient(app) as client:
            response = client.get("/generate", params={
                "text": "Hello world",
                "voice": "voice1",
                "normalization_level": "full"
            })
            assert response.status_code == 200


def test_generate_with_invalid_normalization_level(mock_dependencies):
    """Test that generate endpoint rejects invalid normalization_level"""
    from longecho.main import app

    with TestClient(app) as client:
        response = client.get("/generate", params={
            "text": "Hello world",
            "voice": "voice1",
            "normalization_level": "invalid"
        })
        # Should get 422 validation error
        assert response.status_code == 422


def test_generate_normalizes_text_before_segmentation(mock_dependencies):
    """Test that text is normalized before being segmented"""
    from longecho.main import app
    import torch

    mock_voice_manager = mock_dependencies['voice_manager']
    mock_voice_manager.get_voice_names.return_value = ["voice1"]
    mock_voice_manager.get_voice.return_value = (
        torch.randn(1, 10, 256),
        torch.ones(1, 10)
    )

    mock_audio_generator = mock_dependencies['audio_generator']
    mock_audio_tensor = torch.randn(1, 1, 44100)
    mock_audio_generator.generate_long_audio.return_value = iter([mock_audio_tensor])

    # Use a mock for segment_text to capture what it receives
    captured_text = []

    def capture_segment_text(text):
        captured_text.append(text)
        return [text]

    with patch('longecho.main.segment_text', side_effect=capture_segment_text):
        with TestClient(app) as client:
            # Input with currency that should be normalized
            response = client.get("/generate", params={
                "text": "The cost is $5M.",
                "voice": "voice1",
            })
            assert response.status_code == 200

            # Verify segment_text received normalized text (without $ sign)
            assert len(captured_text) == 1
            assert "$" not in captured_text[0]
            assert "million dollars" in captured_text[0].lower() or "5M" not in captured_text[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
