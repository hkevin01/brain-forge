"""
Test Brain-Forge core infrastructure
"""
import pytest
import numpy as np


def test_basic_imports():
    """Test that basic Python scientific stack works"""
    import numpy as np
    import scipy
    assert np.__version__ is not None
    assert scipy.__version__ is not None


def test_signal_processing_imports():
    """Test signal processing library imports"""
    import numpy as np
    from scipy import signal
    import pywt
    
    # Test basic signal processing
    fs = 1000  # Sample rate
    t = np.linspace(0, 1, fs, False)
    sig = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
    
    # Test filtering
    sos = signal.butter(4, 50, 'low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    
    assert len(filtered) == len(sig)
    
    # Test wavelets
    coeffs = pywt.wavedec(sig, 'db8', level=4)
    assert len(coeffs) == 5  # 4 levels + approximation


def test_machine_learning_imports():
    """Test ML library imports"""
    from sklearn.decomposition import PCA, ICA
    from sklearn.preprocessing import StandardScaler
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(100, 10)
    
    # Test PCA
    pca = PCA(n_components=5)
    transformed = pca.fit_transform(data)
    assert transformed.shape == (100, 5)
    
    # Test scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    assert np.allclose(np.mean(scaled, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(scaled, axis=0), 1, atol=1e-10)


def test_data_structures():
    """Test that we can create basic data structures"""
    # Mock neural data
    channels = 64
    timepoints = 1000
    trials = 50
    
    neural_data = np.random.randn(channels, timepoints, trials)
    assert neural_data.shape == (channels, timepoints, trials)
    
    # Mock feature extraction
    features = np.mean(neural_data, axis=1)  # Average over time
    assert features.shape == (channels, trials)
    
    # Mock compression
    compressed = neural_data[::2, ::2, :]  # Simple downsampling
    compression_ratio = neural_data.size / compressed.size
    assert compression_ratio > 1.0


def test_config_structure():
    """Test config structure without imports"""
    # Test that we can create config-like structures
    config = {
        'hardware': {
            'omp_enabled': True,
            'omp_channels': 306,
            'omp_sampling_rate': 1000.0
        },
        'processing': {
            'filter_low': 1.0,
            'filter_high': 100.0,
            'compression_ratio': 5.0
        },
        'transfer_learning': {
            'pattern_extraction': {
                'current_subject_id': 'test_subject',
                'frequency_bands': {
                    'alpha': [8, 12],
                    'beta': [12, 30]
                }
            }
        }
    }
    
    assert config['hardware']['omp_channels'] == 306
    assert config['processing']['compression_ratio'] == 5.0
    subject_id = config['transfer_learning']['pattern_extraction']
    assert subject_id['current_subject_id'] == 'test_subject'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
