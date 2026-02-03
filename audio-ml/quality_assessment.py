import librosa
from numpy import mean, clip


def quality_assessment(audio):
    """
    Returns quality metrics that inform processing decisions.
    
    Returns:
    {
        'rms_energy': float,
        'spectral_centroid': float,
        'spectral_flatness': float,
        'zero_crossing_rate': float,
        'enhancement_strategy': str,  # 'aggressive' | 'moderate' | 'light'
        'requires_denoising': bool,
        'severity_score': float  # 0-1, where 1 = needs heavy work
    }
    """
    spectral_flatness = mean(librosa.feature.spectral_flatness(y=audio))
    spectral_centroid = mean(librosa.feature.spectral_centroid(y=audio))
    zero_crossing_rate = mean(librosa.feature.zero_crossing_rate(y=audio))
    rms = mean(librosa.feature.rms(y=audio))

    scores = normalize_metrics(rms, spectral_centroid, spectral_flatness, zero_crossing_rate)

    severity = calculate_severity(*scores)

    strategy_info = determine_strategy(severity)

    return {
        'rms_energy': float(rms),
        'spectral_flatness': float(spectral_flatness),
        'spectral_centroid': float(spectral_centroid),
        'zero_crossing_rate': float(zero_crossing_rate),
        'severity_score': float(severity),
        **strategy_info  # Unpack strategy dict
    } 




# Metric Helpers
def normalize_metrics(rms, spectral_centroid, spectral_flatness, zcr):
    """
    Convert each metric to 0-1 severity score.
    Higher = worse quality = needs more enhancement
    """
    
    # RMS: Lower = worse (weak signal)
    # Typical range: 0.01 - 0.3
    rms_score = 1 - clip(rms / 0.3, 0, 1)
    
    # Spectral Centroid: Lower = worse (muffled)
    # Typical range: 1000 - 4000 Hz
    centroid_score = 1 - clip(spectral_centroid / 4000, 0, 1)
    
    # Spectral Flatness: Lower = worse (noisy/tonal artifacts)
    # Range: 0 - 1 already
    flatness_score = 1 - spectral_flatness
    
    # Zero Crossing Rate: Higher = worse (more noise)
    # Typical range: 0.01 - 0.5
    zcr_score = clip(zcr / 0.5, 0, 1)
    
    return rms_score, centroid_score, flatness_score, zcr_score


def calculate_severity(rms_score, centroid_score, flatness_score, zcr_score):
    """
    Combine normalized scores with weights.
    Returns severity: 0 (perfect) to 1 (terrible)
    """
    
    weights = {
        'rms': 0.3,        # Signal strength important
        'centroid': 0.2,   # Clarity matters
        'flatness': 0.3,   # Noisiness critical
        'zcr': 0.2         # Additional noise indicator
    }
    
    severity = (
        weights['rms'] * rms_score +
        weights['centroid'] * centroid_score +
        weights['flatness'] * flatness_score +
        weights['zcr'] * zcr_score
    )
    
    return severity  # 0.0 to 1.0


def determine_strategy(severity):
    """
    Map severity score to enhancement strategy.
    """
    
    if severity >= 0.7:
        return {
            'strategy': 'aggressive',
            'denoise_strength': 1.0,
            'enhance_strength': 1.0,
            'requires_denoising': True
        }
    elif severity >= 0.4:
        return {
            'strategy': 'moderate',
            'denoise_strength': 0.6,
            'enhance_strength': 0.8,
            'requires_denoising': True
        }
    else:
        return {
            'strategy': 'light',
            'denoise_strength': 0.0,
            'enhance_strength': 0.5,
            'requires_denoising': False
        }