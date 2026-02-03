from pathlib import Path
import librosa
import noisereduce as nr
import soundfile as sf

# Paths
project_root = Path.cwd()
audio_path = project_root / "data" / "audio" / "SpV1U1A.mp3"
output_path = project_root / "data" / "processed" / "SpV1U1A-Enhanced.mp3"

# Load
audio, sr = librosa.load(audio_path)

# Denoise
reduced_audio = nr.reduce_noise(y=audio, sr=sr)

# Save
sf.write(output_path, reduced_audio, sr)