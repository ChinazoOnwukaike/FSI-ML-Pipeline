from pathlib import Path
import librosa
from quality_assessment import quality_assessment
import soundfile as sf
import json



project_root = Path.cwd()
audio_folder = project_root / "data" / "audio" 
output_dir = project_root / "data" / "audio_assessments"
output_dir.mkdir(parents=True, exist_ok=True)

def main():
    
    # Paths
    for file in audio_folder.glob('*.mp3'):  # Only gets .mp3 files
        print(f"Assessing: {file.name}")

        # Get quality assessment
        results = assess_audio(file)

        #Convert to json
        file_pref = file.stem
        output_path =  output_dir / f'{file_pref}.json' 

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    


def assess_audio(file):

    # Load
    audio, sr = librosa.load(file)

    # Assess Audio Quality
    assessment = quality_assessment(audio)
    assessment["filename"] = file.name
    
    return assessment




main()