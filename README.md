# FSI ML Pipeline

Production ML system for restoring historical audio and extracting structured educational content from Foreign Service Institute language materials.

## What This Does

Converts 50+ year old FSI language courses into modern, structured learning data through:

1. **Audio Restoration (Phase 0)** - Deep learning-based denoising and enhancement of degraded historical recordings
2. **NLP Text Processing (Phase 0.5)** - Automated extraction and classification of lesson content from inconsistent source materials  
3. **Audio-Text Alignment (Phase 0.5)** - Forced alignment to generate precise timestamps for synchronized playback

## Why This Exists

FSI materials are public domain and pedagogically valuable, but unusable in their raw form. This pipeline makes them accessible for modern language learning applications.

## Architecture

**Modular ML components:**

- `audio_ml/` - Audio restoration using spectral denoising and deep learning enhancement
- `nlp_pipeline/` - Text extraction, language identification, semantic classification  
- `alignment/` - Montreal Forced Aligner integration for audio-text synchronization
- `api/` - FastAPI service exposing the full pipeline

Each module operates independently with clear data contracts.

## Current Status

🟢 **Phase 0 (Audio)** - In progress  
⚪ **Phase 0.5 (NLP)** - Not started  
⚪ **Phase 2 (Alignment)** - Not started

## Tech Stack

**ML/NLP:** PyTorch, librosa, spaCy, noisereduce, Whisper  
**API:** FastAPI, Pydantic  
**Storage:** PostgreSQL (structured data), R2/B2 (media)  
**Deployment:** Docker, modal.com (planned)

## Setup
```bash
# Clone and setup environment
git clone https://github.com/yourusername/fsi-ml-pipeline.git
cd fsi-ml-pipeline
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run audio restoration on sample
python audio_ml/denoise.py --input data/raw/sample.mp3 --output data/processed/sample_clean.mp3
```

## Roadmap

- [ ] Phase 0: Audio enhancement pipeline with quality metrics
- [ ] Phase 0.5: NLP text structuring and semantic classification
- [ ] Audio-text forced alignment with confidence scoring
- [ ] FastAPI service with batch processing support
- [ ] Evaluation metrics and benchmark dataset

## License

MIT - FSI source materials are public domain