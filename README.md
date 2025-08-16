# Multimedia-Enhanced Requirements Engineering Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16885966.svg)](https://doi.org/10.5281/zenodo.16885966)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This dataset accompanies the research paper "Knowledge Extraction from Multimedia Data in Requirements Engineering: An Empirical Study" published in Requirements Engineering journal.

The dataset contains anonymized multimedia data from a controlled experiment with 60 participants across three Czech universities, comparing traditional text-based requirements elicitation with multimedia-enhanced approaches.

## Dataset Description

- **Participants**: 60 (20 educators, 20 students, 20 administrators)
- **Institutions**: Tomas Bata University, Czech Technical University, University of Economics Prague
- **Domain**: E-learning platform requirements
- **Collection Period**: January-March 2025
- **Ethics Approval**: IRB-2025-SE-003

## Data Structure

```
dataset/
├── metadata/
│   ├── dataset_info.json
│   ├── participant_demographics.csv
│   └── session_metadata.csv
├── audio_recordings/
│   ├── control_group/
│   └── treatment_group/
├── video_recordings/
│   ├── interaction_sessions/
│   └── screen_recordings/
├── image_annotations/
│   ├── screenshots/
│   └── annotations/
├── transcripts/
│   ├── manual_transcripts/
│   └── automated_transcripts/
├── ground_truth/
│   ├── expert_requirements.json
│   ├── validation_scores.csv
│   └── inter_rater_reliability.csv
└── participant_data/
    ├── satisfaction_scores.csv
    ├── demographic_data.csv
    └── session_logs.csv
```

## Data Statistics

| Data Type | Control Group | Treatment Group | Total |
|-----------|--------------|-----------------|-------|
| Audio Recordings | 30 | 30 | 60 |
| Video Sessions | 0 | 30 | 30 |
| Screen Recordings | 0 | 30 | 30 |
| Annotated Screenshots | 0 | 180 | 180 |
| Manual Transcripts | 30 | 30 | 60 |
| Expert Requirements | 127 unique requirements |

## Usage

### Loading the Dataset

```python
import pandas as pd
import json

# Load participant demographics
demographics = pd.read_csv('metadata/participant_demographics.csv')

# Load ground truth requirements
with open('ground_truth/expert_requirements.json', 'r') as f:
    requirements = json.load(f)

# Load satisfaction scores
satisfaction = pd.read_csv('participant_data/satisfaction_scores.csv')
```

### Audio Processing

```python
import librosa
import os

# Load audio recording
audio_path = 'audio_recordings/treatment_group/participant_001.wav'
audio, sr = librosa.load(audio_path, sr=16000)
```

## Privacy and Ethics

- All data has been anonymized following GDPR requirements
- Participant identifiers have been replaced with random IDs
- Personal information has been removed from transcripts
- Ethics approval obtained from university review board (IRB-2024-SE-003)
- All participants provided informed consent for data sharing

## File Formats

- **Audio**: WAV format, 16kHz sampling rate, mono
- **Video**: MP4 format, 1080p resolution, H.264 codec
- **Images**: PNG format, variable resolution
- **Transcripts**: JSON format with timestamps
- **Metadata**: CSV and JSON formats

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{okechukwu2024multimedia,
  title={Knowledge Extraction from Multimedia Data in Requirements Engineering: An Empirical Study},
  author={Okechukwu, Cornelius Chimuanya},
  journal={****},
  year={2025},
  publisher={****}
}
```

## License

This dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Contact

For questions about the dataset, please contact:
- Cornelius Chimuanya Okechukwu: okechukwu@utb.cz
- Faculty of Applied Informatics, Tomas Bata University in Zlin

## Acknowledgments

We thank all participants from the three universities who contributed to this research. Special thanks to the expert panel members for their validation work.

## Version History

- v1.0.0 (2024-08-15): Initial release with complete dataset
