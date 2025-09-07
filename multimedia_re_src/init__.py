"""
Multimedia Requirements Engineering (multimedia_re_src)

A comprehensive framework for extracting requirements from multimedia data
including audio, video, and image processing for requirements engineering.

This package provides tools for:
- Multimedia data synchronization and processing
- Audio analysis (speech-to-text, sentiment analysis, keyword extraction)
- Video analysis (user behavior, interaction patterns, confusion detection)
- Image processing (OCR, UI component detection, annotation analysis)
- Cross-modal validation and requirements extraction

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University, Czech Republic
"""

__version__ = "1.0.0"
__author__ = "Cornelius Chimuanya Okechukwu"
__email__ = "okechukwu@utb.cz"
__institution__ = "Tomas Bata University, Czech Republic"

# Import main functions from each module
try:
    # Multimedia processing core functions
    from .multimedia_processing import (
        synchronize_multimedia_streams,
        validate_data_quality,
        create_unified_timeline,
        align_to_timeline,
        extract_audio_timestamps,
        extract_video_timestamps,
        extract_image_metadata_timestamps
    )
except ImportError as e:
    print(f"Warning: Could not import multimedia_processing module: {e}")

try:
    # Audio processing functions
    from .audio_processing import (
        robust_audio_processing,
        fallback_audio_processing,
        validate_audio_results,
        process_audio_primary,
        check_audio_snr,
        extract_sentiment,
        extract_keywords,
        detect_vocal_emphasis
    )
except ImportError as e:
    print(f"Warning: Could not import audio_processing module: {e}")

try:
    # Video processing functions
    from .video_processing import (
        process_video_with_quality_control,
        assess_video_quality,
        process_high_quality_video,
        process_low_quality_video,
        detect_user_interactions,
        identify_hesitation_points,
        detect_confusion_indicators,
        track_attention_regions
    )
except ImportError as e:
    print(f"Warning: Could not import video_processing module: {e}")

try:
    # Image processing functions
    from .image_processing import (
        process_images,
        extract_text_from_image,
        detect_ui_components,
        extract_annotations,
        analyze_visual_elements,
        map_spatial_annotations
    )
except ImportError as e:
    print(f"Warning: Could not import image_processing module: {e}")

try:
    # Requirements extraction functions
    from .requirements_extraction import (
        extract_requirements_from_multimedia,
        classify_requirements,
        validate_cross_modal_evidence,
        generate_requirements_report,
        create_traceability_matrix
    )
except ImportError as e:
    print(f"Warning: Could not import requirements_extraction module: {e}")

# Define what gets imported with "from multimedia_re_src import *"
__all__ = [
    # Core multimedia processing
    'synchronize_multimedia_streams',
    'validate_data_quality',
    'create_unified_timeline',
    'align_to_timeline',
    
    # Audio processing
    'robust_audio_processing',
    'fallback_audio_processing',
    'validate_audio_results',
    'process_audio_primary',
    
    # Video processing
    'process_video_with_quality_control',
    'assess_video_quality',
    'detect_user_interactions',
    'identify_hesitation_points',
    'detect_confusion_indicators',
    
    # Image processing
    'process_images',
    'extract_text_from_image',
    'detect_ui_components',
    'extract_annotations',
    
    # Requirements extraction
    'extract_requirements_from_multimedia',
    'classify_requirements',
    'validate_cross_modal_evidence',
    'generate_requirements_report',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__institution__'
]

# Convenience functions for easy access
def get_version():
    """Return the package version."""
    return __version__

def get_author_info():
    """Return author information."""
    return {
        'author': __author__,
        'email': __email__,
        'institution': __institution__
    }

def list_available_functions():
    """List all available functions in this package."""
    functions = [name for name in __all__ if not name.startswith('__')]
    return sorted(functions)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'librosa', 'opencv-python', 'transformers',
        'spacy', 'pytesseract', 'ultralytics', 'scikit-learn',
        'matplotlib', 'seaborn', 'vaderSentiment', 'yake'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("All required dependencies are installed!")
        return True

def setup_environment():
    """Quick setup function to verify environment and download models."""
    print(f"Multimedia RE Package v{__version__}")
    print(f"Author: {__author__}")
    print("-" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Try to download models if dependencies are OK
    if deps_ok:
        try:
            import spacy
            import transformers
            
            print("Downloading required models...")
            
            # Download SpaCy model
            try:
                spacy.load("en_core_web_sm")
                print("✓ SpaCy en_core_web_sm model available")
            except OSError:
                print("⚠ SpaCy model missing. Run: python -m spacy download en_core_web_sm")
            
            # Check BERT model
            try:
                transformers.AutoModel.from_pretrained('bert-base-uncased')
                print("✓ BERT bert-base-uncased model available")
            except Exception:
                print("⚠ BERT model will be downloaded on first use")
                
        except ImportError:
            print("⚠ Could not verify models due to missing dependencies")
    
    return deps_ok

# Package-level configuration
class Config:
    """Package configuration settings."""
    
    # Default quality thresholds
    AUDIO_MIN_SNR = 60  # dB
    AUDIO_MAX_WER = 0.05  # 5% word error rate
    VIDEO_MIN_RESOLUTION = (1920, 1080)
    VIDEO_MIN_FPS = 30
    IMAGE_MIN_RESOLUTION = 2000000  # 2MP
    SYNC_ACCURACY_THRESHOLD = 0.95
    
    # Processing settings
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FRAME_EXTRACTION_FPS = 1
    TIMESTAMP_INTERVAL = 1  # seconds
    
    # Cross-modal validation confidence levels
    HIGH_CONFIDENCE_MODALITIES = 3
    MEDIUM_CONFIDENCE_MODALITIES = 2
    LOW_CONFIDENCE_MODALITIES = 1

# Make config accessible
config = Config()

# Package initialization message
def _welcome_message():
    """Display welcome message when package is imported."""
    return f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║               Multimedia Requirements Engineering            ║
    ║                          v{__version__}                               ║
    ║                                                              ║
    ║  Framework for extracting requirements from multimedia data  ║
    ║  Author: {__author__:<48} ║
    ║  Institution: {__institution__:<43} ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Quick start:
    >>> import multimedia_re_src as mmre
    >>> mmre.setup_environment()  # Check dependencies and models
    >>> mmre.list_available_functions()  # See available functions
    
    For help: help(multimedia_re_src)
    """

# Optional: Print welcome message on import (comment out if not desired)
# print(_welcome_message())

# Module documentation
__doc__ += f"""

Package Structure:
==================

multimedia_re_src/
├── multimedia_processing.py    # Core synchronization and quality control
├── audio_processing.py         # Speech-to-text, sentiment, keyword extraction  
├── video_processing.py         # User behavior and interaction analysis
├── image_processing.py         # OCR, UI detection, annotation processing
└── requirements_extraction.py  # Cross-modal validation and extraction

Quick Start:
============

    import multimedia_re_src as mmre
    
    # Check setup
    mmre.setup_environment()
    
    # Process multimedia data
    synchronized_data = mmre.synchronize_multimedia_streams(
        audio_file, video_file, image_files, session_start_time
    )
    
    # Extract requirements
    requirements = mmre.extract_requirements_from_multimedia(synchronized_data)

Configuration:
==============

Access package settings via:
    mmre.config.AUDIO_MIN_SNR
    mmre.config.VIDEO_MIN_RESOLUTION
    mmre.config.HIGH_CONFIDENCE_MODALITIES

Version: {__version__}
Author: {__author__}
"""
