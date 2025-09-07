#!/usr/bin/env python3
"""
Verify that all required packages and models are properly installed
"""

def verify_packages():
    """Verify all required packages can be imported"""
    required_packages = [
        'transformers', 'librosa', 'cv2', 'pytesseract', 'spacy',
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'ultralytics', 'speech_recognition', 'vaderSentiment', 'yake'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")

def verify_models():
    """Verify ML models are available"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ SpaCy en_core_web_sm model loaded")
    except Exception as e:
        print(f"✗ SpaCy model: {e}")
    
    try:
        import transformers
        model = transformers.AutoModel.from_pretrained('bert-base-uncased')
        print("✓ BERT bert-base-uncased model loaded")
    except Exception as e:
        print(f"✗ BERT model: {e}")

if __name__ == "__main__":
    print("Verifying package installation...")
    verify_packages()
    print("\nVerifying ML models...")
    verify_models()
    print("\nSetup verification complete!")
