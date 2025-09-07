#!/usr/bin/env python3
"""
Download required ML models for multimedia processing
"""

def download_spacy_model():
    """Download SpaCy English model"""
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
        print("✓ SpaCy en_core_web_sm model downloaded")
    except Exception as e:
        print(f"✗ Failed to download SpaCy model: {e}")

def download_transformers_models():
    """Download BERT model"""
    try:
        import transformers
        transformers.AutoModel.from_pretrained('bert-base-uncased')
        print("✓ BERT bert-base-uncased model downloaded")
    except Exception as e:
        print(f"✗ Failed to download BERT model: {e}")

if __name__ == "__main__":
    print("Downloading required ML models...")
    download_spacy_model()
    download_transformers_models()
    print("Model download complete!")
