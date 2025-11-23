"""
Script to prepare model files for Vercel deployment
Copies models from Downloads folder to project models directory
"""

import os
import shutil

# Source paths
SOURCE_DIR = '/Users/jcors09/Downloads/face-detection-python-master'
SOURCE_MODEL = os.path.join(SOURCE_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
SOURCE_CONFIG = os.path.join(SOURCE_DIR, 'deploy.prototxt.txt')

# Destination paths
DEST_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEST_MODEL = os.path.join(DEST_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
DEST_CONFIG = os.path.join(DEST_DIR, 'deploy.prototxt')

def prepare_models():
    """Copy model files to models directory"""
    
    # Create models directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Copy model file
    if os.path.exists(SOURCE_MODEL):
        print(f"Copying model file...")
        shutil.copy2(SOURCE_MODEL, DEST_MODEL)
        print(f"✓ Model copied to: {DEST_MODEL}")
    else:
        print(f"✗ Model file not found: {SOURCE_MODEL}")
        return False
    
    # Copy config file
    if os.path.exists(SOURCE_CONFIG):
        print(f"Copying config file...")
        shutil.copy2(SOURCE_CONFIG, DEST_CONFIG)
        print(f"✓ Config copied to: {DEST_CONFIG}")
    else:
        print(f"✗ Config file not found: {SOURCE_CONFIG}")
        return False
    
    # Check file sizes
    model_size = os.path.getsize(DEST_MODEL) / (1024 * 1024)
    config_size = os.path.getsize(DEST_CONFIG) / 1024
    
    print(f"\nFile sizes:")
    print(f"  Model: {model_size:.2f} MB")
    print(f"  Config: {config_size:.2f} KB")
    
    if model_size > 50:
        print(f"\n⚠️  WARNING: Model file is {model_size:.2f} MB")
        print("   Vercel has deployment size limits. Consider:")
        print("   1. Using the FP16 model (smaller)")
        print("   2. Hosting models externally (S3, CDN)")
        print("   3. Using Vercel Pro for larger limits")
    
    print("\n✓ Models prepared for deployment!")
    print("\nNext steps:")
    print("1. Test locally: python app.py")
    print("2. Deploy to Vercel: vercel deploy")
    
    return True

if __name__ == "__main__":
    prepare_models()
