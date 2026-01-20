#!/usr/bin/env python3
"""
Copy the small PyTorch model to the api directory
"""
import shutil
import os

def copy_model():
    source = "ultra_fast_subtitle_model.pth"
    destination = "api/ultra_fast_subtitle_model.pth"
    
    try:
        if os.path.exists(source):
            shutil.copy2(source, destination)
            print(f"✅ Model copied from {source} to {destination}")
            
            # Check file size
            size = os.path.getsize(destination)
            size_mb = size / (1024 * 1024)
            print(f"📊 Model size: {size_mb:.2f} MB")
            
            if size_mb > 50:
                print("⚠️ Warning: Model is larger than Vercel's 50MB limit")
            else:
                print("✅ Model size is compatible with Vercel")
                
        else:
            print(f"❌ Source model file not found: {source}")
            
    except Exception as e:
        print(f"❌ Error copying model: {e}")

if __name__ == "__main__":
    copy_model()