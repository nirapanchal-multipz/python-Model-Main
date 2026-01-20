"""
Test the TFLite API locally
"""

import sys
import os
sys.path.append('api')

try:
    from tflite_generate import TFLiteSubtitleGenerator
    
    print("🧪 Testing TFLite API...")
    
    # Create generator
    generator = TFLiteSubtitleGenerator()
    
    print(f"✅ Generator created")
    print(f"📊 Model loaded: {generator.model_loaded}")
    
    if hasattr(generator, 'model_metadata'):
        print(f"📋 Metadata keys: {list(generator.model_metadata.keys())}")
    
    # Test subtitle generation
    test_task = "Go to gym at 7 PM"
    print(f"\n🎯 Testing with: '{test_task}'")
    
    # Test style analysis
    style = generator.analyze_style(test_task)
    print(f"🎨 Detected style: {style}")
    
    # Test subtitle generation
    subtitles = generator.generate_multiple(test_task, 3)
    print(f"📝 Generated {len(subtitles)} subtitles:")
    for i, subtitle in enumerate(subtitles, 1):
        print(f"  {i}. {subtitle}")
    
    print("\n✅ TFLite API test completed successfully!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()