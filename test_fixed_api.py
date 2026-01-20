"""
Test the fixed TFLite API
"""

import sys
import os
sys.path.append('api')

try:
    from tflite_generate import handler, generator
    
    print("🧪 Testing Fixed TFLite API...")
    
    # Test generator
    print(f"✅ Generator created")
    print(f"📊 Model loaded: {generator.model_loaded}")
    print(f"📊 Real TFLite: {getattr(generator, 'use_real_tflite', False)}")
    
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
    
    print("\n✅ Fixed TFLite API test completed successfully!")
    print("🚀 Ready for deployment!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()