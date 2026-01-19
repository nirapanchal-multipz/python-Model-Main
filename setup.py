#!/usr/bin/env python3
"""
Setup script for Optimized AI Subtitle Generator
Handles installation, model downloads, and testing
"""

import subprocess
import sys
import os
import time

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model"""
    print("🧠 Installing spaCy English model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  spaCy model installation failed: {e}")
        print("   You can install it manually with: python -m spacy download en_core_web_sm")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ['models', 'data', 'logs', 'checkpoints']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Created {directory}/")
        else:
            print(f"  📁 {directory}/ already exists")

def test_optimized_installation():
    """Test if the optimized version is working"""
    print("🧪 Testing optimized installation...")
    
    try:
        from subtitle_generator import OptimizedSubtitleGenerator
        
        # Initialize generator
        print("  � Initializing optimized generator...")
        start_time = time.time()
        generator = OptimizedSubtitleGenerator()
        init_time = time.time() - start_time
        
        # Test subtitle generation with grammar correction
        test_tasks = [
            "tomorow at 7 pm i have to go gym",  # With spelling errors
            "meeting with client at 2 PM today"   # Correct grammar
        ]
        
        print("  ✅ Generator initialized successfully!")
        print(f"  ⚡ Initialization time: {init_time:.3f}s")
        
        for i, test_task in enumerate(test_tasks, 1):
            print(f"\n  📝 Test {i}: {test_task}")
            
            # Test grammar correction
            corrected = generator.correct_grammar(test_task)
            if corrected != test_task:
                print(f"  ✏️  Corrected: {corrected}")
            
            # Test subtitle generation
            start_gen = time.time()
            subtitles = generator.generate_multiple_subtitles(test_task, 3)
            gen_time = (time.time() - start_gen) * 1000  # Convert to ms
            
            print(f"  � Generated subtitles ({gen_time:.2f}ms):")
            for j, subtitle in enumerate(subtitles, 1):
                print(f"     {j}. {subtitle}")
        
        # Test performance metrics
        metrics = generator.get_performance_metrics()
        print(f"\n  📊 Performance Metrics:")
        print(f"     Total Requests: {metrics['total_requests']}")
        print(f"     Avg Response Time: {metrics['avg_response_time_ms']:.2f}ms")
        print(f"     Accuracy Score: {metrics['accuracy_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print(f"   Error details: {str(e)}")
        return False

def test_api_server():
    """Test if API server can start"""
    print("🌐 Testing API server startup...")
    
    try:
        # Import to check if all dependencies are available
        from api_server import app, generator
        
        if generator is None:
            print("  ⚠️  Generator not initialized in API server")
            return False
        
        print("  ✅ API server can start successfully!")
        print("  💡 To start the server: python api_server.py")
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n📖 USAGE INSTRUCTIONS")
    print("=" * 50)
    print("🚀 Quick Start:")
    print("  1. Start API server:    python api_server.py")
    print("  2. Test with client:    python client_example.py")
    print("  3. Use directly:        python subtitle_generator.py")
    print("  4. Train model:         python optimized_train_model.py")
    
    print("\n🎯 Features:")
    print("  • Automatic grammar and spelling correction")
    print("  • Precise time extraction and formatting")
    print("  • Microsecond response times")
    print("  • Real-time performance metrics")
    print("  • Multiple subtitle styles")
    print("  • Enhanced accuracy with confidence scores")
    
    print("\n🔗 API Endpoints:")
    print("  • Health check:         GET  /")
    print("  • Generate subtitles:   POST /api/generate-subtitles")
    print("  • Analyze task:         POST /api/analyze-task")
    print("  • Performance metrics:  GET  /api/metrics")
    print("  • Documentation:        GET  /api/docs")
    
    print("\n📊 Performance Targets:")
    print("  • Response time: < 1ms (microsecond range)")
    print("  • Accuracy: > 98%")
    print("  • Grammar correction: Automatic")
    print("  • Time extraction: Precise with 24h conversion")

def main():
    """Main setup function"""
    print("🚀 OPTIMIZED AI SUBTITLE GENERATOR SETUP")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version.split()[0]}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection.")
        return False
    
    # Install spaCy model
    install_spacy_model()  # Non-critical, continue even if it fails
    
    # Test optimized installation
    if not test_optimized_installation():
        print("❌ Optimized generator test failed")
        return False
    
    # Test API server
    if not test_api_server():
        print("⚠️  API server test failed, but core functionality works")
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Show usage instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✨ Ready to generate amazing subtitles! ✨")
    else:
        print(f"\n❌ Setup failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1)