"""
Create a real TensorFlow Lite model for subtitle generation
This creates an actual .tflite file that can be used for inference
"""

import os
import json
import numpy as np

def create_real_tflite_model():
    """Create a real TensorFlow Lite model"""
    
    print("🚀 Creating Real TensorFlow Lite Model")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow found, creating real TFLite model...")
        
        # Create a simple but real neural network for subtitle style classification
        model = tf.keras.Sequential([
            # Input layer - expects tokenized text (max 128 tokens)
            tf.keras.layers.Input(shape=(128,), name='input_ids'),
            
            # Embedding layer - convert tokens to dense vectors
            tf.keras.layers.Embedding(
                input_dim=10000,  # vocab size
                output_dim=64,    # embedding dimension
                input_length=128,
                name='embedding'
            ),
            
            # Global average pooling to get fixed-size representation
            tf.keras.layers.GlobalAveragePooling1D(name='pooling'),
            
            # Dense layers for classification
            tf.keras.layers.Dense(128, activation='relu', name='dense1'),
            tf.keras.layers.Dropout(0.2, name='dropout1'),
            tf.keras.layers.Dense(64, activation='relu', name='dense2'),
            tf.keras.layers.Dropout(0.1, name='dropout2'),
            
            # Output layer - 6 classes for subtitle styles
            tf.keras.layers.Dense(6, activation='softmax', name='output')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture created")
        print(f"📊 Model summary:")
        model.summary()
        
        # Create synthetic training data to initialize weights properly
        print("🔄 Creating synthetic training data...")
        
        # Generate synthetic data that represents different subtitle styles
        np.random.seed(42)  # For reproducible results
        
        # Create training data
        X_train = []
        y_train = []
        
        # Style keywords for synthetic data generation
        style_keywords = {
            0: ['gym', 'workout', 'exercise', 'fitness', 'training'],  # motivational
            1: ['urgent', 'deadline', 'critical', 'asap', 'important'],  # urgent
            2: ['casual', 'relax', 'easy', 'simple', 'friendly'],  # casual
            3: ['meeting', 'work', 'business', 'professional', 'office'],  # professional
            4: ['creative', 'art', 'music', 'design', 'paint'],  # creative
            5: ['game', 'sport', 'match', 'practice', 'play']  # sports
        }
        
        # Generate 1000 synthetic samples
        for _ in range(1000):
            # Random style
            style = np.random.randint(0, 6)
            
            # Create a sequence with style-specific keywords
            sequence = np.random.randint(1, 1000, size=128)  # Random base tokens
            
            # Inject style-specific "keywords" (represented as specific token ranges)
            keyword_positions = np.random.choice(128, size=3, replace=False)
            for pos in keyword_positions:
                # Use different token ranges for different styles
                sequence[pos] = 1000 + style * 100 + np.random.randint(0, 50)
            
            X_train.append(sequence)
            y_train.append(style)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"✅ Created {len(X_train)} training samples")
        
        # Train the model briefly to get meaningful weights
        print("🔄 Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print("✅ Model training completed")
        
        # Convert to TensorFlow Lite
        print("🔄 Converting to TensorFlow Lite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Use float16 for smaller size
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the TFLite model
        os.makedirs("api", exist_ok=True)
        tflite_path = "api/subtitle_model.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / (1024 * 1024)  # MB
        print(f"✅ TFLite model saved to: {tflite_path}")
        print(f"📊 Model size: {model_size:.2f} MB")
        
        # Create model info
        model_info = {
            "model_type": "tensorflow_lite",
            "version": "1.0",
            "input_shape": [1, 128],
            "output_shape": [1, 6],
            "vocab_size": 10000,
            "max_sequence_length": 128,
            "num_classes": 6,
            "classes": [
                "motivational",
                "urgent", 
                "casual",
                "professional",
                "creative",
                "sports"
            ],
            "model_size_mb": model_size,
            "training_accuracy": float(history.history['accuracy'][-1]),
            "validation_accuracy": float(history.history['val_accuracy'][-1])
        }
        
        # Save model info
        info_path = "api/tflite_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model info saved to: {info_path}")
        
        # Test the model
        print("🧪 Testing TFLite model...")
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📋 Input details: {input_details[0]['shape']}")
        print(f"📋 Output details: {output_details[0]['shape']}")
        
        # Test with sample data
        test_input = np.random.randint(0, 1000, size=(1, 128)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]
        
        print(f"✅ Test inference successful!")
        print(f"📊 Predicted class: {predicted_class} ({model_info['classes'][predicted_class]})")
        print(f"📊 Confidence: {confidence:.3f}")
        
        print("\n" + "=" * 50)
        print("✅ Real TensorFlow Lite model created successfully!")
        print(f"📁 Model file: {tflite_path} ({model_size:.2f} MB)")
        print(f"📁 Model info: {info_path}")
        print("🚀 Ready for Vercel deployment!")
        print("=" * 50)
        
        return tflite_path, info_path
        
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("📦 Install with: pip install tensorflow")
        print("🔄 Creating fallback lightweight model...")
        return create_fallback_model()
    
    except Exception as e:
        print(f"❌ Error creating TFLite model: {e}")
        print("🔄 Creating fallback lightweight model...")
        return create_fallback_model()

def create_fallback_model():
    """Create a fallback model if TensorFlow is not available"""
    
    print("🔄 Creating fallback model...")
    
    # Create a simple binary file that looks like a model
    os.makedirs("api", exist_ok=True)
    
    # Create a dummy .tflite file (just for demonstration)
    dummy_model = b"TFL3\x00\x00\x00\x00" + b"\x00" * 1000  # Dummy TFLite header + data
    
    tflite_path = "api/subtitle_model_fallback.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(dummy_model)
    
    # Create model info
    model_info = {
        "model_type": "fallback",
        "version": "1.0",
        "note": "This is a fallback model. Install TensorFlow to create a real TFLite model.",
        "classes": ["motivational", "urgent", "casual", "professional", "creative", "sports"]
    }
    
    info_path = "api/tflite_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"⚠️ Fallback model created: {tflite_path}")
    print("📦 Install TensorFlow to create a real model!")
    
    return tflite_path, info_path

if __name__ == "__main__":
    create_real_tflite_model()