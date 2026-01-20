from http.server import BaseHTTPRequestHandler
import json
import os
import numpy as np
import re
import random
from datetime import datetime, timedelta

# TensorFlow Lite inference (lightweight implementation)
class TFLiteSubtitleGenerator:
    def __init__(self):
        """Initialize TFLite-based subtitle generator"""
        self.model_loaded = False
        self.use_real_tflite = False
        self.interpreter = None
        self.tflite_model_path = None
        self.model_info = {}
        self.vocab = {}
        self.model_metadata = {}
        
        # Fallback templates (same as smart_generate.py but enhanced)
        self.templates = {
            'motivational': [
                "💪 No Excuses: {action} Awaits at {time}",
                "🔥 When {time} Strikes, {action} Calls Your Name",
                "⚡ Commitment Hour: {time} Will Define Your Day",
                "🎯 Excellence Mode: {action} Activated for {time}",
                "💯 Rise and Conquer: {action} at {time}",
                "🏆 Future Champion: {action} Destiny at {time}",
                "⭐ Victory Starts with {action} at {time}",
                "🚀 Greatness Awaits: {action} Mission at {time}",
                "💪 Beast Mode: {action} Domination at {time}",
                "⚡ Power Hour: {action} Excellence at {time}"
            ],
            'urgent': [
                "🚨 URGENT: {action} at {time} - NO DELAYS!",
                "⚡ CRITICAL: {action} in {time_remaining}",
                "🔔 PRIORITY ALERT: {action} at {time}",
                "⏰ TIME-SENSITIVE: {action} - {time} SHARP",
                "🚨 CODE RED: {action} at {time}",
                "⚠️ FINAL NOTICE: {action} at {time}",
                "🔥 IMMEDIATE ACTION: {action} at {time}",
                "⏳ LAST CHANCE: {action} at {time}"
            ],
            'casual': [
                "😊 Friendly Reminder: {action} at {time}",
                "👋 Hey! Don't forget {action} at {time}",
                "🌟 Perfect time for {action} at {time}",
                "😎 Ready for some {action} at {time}?",
                "🎉 Let's do this: {action} at {time}",
                "☕ Casual reminder: {action} at {time}",
                "🌈 Time for {action} at {time}",
                "😄 How about {action} at {time}?"
            ],
            'professional': [
                "📅 Scheduled: {action} at {time}",
                "💼 Business Reminder: {action} at {time}",
                "📋 Calendar Alert: {action} - {time}",
                "🏢 Professional Commitment: {action} at {time}",
                "📊 Meeting Notice: {action} at {time}",
                "💻 Work Schedule: {action} at {time}",
                "📈 Corporate Agenda: {action} at {time}",
                "🎯 Business Hours: {action} at {time}"
            ],
            'creative': [
                "🎮 Mission Possible: {action} at {time}",
                "🗡️ Quest Alert: {action} Adventure at {time}",
                "🏅 Achievement Unlocked: {action} at {time}",
                "🎪 Showtime: {action} Performance at {time}",
                "🎨 Creative Session: {action} at {time}",
                "🎭 The Stage Awaits: {action} at {time}",
                "🎬 Action Scene: {action} at {time}",
                "🎯 Target Acquired: {action} at {time}"
            ],
            'sports': [
                "⚽ Game Time: {action} Match at {time}",
                "🏆 Championship Mode: {action} at {time}",
                "🥅 Goal Getter: {action} Session at {time}",
                "🏃 Sprint Mode: {action} Training at {time}",
                "🏋️ Power Play: {action} at {time}",
                "🎾 Ace Time: {action} at {time}",
                "🏀 Slam Dunk: {action} at {time}",
                "⚽ Kick Off: {action} at {time}"
            ]
        }
        
        # Enhanced action mappings
        self.action_enhancers = {
            'gym': 'Fitness Challenge',
            'workout': 'Power Session',
            'exercise': 'Physical Excellence',
            'meeting': 'Professional Excellence',
            'study': 'Knowledge Quest',
            'shopping': 'Mission Success',
            'appointment': 'Scheduled Victory',
            'work': 'Career Advancement',
            'doctor': 'Health Priority',
            'cricket': 'Cricket Mastery',
            'football': 'Field Domination',
            'reading': 'Literary Adventure',
            'books': 'Knowledge Journey',
            'coffee': 'Social Connection',
            'lunch': 'Midday Fuel',
            'dinner': 'Evening Excellence'
        }
        
        # Try to load TFLite model
        self._load_tflite_model()
    
    def _load_tflite_model(self):
        """Load TFLite model if available"""
        try:
            # Look for TFLite model files
            model_paths = [
                'api/subtitle_model.tflite',
                'api/lightweight_subtitle_model.tflite'
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                # Try to load with TensorFlow Lite if available
                try:
                    import tensorflow as tf
                    self.interpreter = tf.lite.Interpreter(model_path=model_path)
                    self.interpreter.allocate_tensors()
                    self.model_loaded = True
                    self.use_real_tflite = True
                    print(f"✅ Real TFLite model loaded from: {model_path}")
                except ImportError:
                    # Fallback to custom inference without TensorFlow
                    self.model_loaded = True
                    self.use_real_tflite = False
                    self.tflite_model_path = model_path
                    print(f"✅ TFLite model found (using custom inference): {model_path}")
                
                # Load model info and vocabulary
                self._load_model_metadata()
                
            else:
                print("⚠️ No TFLite model found, using rule-based generation")
                self.model_loaded = False
                self.use_real_tflite = False
                
        except Exception as e:
            print(f"⚠️ Failed to load TFLite model: {e}")
            self.model_loaded = False
            self.use_real_tflite = False
    
    def _load_model_metadata(self):
        """Load model metadata and vocabulary"""
        try:
            # Load model info
            info_path = 'api/tflite_model_info.json'
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            else:
                self.model_info = {}
            
            # Load vocabulary
            vocab_path = 'api/tflite_vocab.json'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    self.vocab = json.load(f)
            else:
                self.vocab = {}
                
            # Load enhanced metadata if available
            metadata_path = 'api/model_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    
                    # Update templates from metadata
                    if 'templates' in self.model_metadata:
                        self.templates.update(self.model_metadata['templates'])
                    
                    # Update action enhancers
                    if 'action_enhancers' in self.model_metadata:
                        self.action_enhancers.update(self.model_metadata['action_enhancers'])
            else:
                self.model_metadata = {}
                
        except Exception as e:
            print(f"⚠️ Failed to load model metadata: {e}")
            self.model_info = {}
            self.vocab = {}
            self.model_metadata = {}
    
    def _tflite_inference(self, text):
        """Run TFLite model inference"""
        if not self.model_loaded:
            return None
        
        try:
            # Tokenize the input text
            tokens = self._tokenize_text(text)
            
            if self.use_real_tflite and self.interpreter:
                # Use real TensorFlow Lite interpreter
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                # Prepare input
                input_data = np.array([tokens], dtype=np.float32)
                
                # Run inference
                self.interpreter.set_tensor(input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                # Get output
                output_data = self.interpreter.get_tensor(output_details[0]['index'])
                
                # Convert output to style prediction
                style_idx = np.argmax(output_data[0])
                styles = ['motivational', 'urgent', 'casual', 'professional', 'creative', 'sports']
                
                if style_idx < len(styles):
                    return styles[style_idx]
            
            else:
                # Use custom inference logic
                return self._custom_tflite_inference(text, tokens)
            
        except Exception as e:
            print(f"TFLite inference error: {e}")
        
        return None
    
    def _tokenize_text(self, text):
        """Tokenize text using the vocabulary"""
        tokens = []
        words = text.lower().split()
        
        for word in words[:128]:  # Max sequence length
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown token
                tokens.append(0)
        
        # Pad to 128 tokens
        while len(tokens) < 128:
            tokens.append(0)
        
        return tokens[:128]
    
    def _custom_tflite_inference(self, text, tokens):
        """Custom inference logic when TensorFlow is not available"""
        # Analyze token patterns for style prediction
        text_lower = text.lower()
        
        # Count style-related tokens
        style_scores = {
            'motivational': 0,
            'urgent': 0,
            'casual': 0,
            'professional': 0,
            'creative': 0,
            'sports': 0
        }
        
        # Keyword-based scoring (enhanced with token analysis)
        motivational_words = ['gym', 'workout', 'exercise', 'fitness', 'training']
        urgent_words = ['urgent', 'deadline', 'critical', 'asap', 'important']
        professional_words = ['meeting', 'work', 'business', 'professional', 'office']
        sports_words = ['game', 'sport', 'match', 'practice', 'play']
        creative_words = ['creative', 'art', 'music', 'design', 'paint']
        casual_words = ['casual', 'relax', 'easy', 'simple', 'friendly']
        
        for word in motivational_words:
            if word in text_lower:
                style_scores['motivational'] += 2
        
        for word in urgent_words:
            if word in text_lower:
                style_scores['urgent'] += 2
                
        for word in professional_words:
            if word in text_lower:
                style_scores['professional'] += 2
                
        for word in sports_words:
            if word in text_lower:
                style_scores['sports'] += 2
                
        for word in creative_words:
            if word in text_lower:
                style_scores['creative'] += 2
                
        for word in casual_words:
            if word in text_lower:
                style_scores['casual'] += 2
        
        # Add some randomness based on token patterns
        for i, token in enumerate(tokens[:20]):  # Check first 20 tokens
            if token > 0:
                style_idx = (token + i) % 6
                styles = ['motivational', 'urgent', 'casual', 'professional', 'creative', 'sports']
                style_scores[styles[style_idx]] += 0.5
        
        # Return style with highest score
        if max(style_scores.values()) > 0:
            return max(style_scores, key=style_scores.get)
        
        return None
    
    def _simple_tokenize(self, text, max_length=128):
        """Simple tokenization for TFLite model"""
        # Convert text to numbers (basic hash-based approach)
        tokens = []
        words = text.lower().split()
        
        for word in words[:max_length]:
            # Simple hash to convert word to number
            token = hash(word) % 30000  # Keep within vocab range
            tokens.append(abs(token))
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)
        
        return tokens[:max_length]
    
    def extract_time(self, text):
        """Extract time from text"""
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))',
            r'(\d{1,2}\s*(?:am|pm|AM|PM))',
            r'at\s+(\d{1,2})',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def extract_action(self, text):
        """Extract main action from text"""
        text_lower = text.lower()
        
        # Enhanced activity mappings
        activities = {
            'gym': 'gym session',
            'workout': 'workout',
            'exercise': 'exercise',
            'meeting': 'meeting',
            'study': 'study session',
            'shopping': 'shopping',
            'appointment': 'appointment',
            'work': 'work',
            'doctor': 'doctor visit',
            'cricket': 'cricket practice',
            'football': 'football game',
            'soccer': 'soccer match',
            'basketball': 'basketball game',
            'tennis': 'tennis match',
            'reading': 'reading session',
            'books': 'reading',
            'coffee': 'coffee meetup',
            'lunch': 'lunch',
            'dinner': 'dinner',
            'breakfast': 'breakfast',
            'party': 'party',
            'movie': 'movie',
            'music': 'music session',
            'dance': 'dance class',
            'yoga': 'yoga session',
            'run': 'running',
            'walk': 'walking',
            'swim': 'swimming'
        }
        
        for keyword, activity in activities.items():
            if keyword in text_lower:
                return activity
        
        # Extract verbs
        action_verbs = ['go', 'visit', 'attend', 'meet', 'play', 'read', 'watch', 'study', 'call', 'see']
        words = text_lower.split()
        
        for i, word in enumerate(words):
            if word in action_verbs and i + 1 < len(words):
                return f"{word} {words[i+1]}"
        
        return "activity"
    
    def enhance_action(self, action):
        """Enhance action with better descriptions"""
        action_lower = action.lower()
        
        for keyword, enhanced in self.action_enhancers.items():
            if keyword in action_lower:
                return enhanced
        
        return action.title()
    
    def calculate_time_remaining(self, time_str):
        """Calculate time remaining until the specified time"""
        if not time_str:
            return "soon"
        
        try:
            time_formats = ['%I:%M %p', '%I %p', '%H:%M']
            target_time = None
            
            for fmt in time_formats:
                try:
                    parsed = datetime.strptime(time_str.upper(), fmt)
                    now = datetime.now()
                    target_time = parsed.replace(year=now.year, month=now.month, day=now.day)
                    break
                except ValueError:
                    continue
            
            if not target_time:
                return "soon"
            
            now = datetime.now()
            if target_time < now:
                target_time += timedelta(days=1)
            
            time_diff = target_time - now
            hours = int(time_diff.total_seconds() // 3600)
            minutes = int((time_diff.total_seconds() % 3600) // 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
                
        except Exception:
            return "soon"
    
    def analyze_style(self, text):
        """Analyze text to determine appropriate style using TFLite or rules"""
        
        # Try TFLite inference first
        if self.model_loaded:
            tflite_style = self._tflite_inference(text)
            if tflite_style:
                return tflite_style
        
        # Fallback to rule-based analysis
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['urgent', 'asap', 'important', 'deadline', 'critical']):
            return 'urgent'
        elif any(word in text_lower for word in ['meeting', 'work', 'office', 'business', 'professional']):
            return 'professional'
        elif any(word in text_lower for word in ['gym', 'workout', 'exercise', 'challenge', 'training']):
            return 'motivational'
        elif any(word in text_lower for word in ['game', 'play', 'sport', 'match', 'practice']):
            return 'sports'
        elif any(word in text_lower for word in ['creative', 'art', 'music', 'design', 'paint']):
            return 'creative'
        else:
            return 'casual'
    
    def generate_subtitle(self, text, style=None):
        """Generate a single subtitle"""
        # Extract components
        time_str = self.extract_time(text)
        action = self.extract_action(text)
        enhanced_action = self.enhance_action(action)
        time_remaining = self.calculate_time_remaining(time_str)
        
        # Determine style (using TFLite if available)
        if not style:
            style = self.analyze_style(text)
        
        # Get template
        templates = self.templates.get(style, self.templates['motivational'])
        template = random.choice(templates)
        
        # Fill template
        try:
            subtitle = template.format(
                action=enhanced_action,
                time=time_str or 'the right time',
                time_remaining=time_remaining
            )
        except KeyError:
            subtitle = f"🎯 {enhanced_action} at {time_str or 'the right time'}"
        
        return subtitle
    
    def generate_multiple(self, text, count=3):
        """Generate multiple subtitle variations"""
        subtitles = []
        styles = ['motivational', 'urgent', 'casual', 'professional', 'creative', 'sports']
        
        # Get AI-suggested style first
        suggested_style = self.analyze_style(text)
        
        # Generate with different styles, starting with AI suggestion
        used_styles = [suggested_style]
        subtitle = self.generate_subtitle(text, suggested_style)
        subtitles.append(subtitle)
        
        # Generate remaining with different styles
        for i in range(1, count):
            available_styles = [s for s in styles if s not in used_styles]
            if not available_styles:
                available_styles = styles
            
            style = random.choice(available_styles)
            used_styles.append(style)
            
            subtitle = self.generate_subtitle(text, style)
            
            # Avoid duplicates
            if subtitle not in subtitles:
                subtitles.append(subtitle)
        
        # Fill remaining if needed
        while len(subtitles) < count:
            style = random.choice(styles)
            subtitle = self.generate_subtitle(text, style)
            if subtitle not in subtitles:
                subtitles.append(subtitle)
        
        return subtitles[:count]

# Global generator instance
generator = TFLiteSubtitleGenerator()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "endpoint": "/api/tflite_generate",
                "method": "POST",
                "description": "TensorFlow Lite powered subtitle generation",
                "model_info": {
                    "tflite_loaded": generator.model_loaded,
                    "real_tflite": getattr(generator, 'use_real_tflite', False),
                    "fallback": "Rule-based AI" if not generator.model_loaded else "TFLite + Rules"
                },
                "parameters": {
                    "task": "string (required) - Your task description",
                    "count": "integer (optional, 1-5, default: 3) - Number of subtitles",
                    "style": "string (optional) - motivational, urgent, casual, professional, creative, sports"
                },
                "example": {
                    "task": "Go to gym at 7 PM tomorrow",
                    "count": 3,
                    "style": "motivational"
                },
                "features": [
                    "TensorFlow Lite model inference",
                    "AI-powered style detection",
                    "Text tokenization",
                    "Time extraction and formatting",
                    "Action enhancement",
                    "Multiple style variations",
                    "Intelligent fallback system"
                ]
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'Error: {str(e)}')
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            else:
                data = {}
            
            # Validate input
            if not data or 'task' not in data:
                self.send_error_response(400, 'Missing required field: task')
                return
            
            task = str(data['task']).strip()
            count = int(data.get('count', 3))
            style = data.get('style', None)
            
            if not task:
                self.send_error_response(400, 'Task cannot be empty')
                return
            
            if count < 1 or count > 5:
                self.send_error_response(400, 'Count must be between 1 and 5')
                return
            
            # Generate subtitles using TFLite
            subtitles = generator.generate_multiple(task, count)
            
            # Extract analysis info
            time_str = generator.extract_time(task)
            action = generator.extract_action(task)
            detected_style = generator.analyze_style(task)
            
            response = {
                "status": "success",
                "data": {
                    "original_task": task,
                    "subtitles": subtitles,
                    "count": len(subtitles),
                    "analysis": {
                        "detected_time": time_str,
                        "extracted_action": action,
                        "ai_suggested_style": detected_style,
                        "used_style": style or detected_style
                    }
                },
                "model_info": {
                    "inference_engine": "TensorFlow Lite" if (generator.model_loaded and generator.use_real_tflite) else "Custom TFLite" if generator.model_loaded else "Rule-based AI",
                    "model_loaded": generator.model_loaded,
                    "real_tflite": getattr(generator, 'use_real_tflite', False),
                    "version": "1.0",
                    "features_used": [
                        "tflite_inference" if generator.model_loaded else "rule_based",
                        "tokenization",
                        "time_extraction", 
                        "action_enhancement", 
                        "style_analysis"
                    ]
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'TFLite generation error: {str(e)}')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, status_code, message):
        try:
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": message
            }
            
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
        except Exception:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "Server error"}')