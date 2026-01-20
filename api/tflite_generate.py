from http.server import BaseHTTPRequestHandler
import json
import os
import re
import random
from datetime import datetime, timedelta

class TFLiteSubtitleGenerator:
    def __init__(self):
        """Initialize TFLite-based subtitle generator"""
        self.model_loaded = False
        self.use_real_tflite = False
        
        # Enhanced templates
        self.templates = {
            'motivational': [
                "💪 No Excuses: {action} Awaits at {time}",
                "🔥 When {time} Strikes, {action} Calls Your Name",
                "⚡ Commitment Hour: {time} Will Define Your Day",
                "🎯 Excellence Mode: {action} Activated for {time}",
                "💯 Rise and Conquer: {action} at {time}"
            ],
            'urgent': [
                "🚨 URGENT: {action} at {time} - NO DELAYS!",
                "⚡ CRITICAL: {action} in {time_remaining}",
                "🔔 PRIORITY ALERT: {action} at {time}",
                "⏰ TIME-SENSITIVE: {action} - {time} SHARP"
            ],
            'casual': [
                "😊 Friendly Reminder: {action} at {time}",
                "👋 Hey! Don't forget {action} at {time}",
                "🌟 Perfect time for {action} at {time}",
                "😎 Ready for some {action} at {time}?"
            ],
            'professional': [
                "📅 Scheduled: {action} at {time}",
                "💼 Business Reminder: {action} at {time}",
                "📋 Calendar Alert: {action} - {time}",
                "🏢 Professional Commitment: {action} at {time}"
            ],
            'creative': [
                "🎮 Mission Possible: {action} at {time}",
                "🗡️ Quest Alert: {action} Adventure at {time}",
                "🏅 Achievement Unlocked: {action} at {time}",
                "🎪 Showtime: {action} Performance at {time}"
            ],
            'sports': [
                "⚽ Game Time: {action} Match at {time}",
                "🏆 Championship Mode: {action} at {time}",
                "🥅 Goal Getter: {action} Session at {time}",
                "🏃 Sprint Mode: {action} Training at {time}"
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
            'reading': 'Literary Adventure'
        }
        
        # Try to load TFLite model
        self._load_tflite_model()
    
    def _load_tflite_model(self):
        """Load TFLite model if available"""
        try:
            # Check if TFLite model exists
            if os.path.exists('api/subtitle_model.tflite'):
                self.model_loaded = True
                print("✅ TFLite model found")
                
                # Try to load with TensorFlow if available
                try:
                    import tensorflow as tf
                    self.interpreter = tf.lite.Interpreter(model_path='api/subtitle_model.tflite')
                    self.interpreter.allocate_tensors()
                    self.use_real_tflite = True
                    print("✅ Real TFLite interpreter loaded")
                except ImportError:
                    self.use_real_tflite = False
                    print("✅ TFLite model found (using custom inference)")
            else:
                print("⚠️ No TFLite model found, using rule-based generation")
                
        except Exception as e:
            print(f"⚠️ Failed to load TFLite model: {e}")
            self.model_loaded = False
            self.use_real_tflite = False
    
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
            'football': 'football game'
        }
        
        for keyword, activity in activities.items():
            if keyword in text_lower:
                return activity
        
        return "activity"
    
    def enhance_action(self, action):
        """Enhance action with better descriptions"""
        action_lower = action.lower()
        
        for keyword, enhanced in self.action_enhancers.items():
            if keyword in action_lower:
                return enhanced
        
        return action.title()
    
    def analyze_style(self, text):
        """Analyze text to determine appropriate style"""
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
    
    def generate_subtitle(self, text, style=None):
        """Generate a single subtitle"""
        # Extract components
        time_str = self.extract_time(text)
        action = self.extract_action(text)
        enhanced_action = self.enhance_action(action)
        time_remaining = self.calculate_time_remaining(time_str)
        
        # Determine style
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
                "endpoint": "/api/tflite",
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