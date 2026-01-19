from http.server import BaseHTTPRequestHandler
import json
import random
import re
from datetime import datetime, timedelta

class SubtitleGenerator:
    def __init__(self):
        """Lightweight subtitle generator for Vercel deployment"""
        self.templates = {
            'motivational': [
                "💪 No Excuses: {action} Awaits at {time}",
                "🔥 When {time} Strikes, {action} Calls Your Name",
                "⚡ Commitment Hour: {time} Will Define Your Day",
                "🎯 Excellence Mode: {action} Activated for {time}",
                "💯 Rise and Conquer: {action} at {time}",
                "🏆 Future Champion: {action} Destiny at {time}",
                "⭐ Victory Starts with {action} at {time}",
                "🚀 Greatness Awaits: {action} Mission at {time}"
            ],
            'urgent': [
                "🚨 URGENT: {action} at {time} - NO DELAYS!",
                "⚡ CRITICAL: {action} in {time_remaining}",
                "🔔 PRIORITY ALERT: {action} at {time}",
                "⏰ TIME-SENSITIVE: {action} - {time} SHARP",
                "🚨 CODE RED: {action} at {time}",
                "⚠️ FINAL NOTICE: {action} at {time}"
            ],
            'casual': [
                "😊 Friendly Reminder: {action} at {time}",
                "👋 Hey! Don't forget {action} at {time}",
                "🌟 Perfect time for {action} at {time}",
                "😎 Ready for some {action} at {time}?",
                "🎉 Let's do this: {action} at {time}",
                "☕ Casual reminder: {action} at {time}"
            ],
            'professional': [
                "📅 Scheduled: {action} at {time}",
                "💼 Business Reminder: {action} at {time}",
                "📋 Calendar Alert: {action} - {time}",
                "🏢 Professional Commitment: {action} at {time}",
                "📊 Meeting Notice: {action} at {time}",
                "💻 Work Schedule: {action} at {time}"
            ]
        }
        
        self.action_enhancers = {
            'gym': 'Fitness Challenge',
            'workout': 'Power Session',
            'meeting': 'Professional Excellence',
            'study': 'Knowledge Quest',
            'shopping': 'Mission Success',
            'appointment': 'Scheduled Victory',
            'work': 'Career Advancement',
            'doctor': 'Health Priority',
            'exercise': 'Physical Excellence'
        }
    
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
        
        # Common activity mappings
        activities = {
            'gym': 'gym session',
            'workout': 'workout',
            'meeting': 'meeting',
            'study': 'study session',
            'shopping': 'shopping',
            'appointment': 'appointment',
            'work': 'work',
            'doctor': 'doctor visit',
            'exercise': 'exercise',
            'cricket': 'cricket practice',
            'football': 'football game',
            'reading': 'reading session',
            'books': 'reading'
        }
        
        for keyword, activity in activities.items():
            if keyword in text_lower:
                return activity
        
        # Extract verbs
        action_verbs = ['go', 'visit', 'attend', 'meet', 'play', 'read', 'watch', 'study']
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
            # Parse time
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
            
            # If time has passed today, assume tomorrow
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
        """Analyze text to determine appropriate style"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['urgent', 'asap', 'important', 'deadline']):
            return 'urgent'
        elif any(word in text_lower for word in ['meeting', 'work', 'office', 'business']):
            return 'professional'
        elif any(word in text_lower for word in ['gym', 'workout', 'exercise', 'challenge']):
            return 'motivational'
        else:
            return 'casual'
    
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
        styles = ['motivational', 'urgent', 'casual', 'professional']
        
        # Generate with different styles
        for i in range(count):
            style = styles[i % len(styles)]
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
generator = SubtitleGenerator()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "endpoint": "/api/smart_generate",
                "method": "POST",
                "description": "AI-powered subtitle generation for tasks",
                "parameters": {
                    "task": "string (required) - Your task description",
                    "count": "integer (optional, 1-5, default: 3) - Number of subtitles",
                    "style": "string (optional) - motivational, urgent, casual, professional"
                },
                "example": {
                    "task": "Go to gym at 7 PM tomorrow",
                    "count": 3,
                    "style": "motivational"
                },
                "features": [
                    "Time extraction and formatting",
                    "Action enhancement",
                    "Style-based generation",
                    "Grammar correction",
                    "Multiple variations"
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
            
            # Generate subtitles
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
                        "suggested_style": detected_style,
                        "used_style": style or detected_style
                    }
                },
                "processing_info": {
                    "model": "Rule-based AI Generator",
                    "version": "1.0",
                    "features_used": ["time_extraction", "action_enhancement", "style_analysis"]
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'Generation error: {str(e)}')
    
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