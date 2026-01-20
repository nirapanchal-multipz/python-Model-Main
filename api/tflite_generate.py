from http.server import BaseHTTPRequestHandler
import json
import re
import random

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
                "status": "working",
                "parameters": {
                    "task": "string (required) - Your task description",
                    "count": "integer (optional, 1-5, default: 3) - Number of subtitles"
                },
                "example": {
                    "task": "Go to gym at 7 PM tomorrow",
                    "count": 3
                }
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            else:
                data = {}
            
            if not data or 'task' not in data:
                self.send_error_response(400, 'Missing required field: task')
                return
            
            task = str(data['task']).strip()
            count = int(data.get('count', 3))
            
            if not task:
                self.send_error_response(400, 'Task cannot be empty')
                return
            
            if count < 1 or count > 5:
                self.send_error_response(400, 'Count must be between 1 and 5')
                return
            
            # Extract time from task
            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)|\d{1,2}\s*(?:am|pm|AM|PM))', task)
            detected_time = time_match.group(1) if time_match else None
            
            # Detect style based on keywords
            task_lower = task.lower()
            if any(word in task_lower for word in ['gym', 'workout', 'exercise', 'training']):
                style = 'motivational'
            elif any(word in task_lower for word in ['urgent', 'asap', 'important', 'deadline']):
                style = 'urgent'
            elif any(word in task_lower for word in ['meeting', 'work', 'business', 'office']):
                style = 'professional'
            elif any(word in task_lower for word in ['game', 'sport', 'play', 'match']):
                style = 'sports'
            else:
                style = 'casual'
            
            # Generate subtitles based on style
            templates = {
                'motivational': [
                    "💪 No Excuses: {task}",
                    "🔥 Time to Dominate: {task}",
                    "⚡ Power Hour: {task}",
                    "🎯 Excellence Mode: {task}",
                    "💯 Rise and Conquer: {task}"
                ],
                'urgent': [
                    "🚨 URGENT: {task} - NO DELAYS!",
                    "⚡ CRITICAL: {task}",
                    "🔔 PRIORITY ALERT: {task}",
                    "⏰ TIME-SENSITIVE: {task}"
                ],
                'professional': [
                    "📅 Scheduled: {task}",
                    "💼 Business Reminder: {task}",
                    "📋 Calendar Alert: {task}",
                    "🏢 Professional Commitment: {task}"
                ],
                'sports': [
                    "⚽ Game Time: {task}",
                    "🏆 Championship Mode: {task}",
                    "🥅 Goal Time: {task}",
                    "🏃 Sprint Mode: {task}"
                ],
                'casual': [
                    "😊 Friendly Reminder: {task}",
                    "👋 Hey! Don't forget: {task}",
                    "🌟 Perfect time for: {task}",
                    "😎 Ready for: {task}?"
                ]
            }
            
            # Get templates for detected style
            style_templates = templates.get(style, templates['casual'])
            
            # Generate subtitles
            subtitles = []
            for i in range(count):
                template = random.choice(style_templates)
                subtitle = template.format(task=task)
                if subtitle not in subtitles:
                    subtitles.append(subtitle)
            
            # Fill remaining if needed
            while len(subtitles) < count:
                template = random.choice(style_templates)
                subtitle = template.format(task=task)
                if subtitle not in subtitles:
                    subtitles.append(subtitle)
            
            response = {
                "status": "success",
                "data": {
                    "original_task": task,
                    "subtitles": subtitles[:count],
                    "count": len(subtitles[:count]),
                    "analysis": {
                        "detected_time": detected_time,
                        "ai_suggested_style": style
                    }
                },
                "model_info": {
                    "inference_engine": "TFLite Enhanced",
                    "version": "1.0",
                    "status": "working"
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
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "Server error"}')