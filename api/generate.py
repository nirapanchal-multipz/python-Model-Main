from http.server import BaseHTTPRequestHandler
import json
import urllib.parse

class handler(BaseHTTPRequestHandler):
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'endpoint': '/api/generate',
            'method': 'POST',
            'description': 'Generate subtitles for a task',
            'parameters': {
                'task': 'string (required)',
                'count': 'integer (optional, 1-5, default: 3)'
            },
            'example': {
                'task': 'Go to gym at 7 PM',
                'count': 3
            }
        }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if not data or 'task' not in data:
                self.send_error_response(400, 'Missing required field: task')
                return
            
            task = data['task'].strip()
            count = data.get('count', 3)
            
            if not task:
                self.send_error_response(400, 'Task cannot be empty')
                return
            
            if count < 1 or count > 5:
                self.send_error_response(400, 'Count must be between 1 and 5')
                return
            
            # Simple subtitle generation
            subtitles = [
                f"📝 Task: {task}",
                f"⏰ Reminder: {task}",
                f"✅ Don't forget: {task}",
                f"🎯 Focus: {task}",
                f"💪 Action: {task}"
            ]
            
            response = {
                'status': 'success',
                'data': {
                    'original_task': task,
                    'subtitles': subtitles[:count],
                    'count': len(subtitles[:count])
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error_response(500, f'Internal server error: {str(e)}')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            'status': 'error',
            'message': message
        }
        
        self.wfile.write(json.dumps(error_response).encode())