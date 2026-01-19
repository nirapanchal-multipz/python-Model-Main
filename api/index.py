from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'status': 'success',
            'message': 'AI Subtitle Generator API is running!',
            'version': '1.0.0',
            'endpoints': {
                'home': '/',
                'hello': '/api/hello',
                'generate': '/api/generate (POST)'
            },
            'usage': {
                'generate_subtitles': {
                    'method': 'POST',
                    'url': '/api/generate',
                    'example_body': {
                        'task': 'Go to gym at 7 PM',
                        'count': 3
                    }
                }
            }
        }
        
        self.wfile.write(json.dumps(response).encode())