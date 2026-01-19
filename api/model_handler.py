from http.server import BaseHTTPRequestHandler
import json
import os
import base64

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Check if model files exist
            model_files = []
            model_dir = os.path.dirname(__file__)
            
            # Look for model files in the api directory
            for file in os.listdir(model_dir):
                if file.endswith('.pth') or file.endswith('.safetensors'):
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path)
                    model_files.append({
                        "name": file,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    })
            
            response = {
                "status": "success",
                "message": "Model handler endpoint",
                "available_models": model_files,
                "note": "Only small models (<50MB) can be deployed on Vercel"
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
            
            model_name = data.get('model_name', 'ultra_fast_subtitle_model.pth')
            text_input = data.get('text', '')
            
            if not text_input:
                self.send_error_response(400, 'Missing text input')
                return
            
            # Simulate model inference (replace with actual model loading)
            response = {
                "status": "success",
                "model_used": model_name,
                "input_text": text_input,
                "prediction": f"Generated subtitle for: {text_input}",
                "confidence": 0.95,
                "note": "This is a placeholder - integrate your actual model here"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'Model inference error: {str(e)}')
    
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