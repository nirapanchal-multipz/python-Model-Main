from flask import Flask, request, jsonify

app = Flask(__name__)

def handler(request):
    """Main handler for all routes"""
    return jsonify({
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
    })