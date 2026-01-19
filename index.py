from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    """Main landing page"""
    return jsonify({
        'status': 'success',
        'message': 'AI Subtitle Generator API',
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
                'body': {
                    'task': 'Your task description',
                    'count': 3
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True)