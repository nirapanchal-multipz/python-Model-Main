from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    """Simple hello endpoint to test Vercel deployment"""
    return jsonify({
        'status': 'success',
        'message': 'Hello from Vercel!',
        'timestamp': '2026-01-19',
        'endpoints': {
            'home': '/',
            'hello': '/api/hello',
            'generate': '/api/generate'
        }
    })