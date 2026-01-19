from flask import Flask, jsonify

app = Flask(__name__)

def handler(request):
    """Simple hello endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Hello from Vercel Python API!',
        'timestamp': '2026-01-19',
        'note': 'This endpoint is working correctly'
    })