from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Handle all routes"""
    if request.method == 'POST' and path == 'generate':
        return generate_subtitles()
    elif path == 'generate' and request.method == 'GET':
        return jsonify({
            'status': 'error',
            'message': 'Use POST method to generate subtitles'
        }), 405
    elif path == 'hello':
        return hello()
    else:
        return home()

def home():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'AI Subtitle Generator API is running!',
        'version': '1.0.0',
        'endpoints': {
            'health': '/',
            'generate': '/api/generate',
            'hello': '/api/hello'
        }
    })

def hello():
    """Simple hello endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Hello from Vercel!',
        'timestamp': '2026-01-19',
        'note': 'API is working correctly'
    })

def generate_subtitles():
    """Generate subtitles endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'task' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: task'
            }), 400
        
        task = data['task'].strip()
        count = data.get('count', 3)
        
        if not task:
            return jsonify({
                'status': 'error',
                'message': 'Task cannot be empty'
            }), 400
        
        if count < 1 or count > 5:
            return jsonify({
                'status': 'error',
                'message': 'Count must be between 1 and 5'
            }), 400
        
        # Simple subtitle generation
        subtitles = [
            f"📝 Task: {task}",
            f"⏰ Reminder: {task}",
            f"✅ Don't forget: {task}",
            f"🎯 Focus: {task}",
            f"💪 Action: {task}"
        ]
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_task': task,
                'subtitles': subtitles[:count],
                'count': len(subtitles[:count]),
                'note': 'Simplified version for Vercel deployment'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500