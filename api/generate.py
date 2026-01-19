from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def generate():
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
                'count': len(subtitles[:count])
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def info():
    """Info about the generate endpoint"""
    return jsonify({
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
    })