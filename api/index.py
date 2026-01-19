from flask import Flask, jsonify, request
import json

app = Flask(__name__)

@app.route('/')
def home():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'AI Subtitle Generator API is running!',
        'version': '1.0.0',
        'endpoints': {
            'health': '/',
            'generate': '/api/generate',
            'docs': '/api/docs'
        }
    })

@app.route('/api/generate', methods=['POST'])
def generate_subtitles():
    """Generate subtitles - simplified version for Vercel"""
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
        
        # Simple subtitle generation (without heavy ML models for Vercel)
        simple_subtitles = [
            f"📝 Task: {task}",
            f"⏰ Reminder: {task}",
            f"✅ Don't forget: {task}"
        ]
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_task': task,
                'subtitles': simple_subtitles[:count],
                'count': len(simple_subtitles[:count]),
                'note': 'Simplified version for Vercel deployment'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/docs')
def api_docs():
    """API documentation"""
    return jsonify({
        'title': 'AI Subtitle Generator API',
        'version': '1.0.0',
        'description': 'Generate subtitles for tasks',
        'endpoints': {
            'POST /api/generate': {
                'description': 'Generate subtitles for a task',
                'parameters': {
                    'task': 'string (required) - The task description',
                    'count': 'integer (optional, 1-5) - Number of subtitles (default: 3)'
                },
                'example': {
                    'task': 'Go to gym at 7 PM',
                    'count': 3
                }
            }
        }
    })

# Vercel serverless function handler
def handler(request, response):
    return app(request, response)

if __name__ == '__main__':
    app.run(debug=True)