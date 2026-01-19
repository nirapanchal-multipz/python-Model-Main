from flask import Flask, jsonify, request

app = Flask(__name__)

def handler(request, response):
    """Generate subtitles endpoint"""
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({
                'status': 'error',
                'message': 'Method not allowed'
            })
        }
    
    try:
        import json
        
        # Parse request body
        if hasattr(request, 'get_json'):
            data = request.get_json()
        else:
            data = json.loads(request.body) if request.body else {}
        
        if not data or 'task' not in data:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'status': 'error',
                    'message': 'Missing required field: task'
                })
            }
        
        task = data['task'].strip()
        count = data.get('count', 3)
        
        if not task:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'status': 'error',
                    'message': 'Task cannot be empty'
                })
            }
        
        # Simple subtitle generation
        subtitles = [
            f"📝 Task: {task}",
            f"⏰ Reminder: {task}",
            f"✅ Don't forget: {task}",
            f"🎯 Focus: {task}",
            f"💪 Action: {task}"
        ]
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'success',
                'data': {
                    'original_task': task,
                    'subtitles': subtitles[:count],
                    'count': len(subtitles[:count])
                }
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'error',
                'message': f'Internal server error: {str(e)}'
            })
        }