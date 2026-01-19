import json

def handler(request, response):
    """Simple hello endpoint to test Vercel deployment"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'success',
            'message': 'Hello from Vercel!',
            'timestamp': '2026-01-19',
            'endpoints': {
                'home': '/',
                'hello': '/api/hello',
                'generate': '/api/generate'
            }
        })
    }