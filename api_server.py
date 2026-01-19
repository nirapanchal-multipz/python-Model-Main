from flask import Flask, request, jsonify
from subtitle_generator import OptimizedSubtitleGenerator
import json
import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the optimized subtitle generator globally for better performance
try:
    generator = OptimizedSubtitleGenerator()
    logger.info("✓ Optimized Subtitle generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize subtitle generator: {e}")
    generator = None

def measure_performance(f):
    """Decorator to measure API performance"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Add performance metrics to response
        if isinstance(result, tuple) and len(result) == 2:
            response_data, status_code = result
            if isinstance(response_data, dict) and 'data' in response_data:
                response_data['performance'] = {
                    'response_time_ms': (end_time - start_time) * 1000,
                    'response_time_us': (end_time - start_time) * 1000000
                }
            return response_data, status_code
        elif isinstance(result, dict):
            result['performance'] = {
                'response_time_ms': (end_time - start_time) * 1000,
                'response_time_us': (end_time - start_time) * 1000000
            }
        
        return result
    return decorated_function

@app.route('/', methods=['GET'])
@measure_performance
def home():
    """Health check endpoint with performance metrics"""
    metrics = generator.get_performance_metrics() if generator else {}
    
    return jsonify({
        'status': 'success',
        'message': 'Optimized AI Subtitle Generator API is running!',
        'version': '2.0.0',
        'features': [
            'Grammar correction',
            'Precise time extraction',
            'Microsecond response times',
            'Performance metrics',
            'Enhanced accuracy'
        ],
        'performance_metrics': metrics,
        'endpoints': {
            'generate': '/api/generate-subtitles',
            'analyze': '/api/analyze-task',
            'metrics': '/api/metrics',
            'health': '/',
            'docs': '/api/docs'
        }
    })

@app.route('/api/generate-subtitles', methods=['POST'])
@measure_performance
def generate_subtitles():
    """Generate subtitles with optimized performance"""
    try:
        # Check if generator is available
        if generator is None:
            return jsonify({
                'status': 'error',
                'message': 'Subtitle generator not available'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'task' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: task'
            }), 400
        
        task = data['task'].strip()
        count = data.get('count', 5)  # Default to 5 subtitles
        style = data.get('style', 'auto')  # auto, motivational, urgent, casual, professional, creative
        
        if not task:
            return jsonify({
                'status': 'error',
                'message': 'Task cannot be empty'
            }), 400
        
        if count < 1 or count > 10:
            return jsonify({
                'status': 'error',
                'message': 'Count must be between 1 and 10'
            }), 400
        
        # Measure generation time
        gen_start = time.time()
        
        # Correct grammar first
        corrected_task = generator.correct_grammar(task)
        
        # Auto-detect style if needed
        if style == 'auto':
            style = generator.analyze_sentiment(corrected_task)
        
        # Generate subtitles
        logger.info(f"Generating {count} subtitles for task: {task}")
        subtitles = generator.generate_multiple_subtitles(corrected_task, count)
        
        # Extract entities for additional context
        entities = generator.extract_entities(corrected_task)
        
        gen_time = (time.time() - gen_start) * 1000000  # Convert to microseconds
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_task': task,
                'corrected_task': corrected_task if corrected_task != task else None,
                'subtitles': subtitles,
                'count': len(subtitles),
                'style': style,
                'entities': {
                    'time': entities['time'],
                    'action': entities['action'],
                    'place': entities['place'],
                    'period': entities['period'],
                    'category': entities['category'],
                    'confidence': entities['confidence']
                },
                'generation_time_us': gen_time
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating subtitles: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/analyze-task', methods=['POST'])
@measure_performance
def analyze_task():
    """Analyze a task and extract entities with grammar correction"""
    try:
        if generator is None:
            return jsonify({
                'status': 'error',
                'message': 'Subtitle generator not available'
            }), 500
        
        data = request.get_json()
        
        if not data or 'task' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: task'
            }), 400
        
        task = data['task'].strip()
        
        if not task:
            return jsonify({
                'status': 'error',
                'message': 'Task cannot be empty'
            }), 400
        
        # Correct grammar
        corrected_task = generator.correct_grammar(task)
        
        # Extract entities and analyze sentiment
        entities = generator.extract_entities(corrected_task)
        sentiment = generator.analyze_sentiment(corrected_task)
        
        return jsonify({
            'status': 'success',
            'data': {
                'original_task': task,
                'corrected_task': corrected_task if corrected_task != task else None,
                'entities': {
                    'time': entities['time'],
                    'action': entities['action'],
                    'place': entities['place'],
                    'period': entities['period'],
                    'category': entities['category'],
                    'confidence': entities['confidence'],
                    'time_info': entities['time_info']
                },
                'suggested_style': sentiment,
                'analysis': {
                    'has_time': entities['time'] is not None,
                    'has_place': entities['place'] is not None,
                    'category': entities['category'],
                    'processing_time_ms': entities.get('processing_time', 0) * 1000
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing task: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/metrics', methods=['GET'])
@measure_performance
def get_metrics():
    """Get current performance metrics"""
    try:
        if generator is None:
            return jsonify({
                'status': 'error',
                'message': 'Subtitle generator not available'
            }), 500
        
        metrics = generator.get_performance_metrics()
        
        return jsonify({
            'status': 'success',
            'data': {
                'performance_metrics': metrics,
                'system_info': {
                    'version': '2.0.0',
                    'features': [
                        'Grammar correction',
                        'Precise time extraction',
                        'Microsecond response times',
                        'Performance tracking',
                        'Enhanced accuracy'
                    ]
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/docs', methods=['GET'])
@measure_performance
def api_docs():
    """Enhanced API documentation"""
    docs = {
        'title': 'Optimized AI Subtitle Generator API',
        'version': '2.0.0',
        'description': 'Generate creative subtitles for todo list tasks with grammar correction and microsecond response times',
        'features': [
            'Automatic grammar and spelling correction',
            'Precise time extraction and formatting',
            'Microsecond response times',
            'Performance metrics tracking',
            'Enhanced accuracy with confidence scores',
            'Multiple subtitle styles',
            'Real-time sentiment analysis'
        ],
        'endpoints': {
            'POST /api/generate-subtitles': {
                'description': 'Generate subtitles for a task with grammar correction',
                'parameters': {
                    'task': 'string (required) - The task description (grammar will be auto-corrected)',
                    'count': 'integer (optional, 1-10) - Number of subtitles to generate (default: 5)',
                    'style': 'string (optional) - Style preference: auto, motivational, urgent, casual, professional, creative (default: auto)'
                },
                'example_request': {
                    'task': 'tomorow at 7 pm i have to go gym',  # Note: spelling errors will be corrected
                    'count': 5,
                    'style': 'motivational'
                },
                'example_response': {
                    'status': 'success',
                    'data': {
                        'original_task': 'tomorow at 7 pm i have to go gym',
                        'corrected_task': 'Tomorrow at 7 pm I have to go gym',
                        'subtitles': [
                            '💪 No Excuses: Iron Conquest Awaits at 7:00 PM',
                            '🔥 When 7:00 PM Strikes, Iron Conquest Calls Your Name'
                        ],
                        'count': 5,
                        'style': 'motivational',
                        'entities': {
                            'time': '7:00 PM',
                            'action': 'Iron Conquest',
                            'place': 'gym',
                            'period': 'tomorrow',
                            'category': 'gym',
                            'confidence': 0.85
                        },
                        'generation_time_us': 850.5
                    },
                    'performance': {
                        'response_time_ms': 1.2,
                        'response_time_us': 1200.0
                    }
                }
            },
            'POST /api/analyze-task': {
                'description': 'Analyze a task, correct grammar, and extract entities',
                'parameters': {
                    'task': 'string (required) - The task description'
                },
                'example_request': {
                    'task': 'meetng with clent at 2 pm today'
                },
                'example_response': {
                    'status': 'success',
                    'data': {
                        'original_task': 'meetng with clent at 2 pm today',
                        'corrected_task': 'Meeting with client at 2 pm today',
                        'entities': {
                            'time': '2:00 PM',
                            'action': 'meeting with client',
                            'place': None,
                            'period': 'today',
                            'category': 'work',
                            'confidence': 0.92,
                            'time_info': {
                                'raw_time': '2 pm',
                                'formatted_time': '2:00 PM',
                                'time_24h': '14:00',
                                'is_specific': True
                            }
                        },
                        'suggested_style': 'professional',
                        'analysis': {
                            'has_time': True,
                            'has_place': False,
                            'category': 'work',
                            'processing_time_ms': 0.85
                        }
                    }
                }
            },
            'GET /api/metrics': {
                'description': 'Get current performance metrics',
                'example_response': {
                    'status': 'success',
                    'data': {
                        'performance_metrics': {
                            'total_requests': 1250,
                            'avg_response_time_ms': 0.95,
                            'accuracy_score': 0.987,
                            'success_rate_percent': 99.2,
                            'microsecond_responses': 1240
                        },
                        'system_info': {
                            'version': '2.0.0',
                            'features': ['Grammar correction', 'Precise time extraction', 'Microsecond response times']
                        }
                    }
                }
            }
        },
        'performance_targets': {
            'response_time': '< 1ms (microsecond range)',
            'accuracy': '> 98%',
            'grammar_correction': 'Automatic',
            'time_extraction': 'Precise with 24h conversion'
        }
    }
    
    return jsonify(docs)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            '/',
            '/api/generate-subtitles',
            '/api/analyze-task',
            '/api/metrics',
            '/api/docs'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Optimized AI Subtitle Generator API Server")
    print("📖 Visit http://localhost:5000/api/docs for documentation")
    print("🔗 API endpoint: http://localhost:5000/api/generate-subtitles")
    print("📊 Performance metrics: http://localhost:5000/api/metrics")
    print("⚡ Target response time: < 1ms (microsecond range)")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)