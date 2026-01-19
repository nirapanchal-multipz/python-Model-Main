import requests
import json
import time

class OptimizedSubtitleAPIClient:
    """Client for the Optimized AI Subtitle Generator API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def generate_subtitles(self, task, count=5, style='auto'):
        """Generate subtitles for a task with performance measurement"""
        url = f"{self.base_url}/api/generate-subtitles"
        
        payload = {
            'task': task,
            'count': count,
            'style': style
        }
        
        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            # Add client-side timing
            result['client_response_time_ms'] = (end_time - start_time) * 1000
            
            return result
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Request failed: {e}'}
    
    def analyze_task(self, task):
        """Analyze a task and extract entities with grammar correction"""
        url = f"{self.base_url}/api/analyze-task"
        
        payload = {'task': task}
        
        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            # Add client-side timing
            result['client_response_time_ms'] = (end_time - start_time) * 1000
            
            return result
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Request failed: {e}'}
    
    def get_metrics(self):
        """Get current performance metrics"""
        url = f"{self.base_url}/api/metrics"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Request failed: {e}'}
    
    def health_check(self):
        """Check if the API is running"""
        url = f"{self.base_url}/"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Request failed: {e}'}

def demo_optimized_client():
    """Demonstrate the optimized API client"""
    print("🚀 OPTIMIZED AI SUBTITLE GENERATOR - CLIENT DEMO")
    print("=" * 60)
    
    # Initialize client
    client = OptimizedSubtitleAPIClient()
    
    # Health check
    print("🔍 Checking API health...")
    health = client.health_check()
    
    if health.get('status') == 'success':
        print("✅ API is running!")
        print(f"📊 Version: {health.get('version', 'Unknown')}")
        
        # Show performance metrics if available
        if 'performance_metrics' in health:
            metrics = health['performance_metrics']
            print(f"📈 Current Metrics:")
            print(f"   Total Requests: {metrics.get('total_requests', 0)}")
            print(f"   Avg Response Time: {metrics.get('avg_response_time_ms', 0):.2f}ms")
            print(f"   Accuracy Score: {metrics.get('accuracy_score', 0):.3f}")
    else:
        print("❌ API is not available. Make sure to run 'python api_server.py' first")
        return
    
    # Test tasks with grammar errors to demonstrate correction
    test_tasks = [
        "tomorow at 7 pm i have to go gym",  # Spelling errors
        "meetng with clent at 2 PM today",   # Multiple errors
        "study for exam tomorow morning",    # Spelling error
        "urgent report due in 2 hours",      # Correct grammar
        "coffe with frend at 3 pm",         # Multiple errors
        "workout sesion at 6 am",           # Spelling error
        "famly diner at 7 pm tonite",       # Multiple errors
        "presentashun at ofice tomorow"     # Multiple errors
    ]
    
    print("\n📝 Testing subtitle generation with grammar correction...")
    print("-" * 60)
    
    total_start_time = time.time()
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🎯 Test {i}: {task}")
        print("-" * 40)
        
        # Generate subtitles
        result = client.generate_subtitles(task, count=3, style='auto')
        
        if result.get('status') == 'success':
            data = result['data']
            
            # Show grammar correction
            if data.get('corrected_task'):
                print(f"✏️  Corrected: {data['corrected_task']}")
            
            print(f"🎨 Style: {data['style']}")
            print(f"🏷️  Category: {data['entities']['category']} (Confidence: {data['entities']['confidence']:.2f})")
            
            if data['entities']['time']:
                print(f"⏰ Time: {data['entities']['time']}")
            
            if data['entities']['place']:
                print(f"📍 Place: {data['entities']['place']}")
            
            print("💡 Generated Subtitles:")
            for j, subtitle in enumerate(data['subtitles'], 1):
                print(f"   {j}. {subtitle}")
            
            # Performance metrics
            print(f"⚡ Performance:")
            print(f"   Generation Time: {data.get('generation_time_us', 0):.1f}μs")
            print(f"   Client Response: {result.get('client_response_time_ms', 0):.2f}ms")
            
            if 'performance' in result:
                perf = result['performance']
                print(f"   Server Response: {perf.get('response_time_ms', 0):.2f}ms")
        else:
            print(f"❌ Error: {result.get('message')}")
        
        print("-" * 40)
    
    total_time = time.time() - total_start_time
    
    # Get final metrics
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total Demo Time: {total_time:.2f}s")
    print(f"📈 Average per Task: {total_time/len(test_tasks):.3f}s")
    
    metrics_result = client.get_metrics()
    if metrics_result.get('status') == 'success':
        metrics = metrics_result['data']['performance_metrics']
        print(f"🎯 API Metrics:")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Avg Response Time: {metrics['avg_response_time_ms']:.2f}ms")
        print(f"   Accuracy Score: {metrics['accuracy_score']:.3f}")
        print(f"   Success Rate: {metrics['success_rate_percent']:.1f}%")
        print(f"   Microsecond Responses: {metrics['microsecond_responses']}")

def interactive_optimized_client():
    """Interactive mode for testing with performance monitoring"""
    print("\n🎮 INTERACTIVE MODE - OPTIMIZED CLIENT")
    print("=" * 60)
    
    client = OptimizedSubtitleAPIClient()
    
    # Check API availability
    health = client.health_check()
    if health.get('status') != 'success':
        print("❌ API is not available. Please start the server first.")
        return
    
    print("✅ Connected to Optimized API!")
    print("💡 Features:")
    print("   • Automatic grammar and spelling correction")
    print("   • Precise time extraction and formatting")
    print("   • Microsecond response times")
    print("   • Real-time performance metrics")
    print("📝 Type 'quit' to exit\n")
    
    while True:
        task = input("Enter your task: ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not task:
            print("⚠️  Please enter a valid task")
            continue
        
        # Ask for preferences
        try:
            count = int(input("How many subtitles? (1-10, default 5): ") or "5")
            count = max(1, min(10, count))
        except ValueError:
            count = 5
        
        style = input("Style (auto/motivational/urgent/casual/professional/creative, default auto): ").strip() or "auto"
        
        print(f"\n🔄 Generating {count} subtitles...")
        
        # Generate subtitles
        result = client.generate_subtitles(task, count, style)
        
        if result.get('status') == 'success':
            data = result['data']
            
            print(f"\n📝 Original: {data['original_task']}")
            if data.get('corrected_task'):
                print(f"✏️  Corrected: {data['corrected_task']}")
            
            print(f"🎨 Style: {data['style']}")
            print(f"🏷️  Category: {data['entities']['category']} (Confidence: {data['entities']['confidence']:.2f})")
            
            if data['entities']['time']:
                print(f"⏰ Time: {data['entities']['time']}")
            
            if data['entities']['place']:
                print(f"📍 Place: {data['entities']['place']}")
            
            print("\n💡 Generated Subtitles:")
            for i, subtitle in enumerate(data['subtitles'], 1):
                print(f"   {i}. {subtitle}")
            
            # Performance metrics
            print(f"\n⚡ Performance:")
            print(f"   Generation: {data.get('generation_time_us', 0):.1f}μs")
            print(f"   Total Response: {result.get('client_response_time_ms', 0):.2f}ms")
            
            if 'performance' in result:
                perf = result['performance']
                print(f"   Server Processing: {perf.get('response_time_ms', 0):.2f}ms")
        else:
            print(f"❌ Error: {result.get('message')}")
        
        print("\n" + "-" * 60)

if __name__ == "__main__":
    print("🚀 OPTIMIZED AI SUBTITLE GENERATOR CLIENT")
    print("Make sure the API server is running: python api_server.py")
    print()
    
    # Run demo
    demo_optimized_client()
    
    # Ask if user wants interactive mode
    if input("\nWould you like to try interactive mode? (y/n): ").lower().startswith('y'):
        interactive_optimized_client()