"""
Vercel serverless function entry point for the AI Subtitle Generator
"""
import sys
import os

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

# Vercel expects a handler function
def handler(request, context):
    return app(request.environ, context)

# For Vercel, we need to expose the app
if __name__ == "__main__":
    app.run()