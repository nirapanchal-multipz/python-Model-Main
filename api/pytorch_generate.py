from http.server import BaseHTTPRequestHandler
import json
import os

# Try to import PyTorch, fallback to rule-based if not available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class UltraFastModel(nn.Module):
    """Ultra-lightweight model for Vercel deployment"""
    
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Minimal architecture for speed
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, target_ids=None):
        # Encode
        input_embeds = self.embedding(input_ids)
        encoder_out, (hidden, cell) = self.encoder(input_embeds)
        
        if target_ids is not None:
            # Training mode
            target_embeds = self.embedding(target_ids[:, :-1])
            decoder_out, _ = self.decoder(target_embeds, (hidden, cell))
            logits = self.output_proj(decoder_out)
            return {'logits': logits}
        else:
            # Inference mode
            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)
            
            outputs = []
            current_input = self.embedding(torch.full((batch_size, 1), 2, device=input_ids.device))  # START
            decoder_hidden = (hidden, cell)
            
            for step in range(max_len - 1):
                decoder_out, decoder_hidden = self.decoder(current_input, decoder_hidden)
                logits = self.output_proj(decoder_out)
                outputs.append(logits)
                
                predicted = torch.argmax(logits, dim=-1)
                current_input = self.embedding(predicted)
            
            return {'logits': torch.cat(outputs, dim=1)}

class PyTorchSubtitleGenerator:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.max_length = 16
        
        # Load vocabulary
        self._load_vocab()
        
        # Try to load model
        if PYTORCH_AVAILABLE:
            self._load_model()
    
    def _load_vocab(self):
        """Load vocabulary from file or create default"""
        vocab_path = os.path.join(os.path.dirname(__file__), 'vocab.txt')
        
        # Default vocabulary
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        # Add common words
        common_words = [
            'reading', 'books', 'play', 'cricket', 'football', 'gym', 'workout',
            'meeting', 'study', 'session', 'shopping', 'cooking', 'dinner',
            'swimming', 'pool', 'running', 'park', 'tennis', 'match', 'dance',
            'class', 'movie', 'theater', 'today', 'tomorrow', 'pm', 'am',
            '7', '8', '9', '6', '5', '4', '3', '2', '1', 'at', 'the', 'to',
            'and', 'for', 'with', 'time', 'go', 'have', 'do', 'get'
        ]
        
        for word in common_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"📚 Vocabulary loaded: {len(self.vocab)} words")
    
    def _load_model(self):
        """Load PyTorch model if available"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'ultra_fast_subtitle_model.pth')
            
            if os.path.exists(model_path):
                # Initialize model
                self.model = UltraFastModel(
                    vocab_size=len(self.vocab),
                    embed_dim=32,
                    hidden_dim=64
                )
                
                # Load state dict
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                print(f"✅ PyTorch model loaded successfully from {model_path}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
                
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            self.model = None
    
    def _text_to_indices(self, text):
        """Convert text to token indices"""
        words = text.lower().split()[:self.max_length-2]
        indices = [self.vocab['<START>']]
        
        for word in words:
            indices.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        indices.append(self.vocab['<END>'])
        
        while len(indices) < self.max_length:
            indices.append(self.vocab['<PAD>'])
            
        return indices[:self.max_length]
    
    def _indices_to_text(self, indices):
        """Convert token indices back to text"""
        words = []
        for idx in indices:
            if idx in self.reverse_vocab:
                word = self.reverse_vocab[idx]
                if word not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def generate_with_pytorch(self, task):
        """Generate subtitle using PyTorch model"""
        if not self.model or not PYTORCH_AVAILABLE:
            return None
        
        try:
            # Prepare input
            input_indices = self._text_to_indices(task)
            input_tensor = torch.tensor([input_indices], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                output = self.model(input_tensor)
                logits = output['logits']
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                predicted_indices = predictions[0].tolist()
                
                # Convert back to text
                subtitle = self._indices_to_text(predicted_indices)
                
                # Clean up and format
                if subtitle.strip():
                    return f"🎯 {subtitle.title()}"
                else:
                    return None
                    
        except Exception as e:
            print(f"PyTorch generation error: {e}")
            return None
    
    def generate_fallback(self, task):
        """Fallback rule-based generation"""
        # Simple rule-based fallback
        task_lower = task.lower()
        
        if 'reading' in task_lower or 'books' in task_lower:
            return "📚 Literary Adventure Awaits"
        elif 'cricket' in task_lower:
            return "🏏 Cricket Championship Time"
        elif 'football' in task_lower:
            return "⚽ Field Domination Mode"
        elif 'gym' in task_lower or 'workout' in task_lower:
            return "💪 Iron Conquest Session"
        elif 'meeting' in task_lower:
            return "📊 Professional Excellence"
        elif 'study' in task_lower:
            return "📚 Academic Victory"
        else:
            return "🎯 Mission Excellence Activated"
    
    def generate_subtitle(self, task):
        """Generate subtitle with PyTorch model + fallback"""
        # Try PyTorch model first
        pytorch_result = self.generate_with_pytorch(task)
        
        if pytorch_result:
            return {
                'subtitle': pytorch_result,
                'method': 'pytorch_model',
                'model_available': True
            }
        else:
            # Fallback to rule-based
            fallback_result = self.generate_fallback(task)
            return {
                'subtitle': fallback_result,
                'method': 'rule_based_fallback',
                'model_available': PYTORCH_AVAILABLE and self.model is not None
            }

# Global generator instance
generator = PyTorchSubtitleGenerator()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "endpoint": "/api/pytorch_generate",
                "method": "POST",
                "description": "PyTorch-powered subtitle generation with rule-based fallback",
                "parameters": {
                    "task": "string (required) - Your task description",
                    "count": "integer (optional, 1-5, default: 3) - Number of subtitles"
                },
                "example": {
                    "task": "reading books 7 pm today",
                    "count": 3
                },
                "system_info": {
                    "pytorch_available": PYTORCH_AVAILABLE,
                    "model_loaded": generator.model is not None,
                    "vocab_size": len(generator.vocab) if generator.vocab else 0,
                    "fallback_available": True
                }
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'Error: {str(e)}')
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            else:
                data = {}
            
            # Validate input
            if not data or 'task' not in data:
                self.send_error_response(400, 'Missing required field: task')
                return
            
            task = str(data['task']).strip()
            count = int(data.get('count', 3))
            
            if not task:
                self.send_error_response(400, 'Task cannot be empty')
                return
            
            if count < 1 or count > 5:
                self.send_error_response(400, 'Count must be between 1 and 5')
                return
            
            # Generate subtitles
            subtitles = []
            methods_used = []
            
            for i in range(count):
                result = generator.generate_subtitle(task)
                subtitles.append(result['subtitle'])
                methods_used.append(result['method'])
            
            response = {
                "status": "success",
                "data": {
                    "original_task": task,
                    "subtitles": subtitles,
                    "count": len(subtitles),
                    "generation_info": {
                        "methods_used": methods_used,
                        "pytorch_available": PYTORCH_AVAILABLE,
                        "model_loaded": generator.model is not None,
                        "primary_method": methods_used[0] if methods_used else "unknown"
                    }
                },
                "model_info": {
                    "type": "UltraFastModel",
                    "vocab_size": len(generator.vocab),
                    "architecture": "LSTM Encoder-Decoder",
                    "size": "0.25MB",
                    "fallback": "Rule-based system"
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, f'Generation error: {str(e)}')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, status_code, message):
        try:
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": message
            }
            
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
        except Exception:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "Server error"}')