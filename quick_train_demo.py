#!/usr/bin/env python3
"""
Quick Training Demo - Shows accuracy metrics and performance
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Tuple, Dict
import time
import json

class SimpleSubtitleDataset(Dataset):
    """Simple dataset for quick training demo"""
    
    def __init__(self, data_pairs, max_length=32):
        self.data = []
        for task, subtitles in data_pairs:
            for subtitle in subtitles:
                self.data.append((task.lower().strip(), subtitle.strip()))
        self.max_length = max_length
        
        # Create simple vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
    def _build_vocab(self):
        """Build vocabulary from all text"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        for task, subtitle in self.data:
            for word in (task + ' ' + subtitle).split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        return vocab
    
    def _text_to_indices(self, text):
        """Convert text to indices"""
        words = text.split()[:self.max_length-2]
        indices = [self.vocab['<START>']]
        
        for word in words:
            indices.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        indices.append(self.vocab['<END>'])
        
        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.vocab['<PAD>'])
            
        return indices[:self.max_length]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task, subtitle = self.data[idx]
        
        task_indices = self._text_to_indices(task)
        subtitle_indices = self._text_to_indices(subtitle)
        
        return {
            'input_ids': torch.tensor(task_indices, dtype=torch.long),
            'target_ids': torch.tensor(subtitle_indices, dtype=torch.long)
        }

class FastSubtitleModel(nn.Module):
    """Lightweight model for quick training"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Simple architecture for quick training
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, input_ids, target_ids=None):
        batch_size, seq_len = input_ids.size()
        
        # Encode input
        input_embeds = self.embedding(input_ids)
        encoder_out, (hidden, cell) = self.encoder(input_embeds)
        
        if target_ids is not None:
            # Training mode
            target_embeds = self.embedding(target_ids[:, :-1])
            decoder_out, _ = self.decoder(target_embeds, (hidden, cell))
            logits = self.output_proj(decoder_out)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, self.vocab_size), 
                                target_ids[:, 1:].reshape(-1))
            
            return {'loss': loss, 'logits': logits}
        else:
            # Inference mode
            outputs = []
            current_input = self.embedding(torch.full((batch_size, 1), 2))  # START token
            decoder_hidden = (hidden, cell)
            
            for _ in range(seq_len):
                decoder_out, decoder_hidden = self.decoder(current_input, decoder_hidden)
                logits = self.output_proj(decoder_out)
                outputs.append(logits)
                
                # Use predicted token as next input
                predicted = torch.argmax(logits, dim=-1)
                current_input = self.embedding(predicted)
            
            return {'logits': torch.cat(outputs, dim=1)}

def create_quick_training_data():
    """Create training data for quick demo"""
    
    data = [
        ("tomorrow at 7 pm i have to go gym", [
            "💪 No Excuses: Iron Conquest Awaits at 7:00 PM",
            "🔥 When 7:00 PM Strikes, Fitness Victory Calls",
            "⚡ Commitment Hour: 7 PM Will Define Tomorrow"
        ]),
        ("meeting with client at 2 pm today", [
            "💼 Professional Excellence: Client Meeting at 2:00 PM",
            "🎯 Business Success: 2 PM Client Connection",
            "⚡ Career Moment: Client Meeting at 2:00 PM"
        ]),
        ("study for exam tomorrow morning", [
            "📚 Knowledge Quest: Tomorrow's Exam Prep",
            "🎯 Academic Excellence: Morning Exam Ready",
            "⚡ Study Victory: Tomorrow's Success"
        ]),
        ("workout session at 6 am", [
            "🌅 Champions Rise Before Dawn: 6 AM Workout",
            "💪 Early Bird Excellence: 6:00 AM Training",
            "⚡ Dawn Warrior: 6 AM Fitness Mission"
        ]),
        ("doctor appointment at 10 am", [
            "⚕️ Health Priority: 10 AM Doctor Visit",
            "🎯 Wellness Mission: 10:00 AM Appointment",
            "💯 Health Excellence: 10 AM Checkup"
        ]),
        ("grocery shopping this evening", [
            "🛒 Mission Accomplished: Evening Shopping",
            "🎯 Home Excellence: Evening Grocery Run",
            "💯 Family Care: Evening Shopping Trip"
        ])
    ]
    
    return data

def calculate_accuracy_metrics(model, dataloader, vocab, device):
    """Calculate comprehensive accuracy metrics"""
    model.eval()
    
    total_correct = 0
    total_tokens = 0
    sequence_correct = 0
    total_sequences = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Get predictions
            predicted = torch.argmax(logits, dim=-1)
            
            # Calculate token-level accuracy
            mask = target_ids[:, 1:] != 0  # Ignore padding, shift target
            correct = (predicted == target_ids[:, 1:]) & mask
            
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Calculate sequence-level accuracy
            for i in range(predicted.size(0)):
                pred_seq = predicted[i][mask[i]].cpu().numpy()
                target_seq = target_ids[i, 1:][mask[i]].cpu().numpy()
                
                if np.array_equal(pred_seq, target_seq):
                    sequence_correct += 1
                total_sequences += 1
    
    token_accuracy = total_correct / max(1, total_tokens)
    sequence_accuracy = sequence_correct / max(1, total_sequences)
    
    return {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'total_tokens': total_tokens,
        'total_sequences': total_sequences
    }

def train_quick_model(model, train_loader, val_loader, vocab, num_epochs=5, lr=1e-3):
    """Quick training with comprehensive metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training on device: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    training_history = []
    
    print(f"\n🚀 Starting quick training for {num_epochs} epochs...")
    print("="*80)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids, target_ids)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / train_batches
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids, target_ids)
                total_val_loss += outputs['loss'].item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        
        # Calculate accuracy metrics
        val_metrics = calculate_accuracy_metrics(model, val_loader, vocab, device)
        train_metrics = calculate_accuracy_metrics(model, train_loader, vocab, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch results
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {epoch+1}/{num_epochs} RESULTS | Time: {epoch_time:.2f}s")
        print(f"{'='*80}")
        print(f"📉 LOSSES:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
        print(f"📈 ACCURACY METRICS:")
        print(f"   Train Token Accuracy:    {train_metrics['token_accuracy']:.4f} ({train_metrics['token_accuracy']*100:.1f}%)")
        print(f"   Train Sequence Accuracy: {train_metrics['sequence_accuracy']:.4f} ({train_metrics['sequence_accuracy']*100:.1f}%)")
        print(f"   Val Token Accuracy:      {val_metrics['token_accuracy']:.4f} ({val_metrics['token_accuracy']*100:.1f}%)")
        print(f"   Val Sequence Accuracy:   {val_metrics['sequence_accuracy']:.4f} ({val_metrics['sequence_accuracy']*100:.1f}%)")
        print(f"🎯 PERFORMANCE:")
        print(f"   Tokens Processed: {train_metrics['total_tokens'] + val_metrics['total_tokens']:,}")
        print(f"   Sequences Processed: {train_metrics['total_sequences'] + val_metrics['total_sequences']:,}")
        
        # Save training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_token_accuracy': train_metrics['token_accuracy'],
            'train_sequence_accuracy': train_metrics['sequence_accuracy'],
            'val_token_accuracy': val_metrics['token_accuracy'],
            'val_sequence_accuracy': val_metrics['sequence_accuracy'],
            'epoch_time': epoch_time
        }
        training_history.append(epoch_data)
        
        print("="*80 + "\n")
    
    return training_history

def main():
    """Main training demo function"""
    print("🚀 QUICK TRAINING DEMO - ACCURACY METRICS SHOWCASE")
    print("="*80)
    
    # Create training data
    print("📊 Creating training dataset...")
    training_data = create_quick_training_data()
    print(f"✅ Created {len(training_data)} task examples")
    
    # Calculate total variations
    total_variations = sum(len(subtitles) for _, subtitles in training_data)
    print(f"✅ Total subtitle variations: {total_variations}")
    
    # Split data (simple split for demo)
    train_data = training_data[:4]  # First 4 for training
    val_data = training_data[4:]    # Last 2 for validation
    print(f"✅ Train: {len(train_data)} | Validation: {len(val_data)}")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = SimpleSubtitleDataset(train_data)
    val_dataset = SimpleSubtitleDataset(val_data)
    
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    
    print(f"✅ Vocabulary size: {vocab_size:,}")
    print(f"✅ Train samples: {len(train_dataset):,}")
    print(f"✅ Val samples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n🤖 Initializing lightweight model...")
    model = FastSubtitleModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    print(f"✅ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Train model
    print("\n🎯 Starting quick training demo...")
    training_history = train_quick_model(model, train_loader, val_loader, vocab, num_epochs=3)
    
    # Final evaluation
    print("\n📈 FINAL EVALUATION SUMMARY")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_metrics = calculate_accuracy_metrics(model, val_loader, vocab, device)
    
    print(f"🎯 Final Validation Metrics:")
    print(f"   Token Accuracy:    {final_metrics['token_accuracy']:.4f} ({final_metrics['token_accuracy']*100:.1f}%)")
    print(f"   Sequence Accuracy: {final_metrics['sequence_accuracy']:.4f} ({final_metrics['sequence_accuracy']*100:.1f}%)")
    print(f"   Total Tokens:      {final_metrics['total_tokens']:,}")
    print(f"   Total Sequences:   {final_metrics['total_sequences']:,}")
    
    # Show training progression
    print(f"\n📊 TRAINING PROGRESSION:")
    print("="*80)
    for i, epoch_data in enumerate(training_history):
        print(f"Epoch {epoch_data['epoch']}: "
              f"Val Accuracy {epoch_data['val_token_accuracy']*100:.1f}% | "
              f"Train Loss {epoch_data['train_loss']:.3f} | "
              f"Val Loss {epoch_data['val_loss']:.3f}")
    
    # Calculate improvement
    if len(training_history) > 1:
        initial_acc = training_history[0]['val_token_accuracy'] * 100
        final_acc = training_history[-1]['val_token_accuracy'] * 100
        improvement = final_acc - initial_acc
        print(f"\n🚀 IMPROVEMENT: {improvement:+.1f}% accuracy gain!")
    
    # Save results
    with open('quick_training_results.json', 'w') as f:
        json.dump({
            'training_history': training_history,
            'final_metrics': final_metrics,
            'model_info': {
                'total_params': total_params,
                'vocab_size': vocab_size,
                'training_samples': len(train_dataset),
                'validation_samples': len(val_dataset)
            }
        }, f, indent=2)
    
    print("\n✅ Quick training demo completed successfully!")
    print("💾 Results saved to 'quick_training_results.json'")
    print("="*80)

if __name__ == "__main__":
    main()