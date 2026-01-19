#!/usr/bin/env python3
"""
Ultra Fast Training - Complete in under 2 minutes
Optimized for speed while maintaining quality
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
import json
from datetime import datetime

class FastDataset(Dataset):
    """Ultra-fast dataset for quick training"""
    
    def __init__(self, data_pairs, max_length=16, vocab_file='vocab.txt'):  # Reduced length for speed
        self.data = []
        for task, subtitles in data_pairs:
            # Take only first 2 subtitles per task for speed
            for subtitle in subtitles[:2]:
                self.data.append((task.lower().strip(), subtitle.strip()))
        
        self.max_length = max_length
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        
    def _load_vocab(self, vocab_file):
        """Load vocabulary from vocab.txt file"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and word not in vocab:
                        vocab[word] = len(vocab)
            
            print(f"📚 Loaded vocabulary from {vocab_file}: {len(vocab)} words")
            
        except FileNotFoundError:
            print(f"⚠️  Vocab file {vocab_file} not found, building from training data...")
            # Fallback to building vocab from data
            for task, subtitle in self.data:
                for word in (task + ' ' + subtitle).split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
        
        return vocab
    
    def _text_to_indices(self, text):
        words = text.split()[:self.max_length-2]
        indices = [self.vocab['<START>']]
        
        for word in words:
            word_idx = self.vocab.get(word, self.vocab['<UNK>'])
            # Ensure index is within vocab range
            if word_idx < len(self.vocab):
                indices.append(word_idx)
            else:
                indices.append(self.vocab['<UNK>'])
        
        indices.append(self.vocab['<END>'])
        
        while len(indices) < self.max_length:
            indices.append(self.vocab['<PAD>'])
            
        return indices[:self.max_length]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task, subtitle = self.data[idx]
        
        return {
            'input_ids': torch.tensor(self._text_to_indices(task), dtype=torch.long),
            'target_ids': torch.tensor(self._text_to_indices(subtitle), dtype=torch.long)
        }

class UltraFastModel(nn.Module):
    """Ultra-lightweight model for speed"""
    
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):  # Much smaller
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Minimal architecture for speed
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, input_ids, target_ids=None):
        # Encode
        input_embeds = self.embedding(input_ids)
        encoder_out, (hidden, cell) = self.encoder(input_embeds)
        
        if target_ids is not None:
            # Training
            target_embeds = self.embedding(target_ids[:, :-1])
            decoder_out, _ = self.decoder(target_embeds, (hidden, cell))
            logits = self.output_proj(decoder_out)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), 
                                target_ids[:, 1:].reshape(-1))
            
            return {'loss': loss, 'logits': logits}
        else:
            # Inference mode - FIXED to match training dimensions
            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)
            
            outputs = []
            current_input = self.embedding(torch.full((batch_size, 1), 2, device=input_ids.device))  # START
            decoder_hidden = (hidden, cell)
            
            for step in range(max_len - 1):  # Generate max_len-1 tokens to match target
                decoder_out, decoder_hidden = self.decoder(current_input, decoder_hidden)
                logits = self.output_proj(decoder_out)
                outputs.append(logits)
                
                predicted = torch.argmax(logits, dim=-1)
                current_input = self.embedding(predicted)
            
            return {'logits': torch.cat(outputs, dim=1)}

def create_fast_training_data():
    """Enhanced training data with more examples for better accuracy"""
    
    return [
        # Original data
        ("reading books 7 pm today", [
            "📚 Literary Adventure at 7:00 PM",
            "🎯 Knowledge Quest: Reading Session at 7 PM",
            "📖 Mind Journey: Book Time at 7:00 PM"
        ]),
        ("play cricket today 9 pm", [
            "🏏 Cricket Championship at 9:00 PM", 
            "⚽ Athletic Excellence: Cricket Match at 9 PM",
            "🏆 Sports Victory: Cricket Battle at 9:00 PM"
        ]),
        ("play football 6 pm", [
            "⚽ Field Domination at 6:00 PM",
            "🏆 Football Glory: Match Time 6 PM",
            "🥅 Goal Crusher: Football Action at 6:00 PM"
        ]),
        ("gym workout 8 am", [
            "💪 Iron Conquest at 8:00 AM",
            "🔥 Fitness Domination: Morning Power 8 AM",
            "🏋️ Strength Mode: Gym Session at 8:00 AM"
        ]),
        ("meeting at 2 pm", [
            "📊 Professional Excellence at 2:00 PM",
            "📊 Business Victory: Meeting Time 2 PM",
            "🏢 Corporate Success: Conference at 2:00 PM"
        ]),
        ("study session 4 pm", [
            "📚 Academic Victory at 4:00 PM",
            "🎯 Learning Adventure: Study Time 4 PM",
            "🧠 Brain Power: Knowledge Session at 4:00 PM"
        ]),
        ("shopping mall 3 pm", [
            "🛒 Shopping Success at 3:00 PM",
            "🛍️ Retail Therapy: Mall Mission 3 PM",
            "💳 Purchase Power: Shopping Spree at 3:00 PM"
        ]),
        ("cooking dinner 7 pm", [
            "🍳 Culinary Creation at 7:00 PM",
            "👨‍🍳 Kitchen Mastery: Dinner Prep 7 PM",
            "🔥 Chef Mode: Cooking Excellence at 7:00 PM"
        ]),
        ("swimming pool 5 pm", [
            "🏊 Aquatic Excellence at 5:00 PM",
            "💧 Pool Mastery: Swimming Session 5 PM",
            "🌊 Water Warrior: Pool Time at 5:00 PM"
        ]),
        ("running park 6 am", [
            "🏃 Speed Demon at 6:00 AM",
            "⚡ Running Victory: Morning Sprint 6 AM",
            "🌅 Dawn Runner: Park Session at 6:00 AM"
        ]),
        ("tennis match 4 pm", [
            "🎾 Ace Champion at 4:00 PM",
            "🏆 Tennis Triumph: Match Point 4 PM",
            "🎯 Court Master: Tennis Battle at 4:00 PM"
        ]),
        ("dance class 8 pm", [
            "💃 Rhythm Master at 8:00 PM",
            "🎵 Dance Excellence: Movement Magic 8 PM",
            "✨ Groove Time: Dance Session at 8:00 PM"
        ]),
        ("movie theater 9 pm", [
            "🎬 Cinema Adventure at 9:00 PM",
            "🍿 Movie Magic: Theater Experience 9 PM",
            "🎭 Entertainment Mode: Film Time at 9:00 PM"
        ]),
        ("doctor appointment 10 am", [
            "⚕️ Health Priority at 10:00 AM",
            "🏥 Wellness Journey: Medical Visit 10 AM",
            "💊 Health Excellence: Doctor Time at 10:00 AM"
        ]),
        ("coffee shop 11 am", [
            "☕ Caffeine Mission at 11:00 AM",
            "🌟 Coffee Excellence: Brew Time 11 AM",
            "☕ Energy Boost: Coffee Session at 11:00 AM"
        ]),
        ("library study 2 pm", [
            "📚 Knowledge Hub at 2:00 PM",
            "🤫 Silent Study: Library Focus 2 PM",
            "📖 Academic Zone: Library Session at 2:00 PM"
        ]),
        ("meeting with client at 2 pm", [
            "Professional Excellence: Client Connection Time",
            "Where Deals Are Made: 2 PM Power Hour",
            "Showtime: Your Client Awaits Your Brilliance",
            "Client Chemistry: Make Every Second Count",
            "The Meeting That Could Change Everything"
        ]),
        ("team meeting at 3 pm tomorrow", [
            "Collaboration Station: Team Power Hour",
            "Tomorrow's Synergy: United We Achieve",
            "3 PM: Where Great Minds Align",
            "Team Spirit Activation: Excellence Together",
            "The Huddle That Fuels Victory"
        ]),
        ("conference call at 9 am", [
            "Digital Boardroom: Your Voice Matters",
            "9 AM Sharp: Professional Communication Mode",
            "Virtual Excellence: Connect and Conquer",
            "Morning Call to Greatness",
            "When Distance Means Nothing: Remote Power"
        ]),
        ("prep meeting agenda today", [
            "Strategic Planning: Agenda Mastery Mode",
            "Organization Wins: Blueprint Your Meeting",
            "The Prep That Separates Good from Great",
            "Agenda Architect: Build Success Framework",
            "Preparation Is Victory: Meeting Ready"
        ]),
        ("client demo at 5 pm", [
            "Spotlight Ready: Your Moment to Shine",
            "Demo Day: Where Features Become Dreams",
            "5 PM Performance: Blow Their Minds",
            "Product Pride: Show What You've Built",
            "The Presentation That Seals the Deal"
        ]),
        
        # NEW DATA - Fitness & Sports
        ("yoga class 7 am", [
            "🧘 Morning Zen at 7:00 AM",
            "☀️ Mindful Movement: Yoga Flow 7 AM",
            "🕉️ Balance Master: Morning Stretch at 7:00 AM"
        ]),
        ("basketball practice 5 pm", [
            "🏀 Hoop Dreams at 5:00 PM",
            "🔥 Court Commander: Basketball Drill 5 PM",
            "⛹️ Slam Dunk Session at 5:00 PM"
        ]),
        ("cycling route 6 am", [
            "🚴 Pedal Power at 6:00 AM",
            "🌄 Morning Ride: Cycling Adventure 6 AM",
            "⚡ Two-Wheel Thunder at 6:00 AM"
        ]),
        ("boxing training 7 pm", [
            "🥊 Fight Mode at 7:00 PM",
            "💥 Ring Warrior: Boxing Power 7 PM",
            "🔥 Knockout Training at 7:00 PM"
        ]),
        ("pilates session 9 am", [
            "🤸 Core Strength at 9:00 AM",
            "💪 Pilates Precision: Control Session 9 AM",
            "✨ Flexibility Flow at 9:00 AM"
        ]),
        
        # Work & Professional
        ("quarterly review 1 pm", [
            "📊 Performance Showcase at 1:00 PM",
            "💼 Quarterly Excellence: Review Time 1 PM",
            "🎯 Progress Report: Achievement Hour at 1:00 PM"
        ]),
        ("brainstorming session 10 am", [
            "💡 Innovation Lab at 10:00 AM",
            "🚀 Creative Explosion: Ideation 10 AM",
            "🧠 Think Tank: Brainstorm Power at 10:00 AM"
        ]),
        ("job interview 11 am", [
            "🎯 Career Opportunity at 11:00 AM",
            "⭐ Interview Excellence: Your Moment 11 AM",
            "💼 Professional Breakthrough at 11:00 AM"
        ]),
        ("workshop training 2 pm", [
            "📚 Skill Building at 2:00 PM",
            "🎓 Professional Growth: Workshop 2 PM",
            "🔧 Expertise Upgrade at 2:00 PM"
        ]),
        ("project deadline today", [
            "⏰ Finish Line Focus: Deadline Day",
            "🎯 Final Push: Project Completion Mode",
            "🏁 Victory Lap: Deliver Excellence Today"
        ]),
        
        # Social & Entertainment
        ("birthday party 6 pm", [
            "🎂 Celebration Time at 6:00 PM",
            "🎉 Birthday Bash: Party Mode 6 PM",
            "🎈 Joy Fest: Celebration Hour at 6:00 PM"
        ]),
        ("dinner date 8 pm", [
            "🍽️ Romantic Evening at 8:00 PM",
            "❤️ Date Night: Special Moments 8 PM",
            "🌹 Connection Time: Dinner Date at 8:00 PM"
        ]),
        ("game night 9 pm", [
            "🎮 Epic Gaming at 9:00 PM",
            "🕹️ Victory Quest: Game Night 9 PM",
            "🏆 Championship Hour at 9:00 PM"
        ]),
        ("concert tonight 7 pm", [
            "🎸 Music Magic at 7:00 PM",
            "🎵 Live Performance: Concert Time 7 PM",
            "🎤 Sound Wave: Music Night at 7:00 PM"
        ]),
        ("picnic park 12 pm", [
            "🧺 Outdoor Feast at 12:00 PM",
            "🌳 Nature Break: Picnic Time 12 PM",
            "☀️ Sunshine Gathering at 12:00 PM"
        ]),
        
        # Education & Learning
        ("online course 3 pm", [
            "💻 Digital Learning at 3:00 PM",
            "📱 Course Progress: Study Time 3 PM",
            "🎓 Knowledge Upgrade at 3:00 PM"
        ]),
        ("exam preparation 5 pm", [
            "📝 Test Ready at 5:00 PM",
            "🎯 Exam Excellence: Prep Session 5 PM",
            "🏆 Success Mode: Study Power at 5:00 PM"
        ]),
        ("language class 6 pm", [
            "🗣️ Linguistic Journey at 6:00 PM",
            "🌍 Language Mastery: Learning Hour 6 PM",
            "📚 Communication Skills at 6:00 PM"
        ]),
        ("tutoring session 4 pm", [
            "👨‍🏫 Learning Boost at 4:00 PM",
            "📖 Academic Support: Tutoring 4 PM",
            "🧠 Knowledge Transfer at 4:00 PM"
        ]),
        ("research work 1 pm", [
            "🔬 Discovery Mode at 1:00 PM",
            "📊 Research Excellence: Investigation 1 PM",
            "🎓 Scholar Hour at 1:00 PM"
        ]),
        
        # Health & Wellness
        ("meditation 6 am", [
            "🧘‍♂️ Inner Peace at 6:00 AM",
            "☮️ Mindfulness: Morning Calm 6 AM",
            "🌅 Zen Mode: Meditation at 6:00 AM"
        ]),
        ("therapy session 3 pm", [
            "💭 Mental Wellness at 3:00 PM",
            "🌱 Growth Journey: Therapy Time 3 PM",
            "💚 Self-Care Hour at 3:00 PM"
        ]),
        ("dentist appointment 2 pm", [
            "🦷 Smile Care at 2:00 PM",
            "😁 Dental Health: Checkup Time 2 PM",
            "✨ Bright Smile Session at 2:00 PM"
        ]),
        ("spa treatment 4 pm", [
            "💆 Relaxation Time at 4:00 PM",
            "🌺 Spa Bliss: Pamper Session 4 PM",
            "✨ Rejuvenation Hour at 4:00 PM"
        ]),
        ("nutrition consult 11 am", [
            "🥗 Wellness Planning at 11:00 AM",
            "🍎 Nutrition Guide: Health Talk 11 AM",
            "💚 Fuel Your Body at 11:00 AM"
        ]),
        
        # Daily Tasks & Errands
        ("grocery shopping 5 pm", [
            "🛒 Fresh Finds at 5:00 PM",
            "🥬 Market Mission: Shopping Time 5 PM",
            "🍎 Pantry Power at 5:00 PM"
        ]),
        ("laundry today", [
            "👕 Fresh Clothes Mission Today",
            "🧺 Laundry Victory: Clean Mode Activated",
            "✨ Wardrobe Refresh: Wash Day"
        ]),
        ("car service 10 am", [
            "🚗 Vehicle Care at 10:00 AM",
            "🔧 Auto Maintenance: Service Time 10 AM",
            "⚙️ Road Ready at 10:00 AM"
        ]),
        ("bank visit 1 pm", [
            "🏦 Financial Business at 1:00 PM",
            "💰 Money Matters: Bank Time 1 PM",
            "💳 Finance Hour at 1:00 PM"
        ]),
        ("post office 3 pm", [
            "📮 Mailing Mission at 3:00 PM",
            "✉️ Package Power: Post Time 3 PM",
            "📦 Delivery Prep at 3:00 PM"
        ]),
        
        # Creative & Hobbies
        ("painting session 2 pm", [
            "🎨 Artistic Flow at 2:00 PM",
            "🖌️ Creative Canvas: Paint Time 2 PM",
            "🌈 Color Magic at 2:00 PM"
        ]),
        ("photography walk 5 pm", [
            "📸 Capture Moments at 5:00 PM",
            "🌆 Photo Adventure: Golden Hour 5 PM",
            "📷 Visual Journey at 5:00 PM"
        ]),
        ("guitar practice 7 pm", [
            "🎸 String Mastery at 7:00 PM",
            "🎵 Music Practice: Guitar Session 7 PM",
            "🎶 Melody Maker at 7:00 PM"
        ]),
        ("writing time 9 pm", [
            "✍️ Creative Words at 9:00 PM",
            "📝 Author Mode: Writing Hour 9 PM",
            "📖 Story Craft at 9:00 PM"
        ]),
        ("gardening 8 am", [
            "🌻 Green Thumb at 8:00 AM",
            "🌱 Garden Glory: Plant Time 8 AM",
            "🌿 Nature Nurture at 8:00 AM"
        ]),
        
        # Family & Home
        ("family dinner 7 pm", [
            "👨‍👩‍👧‍👦 Together Time at 7:00 PM",
            "🍽️ Family Feast: Bonding Hour 7 PM",
            "❤️ Home Gathering at 7:00 PM"
        ]),
        ("kids pickup 3 pm", [
            "🚸 Parent Duty at 3:00 PM",
            "👶 Family First: Pickup Time 3 PM",
            "🏫 School Run at 3:00 PM"
        ]),
        ("house cleaning 10 am", [
            "🧹 Home Refresh at 10:00 AM",
            "✨ Clean Sweep: Tidy Time 10 AM",
            "🏠 Space Revival at 10:00 AM"
        ]),
        ("pet vet 2 pm", [
            "🐕 Pet Care at 2:00 PM",
            "🐾 Furry Friend Health: Vet Visit 2 PM",
            "❤️ Animal Wellness at 2:00 PM"
        ]),
        ("home repair 11 am", [
            "🔨 Fix-It Time at 11:00 AM",
            "🏠 Home Improvement: Repair Hour 11 AM",
            "🔧 DIY Mode at 11:00 AM"
        ]),
        
        # Travel & Transportation
        ("flight departure 6 am", [
            "✈️ Journey Begins at 6:00 AM",
            "🌍 Travel Adventure: Takeoff 6 AM",
            "🧳 Sky Bound at 6:00 AM"
        ]),
        ("train commute 8 am", [
            "🚄 Morning Transit at 8:00 AM",
            "🚉 Commute Time: Train Ride 8 AM",
            "🎫 Rail Journey at 8:00 AM"
        ]),
        ("airport pickup 9 pm", [
            "🛬 Welcome Back at 9:00 PM",
            "🚗 Airport Run: Pickup Mission 9 PM",
            "👋 Arrival Time at 9:00 PM"
        ]),
        ("road trip 5 am", [
            "🚗 Adventure Awaits at 5:00 AM",
            "🛣️ Road Warrior: Trip Start 5 AM",
            "🗺️ Journey Quest at 5:00 AM"
        ]),
        ("taxi booking 7 pm", [
            "🚕 Ride Ready at 7:00 PM",
            "📱 Transport Sorted: Taxi Time 7 PM",
            "🚖 On The Move at 7:00 PM"
        ]),
        
        # Technology & Digital
        ("webinar 4 pm", [
            "💻 Virtual Learning at 4:00 PM",
            "🌐 Digital Session: Webinar Time 4 PM",
            "📡 Online Event at 4:00 PM"
        ]),
        ("podcast recording 6 pm", [
            "🎙️ Audio Magic at 6:00 PM",
            "🎧 Podcast Power: Recording Session 6 PM",
            "📻 Voice Time at 6:00 PM"
        ]),
        ("software update today", [
            "⚙️ System Upgrade: Update Mission Today",
            "💾 Tech Refresh: Software Patch Time",
            "🔄 Digital Renewal: Update Ready"
        ]),
        ("video call 3 pm", [
            "📹 Face Time at 3:00 PM",
            "💬 Virtual Connect: Video Chat 3 PM",
            "👥 Screen Meeting at 3:00 PM"
        ]),
        ("backup data 11 pm", [
            "💾 Data Guardian at 11:00 PM",
            "🔒 Security Mode: Backup Time 11 PM",
            "📊 Digital Safety at 11:00 PM"
        ])
    ]

def calculate_fast_accuracy(model, dataloader, device):
    """Quick accuracy calculation - FIXED tensor size issue"""
    model.eval()
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids)
            logits = outputs['logits']
            predicted = torch.argmax(logits, dim=-1)
            
            # Fix tensor size mismatch - ensure same dimensions
            seq_len = min(predicted.size(1), target_ids.size(1) - 1)
            
            predicted_trimmed = predicted[:, :seq_len]
            target_trimmed = target_ids[:, 1:1+seq_len]  # Skip START token, match length
            
            mask = target_trimmed != 0  # Ignore padding
            correct = (predicted_trimmed == target_trimmed) & mask
            
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    return total_correct / max(1, total_tokens)

def ultra_fast_train():
    """Ultra-fast training - complete in under 2 minutes"""
    print("🚀 ULTRA FAST TRAINING - TARGET: <2 MINUTES")
    print("="*60)
    
    start_time = time.time()
    
    # Create minimal data
    training_data = create_fast_training_data()
    print(f"📊 Training examples: {len(training_data)}")
    
    # Split data - FIXED to ensure proper train/val split
    total_examples = len(training_data)
    train_size = max(1, int(total_examples * 0.75))  # 75% for training
    
    train_data = training_data[:train_size]
    val_data = training_data[train_size:] if train_size < total_examples else training_data[-2:]  # Ensure at least 2 for validation
    
    print(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation")
    
    # Create datasets with small batch size for speed
    train_dataset = FastDataset(train_data, max_length=20, vocab_file='vocab.txt')
    val_dataset = FastDataset(val_data, max_length=10, vocab_file='vocab.txt')
    
    # Use the same vocab for both datasets
    val_dataset.vocab = train_dataset.vocab
    val_dataset.vocab_size = train_dataset.vocab_size
    
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    
    print(f"📊 Vocab size: {vocab_size}")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Debug: Check vocab indices
    max_idx = max(vocab.values())
    print(f"📊 Max vocab index: {max_idx}")
    
    # Fast data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Ultra-lightweight model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UltraFastModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 Model parameters: {total_params:,}")
    print(f"🔥 Training on: {device}")
    
    # Fast optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    
    # Ultra-fast training - MORE EPOCHS for better accuracy
    num_epochs = 15  # Increased from 3 to 15
    training_history = []
    
    print(f"\n🎯 Starting {num_epochs} epochs for better accuracy...")
    print("="*60)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_loss = 0
        batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids, target_ids)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss / batches
        
        # Quick validation
        val_accuracy = calculate_fast_accuracy(model, val_loader, device)
        train_accuracy = calculate_fast_accuracy(model, train_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Save immediately after each epoch
        epoch_model = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'val_accuracy': val_accuracy,
            'train_accuracy': train_accuracy,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = f'fast_model_epoch_{epoch+1:02d}.pth'
        torch.save(epoch_model, model_path)
        
        # Track best accuracy
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            # Save as best model
            torch.save(epoch_model, 'best_fast_model.pth')
        
        # Progress indicator
        progress = (epoch + 1) / num_epochs * 100
        
        print(f"📊 EPOCH {epoch+1:2d}/{num_epochs} | Time: {epoch_time:.1f}s | Progress: {progress:5.1f}%")
        print(f"   Loss: {avg_loss:.4f} | Train: {train_accuracy*100:5.1f}% | Val: {val_accuracy*100:5.1f}%", end="")
        
        if is_best:
            print(f" 🏆 NEW BEST!")
        else:
            print(f" (Best: {best_accuracy*100:.1f}%)")
        
        print(f"   💾 Saved: {model_path}")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'time': epoch_time,
            'is_best': is_best
        })
        
        # Show improvement trend every 5 epochs
        if (epoch + 1) % 5 == 0:
            recent_acc = [h['val_accuracy'] for h in training_history[-5:]]
            trend = "📈" if recent_acc[-1] > recent_acc[0] else "📉" if recent_acc[-1] < recent_acc[0] else "➡️"
            print(f"   {trend} 5-epoch trend: {recent_acc[0]*100:.1f}% → {recent_acc[-1]*100:.1f}%")
            print()
    
    # Save final model with best accuracy info
    final_model = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'training_history': training_history,
        'best_accuracy': best_accuracy,
        'final_accuracy': training_history[-1]['val_accuracy'],
        'total_epochs': num_epochs,
        'model_config': {
            'vocab_size': vocab_size,
            'embed_dim': 32,
            'hidden_dim': 64
        }
    }
    
    torch.save(final_model, 'ultra_fast_subtitle_model_enhanced.pth')
    
    total_time = time.time() - start_time
    
    # Save training history with enhanced info
    with open('enhanced_fast_training_history.json', 'w') as f:
        json.dump({
            'training_history': training_history,
            'summary': {
                'total_epochs': num_epochs,
                'best_accuracy': best_accuracy,
                'final_accuracy': training_history[-1]['val_accuracy'],
                'total_time': total_time,
                'improvement': training_history[-1]['val_accuracy'] - training_history[0]['val_accuracy'],
                'training_examples': len(training_data),
                'vocab_size': vocab_size,
                'model_params': total_params
            }
        }, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 ENHANCED ULTRA FAST TRAINING COMPLETED!")
    print("="*60)
    print(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🎯 Final Accuracy: {training_history[-1]['val_accuracy']*100:.1f}%")
    print(f"🏆 Best Accuracy: {best_accuracy*100:.1f}%")
    print(f"📈 Improvement: {(training_history[-1]['val_accuracy'] - training_history[0]['val_accuracy'])*100:+.1f}%")
    print(f"💾 Models saved: {num_epochs} epoch models + 1 best + 1 final")
    print(f"📊 History saved: enhanced_fast_training_history.json")
    
    # Show training progression summary
    print(f"\n📊 TRAINING PROGRESSION:")
    print("="*60)
    milestones = [0, 4, 9, 14]  # Show epochs 1, 5, 10, 15
    for i in milestones:
        if i < len(training_history):
            h = training_history[i]
            marker = "🏆" if h.get('is_best', False) else "📊"
            print(f"{marker} Epoch {h['epoch']:2d}: Loss {h['loss']:.3f} | Val Acc {h['val_accuracy']*100:5.1f}%")
    
    if total_time < 300:  # 5 minutes
        print(f"\n✅ SUCCESS: Completed {num_epochs} epochs in under 5 minutes!")
    else:
        print(f"\n⚠️  Took {total_time:.1f}s for {num_epochs} epochs")
    
    print(f"\n🚀 Ready to generate better subtitles with {best_accuracy*100:.1f}% accuracy!")
    
    return model, vocab, training_history

if __name__ == "__main__":
    model, vocab, history = ultra_fast_train()