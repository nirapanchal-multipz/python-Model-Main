import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta


class EnhancedSubtitleDataset(Dataset):
    """Enhanced dataset using external vocabulary"""

    def __init__(self, data_pairs, vocab, max_length=128):
        self.data = []
        for task, subtitles in data_pairs:
            for subtitle in subtitles:
                self.data.append((task.lower().strip(), subtitle.strip()))

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_length = max_length

        # Create reverse mapping for decoding
        self.idx_to_word = {idx: word for word, idx in vocab.items()}

    def _text_to_indices(self, text):
        """Convert text to indices with better tokenization"""
        # Simple word tokenization (you can improve this)
        words = text.lower().split()[:self.max_length-2]

        indices = [self.vocab.get('<START>', 2)]

        for word in words:
            # Handle punctuation attached to words
            clean_word = word.strip('.,!?;:')
            indices.append(self.vocab.get(
                clean_word, self.vocab.get('<UNK>', 1)))

        indices.append(self.vocab.get('<END>', 3))

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.vocab.get('<PAD>', 0))

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


class ImprovedSubtitleModel(nn.Module):
    """Enhanced model with better architecture"""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=3, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Enhanced embedding with dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Decoder
        self.decoder = nn.LSTM(
            embed_dim + hidden_dim * 2,  # Embedding + context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=0.1)

    def forward(self, input_ids, target_ids=None):
        batch_size, seq_len = input_ids.size()

        # Encode input
        input_embeds = self.embed_dropout(self.embedding(input_ids))
        encoder_out, (hidden, cell) = self.encoder(input_embeds)

        # Apply self-attention
        attended, _ = self.attention(encoder_out, encoder_out, encoder_out)
        attended = self.layer_norm(
            attended + encoder_out)  # Residual connection

        if target_ids is not None:
            # Training mode
            target_embeds = self.embed_dropout(
                self.embedding(target_ids[:, :-1]))

            # Expand attention context for each decoder step
            context = attended.unsqueeze(
                1).expand(-1, target_embeds.size(1), -1, -1)
            context = context.mean(dim=2)  # Average over sequence

            # Concatenate embeddings with context
            decoder_input = torch.cat([target_embeds, context], dim=-1)

            # Initialize decoder hidden state from encoder
            decoder_hidden = (
                hidden[:self.num_layers].contiguous(),
                cell[:self.num_layers].contiguous()
            )

            decoder_out, _ = self.decoder(decoder_input, decoder_hidden)
            logits = self.output_proj(decoder_out)

            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                target_ids[:, 1:].reshape(-1)
            )

            return {'loss': loss, 'logits': logits}
        else:
            # Inference mode
            decoder_hidden = (
                hidden[:self.num_layers].contiguous(),
                cell[:self.num_layers].contiguous()
            )

            outputs = []
            current_input = self.embedding(
                torch.full((batch_size, 1), 2, device=input_ids.device)
            )

            context = attended.mean(dim=1, keepdim=True)

            for _ in range(seq_len):
                decoder_input = torch.cat([current_input, context], dim=-1)
                decoder_out, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                logits = self.output_proj(decoder_out)
                outputs.append(logits)

                predicted = torch.argmax(logits, dim=-1)
                current_input = self.embedding(predicted)

            return {'logits': torch.cat(outputs, dim=1)}


def load_vocabulary(vocab_path='voaca.txt'):
    """Load vocabulary from file"""
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }

    if os.path.exists(vocab_path):
        print(f"📖 Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word not in vocab:
                    vocab[word.lower()] = len(vocab)
        print(f"✅ Loaded {len(vocab):,} words from vocabulary file")
    else:
        print(f"⚠️  Vocabulary file not found: {vocab_path}")
        print("Creating basic vocabulary...")

    return vocab


def create_comprehensive_training_data():
    """Create extensive, high-quality training data"""

    data = []

    # Fitness & Gym Tasks (200+ examples)
    gym_times = ["6:00 AM", "7:00 AM", "8:00 PM",
                 "9:00 PM", "6:30 PM", "5:00 PM"]
    gym_actions = [
        "workout", "gym session", "training", "fitness class",
        "cardio session", "strength training", "yoga class", "pilates"
    ]

    gym_templates = [
        "💪 Unleash Your Power: {} {} Today",
        "🔥 Destiny Awaits: {} at {}",
        "⚡ Champion's Hour: {} Scheduled for {}",
        "🏆 Victory Begins: {} at {} Sharp",
        "🎯 Excellence Demands: {} Session at {}",
        "💯 No Excuses: {} Challenge at {}",
        "🚀 Transform Yourself: {} at {} Tonight",
        "⭐ Greatness Calls: {} Scheduled {}",
        "🔱 Warrior Mode: {} Battle at {}",
        "💥 Beast Mode Activated: {} at {} Today"
    ]

    for action in gym_actions:
        for time in gym_times:
            task = f"{action} at {time}"
            subtitles = [
                template.format(action.title(), time)
                for template in random.sample(gym_templates, 5)
            ]
            data.append((task, subtitles))

    # Work & Professional Tasks (300+ examples)
    work_times = ["9:00 AM", "2:00 PM", "3:30 PM",
                  "11:00 AM", "4:00 PM", "10:00 AM"]
    work_actions = [
        "meeting with client", "team standup", "presentation",
        "conference call", "project review", "strategy session",
        "client presentation", "board meeting", "sales call",
        "performance review", "brainstorming session", "workshop"
    ]

    work_templates = [
        "💼 Professional Excellence: {} at {}",
        "🎯 Business Success: {} Scheduled {}",
        "⚡ Career Milestone: {} at {} Today",
        "🏆 Leadership Moment: {} at {}",
        "💯 Strategic Victory: {} Session {}",
        "📊 Business Impact: {} at {} Sharp",
        "🚀 Professional Growth: {} Scheduled {}",
        "⭐ Excellence Awaits: {} at {} Today",
        "💎 Career Excellence: {} Meeting at {}",
        "🎖️ Professional Pride: {} at {}"
    ]

    for action in work_actions:
        for time in work_times:
            task = f"{action} at {time}"
            subtitles = [
                template.format(action.title(), time)
                for template in random.sample(work_templates, 5)
            ]
            data.append((task, subtitles))

    # Academic & Study Tasks (200+ examples)
    study_times = ["7:00 AM", "3:00 PM",
                   "8:00 PM", "9:00 AM", "6:00 PM", "4:00 PM"]
    study_actions = [
        "exam preparation", "study session", "library time",
        "assignment work", "research project", "group study",
        "lecture review", "quiz preparation", "thesis work",
        "reading assignment", "lab session", "tutorial"
    ]

    study_templates = [
        "📚 Knowledge Quest: {} at {}",
        "🎓 Academic Excellence: {} Scheduled {}",
        "⚡ Learning Victory: {} at {} Today",
        "🏆 Scholar's Path: {} Session at {}",
        "💯 Educational Success: {} at {}",
        "📖 Wisdom Awaits: {} Scheduled {}",
        "🚀 Academic Growth: {} at {} Sharp",
        "⭐ Excellence Through Learning: {} at {}",
        "🎯 Study Victory: {} Session {}",
        "💎 Knowledge Power: {} at {} Today"
    ]

    for action in study_actions:
        for time in study_times:
            task = f"{action} at {time}"
            subtitles = [
                template.format(action.title(), time)
                for template in random.sample(study_templates, 5)
            ]
            data.append((task, subtitles))

    # Health & Wellness Tasks (150+ examples)
    health_times = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM", "11:00 AM"]
    health_actions = [
        "doctor appointment", "dentist visit", "therapy session",
        "medical checkup", "physiotherapy", "counseling session",
        "health screening", "vaccination", "eye checkup",
        "nutrition consultation", "mental health session"
    ]

    health_templates = [
        "⚕️ Health Priority: {} at {}",
        "💚 Wellness Journey: {} Scheduled {}",
        "🎯 Self Care Excellence: {} at {} Today",
        "🏥 Health Victory: {} Session at {}",
        "💯 Taking Care: {} at {}",
        "⭐ Wellness First: {} Scheduled {}",
        "🚀 Health Excellence: {} at {} Sharp",
        "💎 Your Health Matters: {} at {}",
        "🎖️ Wellness Priority: {} Session {}",
        "🌟 Health Success: {} at {} Today"
    ]

    for action in health_actions:
        for time in health_times:
            task = f"{action} at {time}"
            subtitles = [
                template.format(action.title(), time)
                for template in random.sample(health_templates, 5)
            ]
            data.append((task, subtitles))

    # Personal & Life Tasks (200+ examples)
    personal_actions = [
        "grocery shopping", "house cleaning", "laundry day",
        "meal prep", "car maintenance", "gardening time",
        "pet care", "home organizing", "bill payments",
        "family dinner", "movie night", "game night",
        "date night", "birthday party", "celebration"
    ]

    personal_templates = [
        "🏠 Home Excellence: {} Today",
        "🎯 Life Balance: {} Scheduled",
        "⚡ Personal Victory: {} Time",
        "🏆 Life Success: {} Today",
        "💯 Living Well: {} Scheduled",
        "⭐ Personal Excellence: {} Time",
        "🚀 Life Victory: {} Today",
        "💎 Quality Time: {} Scheduled",
        "🎖️ Personal Growth: {} Today",
        "🌟 Life Balance: {} Time"
    ]

    for action in personal_actions:
        subtitles = [
            template.format(action.title())
            for template in random.sample(personal_templates, 5)
        ]
        data.append((action, subtitles))

    # Deadline Tasks (100+ examples)
    deadlines = [
        "project due tonight", "assignment due tomorrow",
        "report due by 5 PM", "presentation due today",
        "submission deadline today", "final draft due tomorrow"
    ]

    deadline_templates = [
        "⚡ Final Sprint: {}",
        "🚨 Deadline Alert: {} Approaching",
        "🎯 Time to Deliver: {}",
        "💯 Final Push: {} Now",
        "🏆 Completion Time: {}",
        "⏰ Deadline Excellence: {}",
        "🔥 Final Hours: {}",
        "💪 Finish Strong: {}",
        "⭐ Deliver Excellence: {}",
        "🎖️ Achievement Moment: {}"
    ]

    for deadline in deadlines:
        subtitles = [
            template.format(deadline.title())
            for template in random.sample(deadline_templates, 5)
        ]
        data.append((deadline, subtitles))

    # Add variations with different phrasings
    variations = [
        ("go to gym tomorrow morning", [
            "💪 Morning Warrior: Gym Battle Tomorrow",
            "🔥 Dawn Training: Tomorrow's Gym Session",
            "⚡ Morning Excellence: Gym Time Tomorrow",
            "🏆 Early Victory: Tomorrow's Workout",
            "🎯 Morning Mission: Gym Challenge Tomorrow"
        ]),
        ("important client meeting this afternoon", [
            "💼 Critical Hour: Client Meeting This Afternoon",
            "🎯 Business Excellence: Afternoon Client Session",
            "⚡ Professional Milestone: Meeting This Afternoon",
            "🏆 Client Success: Important Meeting Today",
            "💯 Business Victory: Afternoon Client Hour"
        ]),
        ("study for final exam tonight", [
            "📚 Victory Quest: Final Exam Prep Tonight",
            "🎓 Academic Battle: Tonight's Study Session",
            "⚡ Knowledge Power: Final Exam Preparation",
            "🏆 Scholar's Mission: Tonight's Study Hour",
            "💯 Exam Excellence: Tonight's Preparation"
        ])
    ]

    data.extend(variations)

    print(f"✅ Generated {len(data)} unique task types")
    total_examples = sum(len(subs) for _, subs in data)
    print(f"✅ Total training examples: {total_examples:,}")

    return data


def calculate_metrics(model, dataloader, vocab, device):
    """Calculate comprehensive metrics"""
    model.eval()

    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            outputs = model(input_ids, target_ids)
            total_loss += outputs['loss'].item()

            logits = outputs['logits']
            predicted = torch.argmax(logits, dim=-1)

            # Token-level accuracy
            mask = target_ids[:, 1:] != 0
            correct_tokens = (predicted == target_ids[:, 1:]) & mask

            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += mask.sum().item()

            # Sequence-level accuracy
            for pred, target in zip(predicted, target_ids[:, 1:]):
                mask_seq = target != 0
                if torch.all(pred[mask_seq] == target[mask_seq]):
                    total_correct_sequences += 1
                total_sequences += 1

    return {
        'loss': total_loss / len(dataloader),
        'token_accuracy': total_correct_tokens / max(1, total_tokens),
        'sequence_accuracy': total_correct_sequences / max(1, total_sequences),
        'total_tokens': total_tokens,
        'total_sequences': total_sequences
    }


def train_model(model, train_loader, val_loader, vocab, num_epochs=50, lr=5e-4):
    """Enhanced training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Training on: {device}")

    model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = 10

    training_history = []

    print(f"\n{'='*90}")
    print(f"🚀 TRAINING START - {num_epochs} EPOCHS")
    print(f"{'='*90}\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            outputs = model(input_ids, target_ids)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / train_batches
                print(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = train_loss / train_batches

        # Validation phase
        val_metrics = calculate_metrics(model, val_loader, vocab, device)
        train_metrics = calculate_metrics(model, train_loader, vocab, device)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        epoch_time = time.time() - epoch_start

        # Display results
        print(f"\n{'='*90}")
        print(
            f"📊 EPOCH {epoch+1}/{num_epochs} COMPLETE | Time: {epoch_time:.1f}s")
        print(f"{'='*90}")
        print(f"LOSSES:")
        print(f"  Train Loss:      {train_metrics['loss']:.4f}")
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
        print(f"\nACCURACY:")
        print(
            f"  Train Token Acc:      {train_metrics['token_accuracy']*100:.2f}%")
        print(
            f"  Train Sequence Acc:   {train_metrics['sequence_accuracy']*100:.2f}%")
        print(
            f"  Val Token Acc:        {val_metrics['token_accuracy']*100:.2f}%")
        print(
            f"  Val Sequence Acc:     {val_metrics['sequence_accuracy']*100:.2f}%")
        print(f"\nSTATS:")
        print(f"  Learning Rate:   {optimizer.param_groups[0]['lr']:.6f}")
        print(
            f"  Tokens Processed: {train_metrics['total_tokens'] + val_metrics['total_tokens']:,}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_token_acc': train_metrics['token_accuracy'],
            'val_token_acc': val_metrics['token_accuracy'],
            'train_seq_acc': train_metrics['sequence_accuracy'],
            'val_seq_acc': val_metrics['sequence_accuracy'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
            'timestamp': datetime.now().isoformat()
        })

        # 🔥 SAVE MODEL IMMEDIATELY AFTER EACH EPOCH
        current_accuracy = val_metrics['token_accuracy']
        
        # Save current epoch model
        epoch_model_path = f'subtitle_model_epoch_{epoch+1}.pth'
        epoch_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': current_accuracy,
            'val_loss': val_metrics['loss'],
            'train_loss': train_metrics['loss'],
            'vocab': vocab,
            'training_history': training_history,
            'model_config': {
                'vocab_size': model.vocab_size,
                'embed_dim': model.embed_dim,
                'hidden_dim': model.hidden_dim,
                'num_layers': model.num_layers
            }
        }
        
        torch.save(epoch_checkpoint, epoch_model_path)
        
        # Save training history after each epoch
        with open(f'training_history_epoch_{epoch+1}.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"💾 EPOCH MODEL SAVED: {epoch_model_path}")
        print(f"📊 HISTORY SAVED: training_history_epoch_{epoch+1}.json")

        # Save best model
        if val_metrics['token_accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['token_accuracy']
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['token_accuracy'],
                'val_loss': val_metrics['loss'],
                'vocab': vocab,
                'training_history': training_history,
                'model_config': {
                    'vocab_size': model.vocab_size,
                    'embed_dim': model.embed_dim,
                    'hidden_dim': model.hidden_dim,
                    'num_layers': model.num_layers
                }
            }

            torch.save(best_checkpoint, 'best_subtitle_model_enhanced.pth')
            print(f"🏆 NEW BEST MODEL! Val Accuracy: {best_val_accuracy*100:.2f}%")
            print(f"💾 BEST MODEL SAVED: best_subtitle_model_enhanced.pth")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{max_patience}")
            print(f"📊 Current Best: {best_val_accuracy*100:.2f}%")

        print(f"{'='*90}\n")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    return training_history


def main():
    print("\n" + "="*90)
    print("🚀 ENHANCED SUBTITLE GENERATOR TRAINING SYSTEM")
    print("="*90)

    # Load vocabulary
    vocab = load_vocabulary('vocab.txt')
    print(f"📊 Vocabulary size: {len(vocab):,} words\n")

    # Create training data
    print("📦 Generating comprehensive training data...")
    training_data = create_comprehensive_training_data()

    # Split data
    train_data, val_data = train_test_split(
        training_data, test_size=0.15, random_state=42)
    print(f"\n📊 Data Split:")
    print(f"  Training:   {len(train_data):,} task types")
    print(f"  Validation: {len(val_data):,} task types")

    # Create datasets
    print("\n🔧 Creating datasets...")
    train_dataset = EnhancedSubtitleDataset(train_data, vocab, max_length=128)
    val_dataset = EnhancedSubtitleDataset(val_data, vocab, max_length=128)

    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Initialize model
    print("\n🤖 Initializing enhanced model...")
    model = ImprovedSubtitleModel(
        vocab_size=len(vocab),
        embed_dim=256,
        hidden_dim=512,
        num_layers=3,
        dropout=0.3
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Train
    print("\n" + "="*90)
    training_history = train_model(
        model,
        train_loader,
        val_loader,
        vocab,
        num_epochs=50,
        lr=5e-4
    )

    # Save final results
    with open('training_history_enhanced.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "="*90)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*90)
    print(f"💾 Model saved: 'best_subtitle_model_enhanced.pth'")
    print(f"📊 History saved: 'training_history_enhanced.json'")
    print(
        f"📈 Best validation accuracy: {max(h['val_token_acc'] for h in training_history)*100:.2f}%")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
