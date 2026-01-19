import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import random
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple
import os

# Import AdamW from torch.optim (newer versions)
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

class SubtitleDataset(Dataset):
    """Dataset with multiple subtitle variations per task"""
    
    def __init__(self, data_pairs, tokenizer, max_length=128):
        self.data = []
        for task, subtitles in data_pairs:
            for subtitle in subtitles:
                self.data.append((task, subtitle))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task, subtitle = self.data[idx]
        
        task_enc = self.tokenizer(task, truncation=True, padding='max_length', 
                                   max_length=self.max_length, return_tensors='pt')
        subtitle_enc = self.tokenizer(subtitle, truncation=True, padding='max_length',
                                       max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': task_enc['input_ids'].flatten(),
            'attention_mask': task_enc['attention_mask'].flatten(),
            'target_ids': subtitle_enc['input_ids'].flatten()
        }

class SubtitleGeneratorModel(nn.Module):
    """Enhanced subtitle generator with better architecture"""
    
    def __init__(self, model_name='bert-base-uncased', hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        self.generation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.encoder.config.vocab_size)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, input_ids, attention_mask, target_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        pooled = self.dropout(pooled)
        logits = self.generation_head(pooled)
        
        if target_ids is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids[:, 0])
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

def create_massive_training_data():
    """Create comprehensive training data for ALL categories"""
    
    data = []
    
    # ============ PROFESSIONAL TASKS ============
    
    # MEETINGS
    data.extend([
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
        ])
    ])
    
    # REPORTS & DEADLINES
    data.extend([
        ("draft quarterly report today", [
            "📄 Report Warrior: Data to Decisions",
            "Quarterly Excellence: Numbers Tell Stories",
            "The Report That Defines Q3 Success",
            "Analysis Mode: Transform Data to Insights",
            "Deadline Domination: Report Edition"
        ]),
        ("submit expense report by 5 pm", [
            "⏰ Final Countdown: Expense Submission",
            "5 PM Deadline: No Receipts Left Behind",
            "Financial Accountability: Report Ready",
            "Expense Excellence: Track Every Dollar",
            "The Last Push: Submit and Celebrate"
        ]),
        ("project deadline tomorrow", [
            "⚡ Deadline Warrior: Victory Awaits",
            "24 Hours to Glory: Project Completion",
            "Tomorrow's Triumph Starts Now",
            "Final Sprint: Excellence Delivery Mode",
            "The Deadline That Will Define You"
        ]),
        ("review project proposal today", [
            "Critical Eye Activated: Proposal Review",
            "Quality Control: Polish That Proposal",
            "The Review That Ensures Success",
            "Feedback Mode: Make It Perfect",
            "Proposal Perfection: Your Expert Touch"
        ])
    ])
    
    # PROJECTS & TASKS
    data.extend([
        ("kickoff planning for project x", [
            "🚀 Project Launch: Define the Vision",
            "Where Great Projects Begin: Planning Mode",
            "Milestone Mapping: Chart Your Course",
            "The Planning That Prevents Chaos",
            "Project Genesis: Build the Foundation"
        ]),
        ("resource allocation meeting", [
            "Strategic Distribution: Resource Command",
            "Team Power-Up: Assign with Purpose",
            "The Allocation That Empowers Success",
            "Resource Mastery: Right People, Right Tasks",
            "Delegation Excellence: Optimize Your Team"
        ]),
        ("email cleanup task today", [
            "✉️ Inbox Zero Mission: Reclaim Control",
            "Digital Declutter: Email Domination",
            "The Cleanup That Restores Sanity",
            "Email Warrior: Archive and Conquer",
            "Organize to Optimize: Inbox Edition"
        ]),
        ("submit time sheet for this week", [
            "Time Tracking Excellence: Log Your Hours",
            "Weekly Accountability: Time Sheet Ready",
            "The Report That Pays: Submit Hours",
            "Hour by Hour: Document Your Value",
            "Time Sheet Triumph: Week Complete"
        ])
    ])
    
    # CALLS & EMAILS
    data.extend([
        ("follow-up call with vendor", [
            "📞 Relationship Builder: Vendor Connect",
            "The Call That Keeps Promises",
            "Follow-Through Excellence: Vendor Edition",
            "Partnership Power: Stay Connected",
            "Vendor Victory: Communication Matters"
        ]),
        ("reply to client query", [
            "✉️ Client Care: Swift Response Mode",
            "The Reply That Builds Trust",
            "Communication Excellence: Client First",
            "Query Crusher: Answer with Authority",
            "Client Satisfaction: Response Ready"
        ]),
        ("zoom interview at 11 am", [
            "🎥 Talent Scout: Find Your Champion",
            "11 AM Interview: Discover Potential",
            "The Conversation That Builds Teams",
            "Candidate Connection: Interview Excellence",
            "Hiring Hero: Meet Your Next Star"
        ])
    ])
    
    # ============ STUDENT TASKS ============
    
    # HOMEWORK & ASSIGNMENTS
    data.extend([
        ("complete math exercises chapter 5", [
            "🧮 Number Ninja: Chapter 5 Conquest",
            "Math Mastery: Problem-Solving Power",
            "Where Equations Become Easy",
            "Chapter 5 Champion: Calculate Victory",
            "Math Mission: Solve and Succeed"
        ]),
        ("science lab report due tomorrow", [
            "🔬 Lab Report Excellence: Science Mode",
            "Tomorrow's Submission: Experiment to Explanation",
            "Scientific Writing: Document Discovery",
            "The Report That Proves Your Hypothesis",
            "Lab to Paper: Transform Results"
        ]),
        ("english essay outline today", [
            "🖋️ Essay Architect: Structure Genius",
            "Outline Excellence: Blueprint Your Ideas",
            "The Framework That Guides Writing",
            "Literary Structure: Build Your Argument",
            "Essay Engineering: Organize to Win"
        ]),
        ("read history pages 45-60 tonight", [
            "📖 History Explorer: Time Travel Reading",
            "Tonight's Journey: Pages to Knowledge",
            "The Reading That Opens Perspectives",
            "Historical Wisdom: Absorb and Understand",
            "Page by Page: Build Context"
        ])
    ])
    
    # EXAMS & STUDY
    data.extend([
        ("review physics formulas for midterm", [
            "⚛️ Formula Master: Physics Prep Mode",
            "Midterm Ready: Equation Excellence",
            "The Review That Aces Exams",
            "Physics Champion: Formula Recall",
            "Test Prep: Physics Edition"
        ]),
        ("biology mock test practice", [
            "🧬 Test Simulation: Practice Makes Perfect",
            "Mock Exam Excellence: Biology Ready",
            "The Practice That Predicts Success",
            "Biology Warrior: Test Your Knowledge",
            "Exam Rehearsal: Performance Preview"
        ]),
        ("study for exam tomorrow morning", [
            "📚 Knowledge Seeker: Tomorrow's Victory Starts Now",
            "Exam Eve Excellence: Final Review",
            "Tonight's Study: Tomorrow's Success",
            "The Preparation That Pays Off",
            "Study Sprint: Exam Ready Mode"
        ]),
        ("flashcards review for spanish quiz", [
            "🇪🇸 Vocabulary Ninja: Flashcard Mastery",
            "Quiz Prep: Spanish Excellence",
            "Word by Word: Build Fluency",
            "Flashcard Champion: Rapid Recall",
            "Spanish Success: Quiz Ready"
        ])
    ])
    
    # PROJECTS & PRESENTATIONS
    data.extend([
        ("group project meeting tonight", [
            "🤝 Collective Brilliance: Study Squad Assemble",
            "Team Synergy: Project Power Hour",
            "The Meeting That Multiplies Ideas",
            "Collaboration Magic: Tonight's Session",
            "Group Genius: United We Achieve"
        ]),
        ("design slides for history presentation", [
            "🎨 Visual Storyteller: Slide Design Mode",
            "Presentation Excellence: Create Impact",
            "The Slides That Captivate Audiences",
            "Design Mastery: History Edition",
            "Visual Victory: Slides That Speak"
        ]),
        ("practice speech for oral presentation", [
            "🎤 Public Speaking: Rehearsal for Greatness",
            "Practice Makes Perfect: Speech Edition",
            "The Rehearsal That Builds Confidence",
            "Vocal Victory: Perfect Your Delivery",
            "Speech Champion: Practice Mode"
        ])
    ])
    
    # ============ FAMILY/HOME TASKS ============
    
    # CHORES & CLEANING
    data.extend([
        ("vacuum living room today", [
            "🧹 Clean Home, Clear Mind: Vacuum Mission",
            "Living Room Love: Deep Clean Edition",
            "The Cleaning That Refreshes Space",
            "Dust Buster: Carpet Perfection",
            "Home Care: Living Room Revival"
        ]),
        ("laundry day wash bedding", [
            "🧺 Fresh Sheets Mission: Laundry Excellence",
            "Bedding Refresh: Clean Sleep Ahead",
            "The Wash That Restores Comfort",
            "Laundry Warrior: Bedding Edition",
            "Clean Sheets Victory: Wash Day"
        ]),
        ("dust furniture in bedroom", [
            "✨ Dust Destroyer: Bedroom Edition",
            "Furniture Care: Polish and Pride",
            "The Dusting That Shines",
            "Bedroom Refresh: Dust-Free Zone",
            "Home Hygiene: Surface Excellence"
        ])
    ])
    
    # SHOPPING & COOKING
    data.extend([
        ("weekly grocery run this evening", [
            "🛒 Mission Possible: Pantry Restocking",
            "Grocery Warrior: Feed the Family",
            "The Shop That Sustains the Week",
            "Market Master: Fresh Food Mission",
            "Shopping Success: Cart to Kitchen"
        ]),
        ("prep sunday lunch for family", [
            "🍳 Culinary Love: Sunday Feast Prep",
            "Kitchen Magic: Family Meal Edition",
            "The Cooking That Brings Together",
            "Sunday Chef: Create Memories",
            "Family Feast: Prepare with Love"
        ]),
        ("write weekly menu plan", [
            "📝 Meal Architect: Plan the Week",
            "Menu Mastery: Organize Nutrition",
            "The Planning That Simplifies Dinners",
            "Weekly Food Strategy: Plan to Win",
            "Kitchen Commander: Menu Edition"
        ])
    ])
    
    # FAMILY EVENTS & APPOINTMENTS
    data.extend([
        ("family dinner at 6 pm tonight", [
            "🎉 Love Gathering: Family Bond Time",
            "6 PM Unity: Family Connection Hour",
            "The Dinner That Strengthens Bonds",
            "Family First: Tonight's Togetherness",
            "Dinner Table Magic: Family Edition"
        ]),
        ("doctor checkup at 10 am", [
            "⚕️ Health First: Your Wellness Journey",
            "10 AM Checkup: Invest in Health",
            "The Appointment That Prevents Problems",
            "Medical Excellence: Checkup Ready",
            "Wellness Warrior: Doctor Visit"
        ]),
        ("birthday gift shopping for niece", [
            "🎁 Gift Hunter: Find Perfect Present",
            "Birthday Magic: Gift Selection Mission",
            "The Shopping That Shows You Care",
            "Perfect Present Quest: Niece Edition",
            "Gift Giving Excellence: Birthday Ready"
        ])
    ])
    
    # PET CARE & REPAIRS
    data.extend([
        ("vet appointment for dog at 2 pm", [
            "🐾 Pet Parent: Wellness Visit for Buddy",
            "2 PM Vet: Because They Deserve Care",
            "The Appointment That Shows Love",
            "Pet Health Hero: Vet Ready",
            "Furry Friend Care: Veterinary Visit"
        ]),
        ("fix leaky faucet in kitchen", [
            "🛠️ Home Hero: Repair Mission Activated",
            "Faucet Fix: DIY Mastery Mode",
            "The Repair That Restores Function",
            "Handyman Excellence: Kitchen Edition",
            "Fix It Friday: Plumbing Victory"
        ]),
        ("change lightbulbs in hallway", [
            "💡 Bright Ideas: Light Restoration",
            "Hallway Illumination: Bulb Change Mission",
            "The Fix That Lights the Way",
            "Home Maintenance: Lighting Edition",
            "Brightness Restored: Bulb Victory"
        ])
    ])
    
    # ============ PERSONAL/OTHER TASKS ============
    
    # PERSONAL GOALS & FITNESS
    data.extend([
        ("tomorrow at 7 pm i have to go gym", [
            "💪 No Excuses: My Date with the Weights Awaits",
            "When the Clock Strikes Seven, Iron Calls",
            "Commitment Time: Why 7 PM Will Define Tomorrow",
            "The Gym Doesn't Care About Your Mood Tomorrow",
            "Twenty-Four Hours Until My Evening Workout Destiny"
        ]),
        ("morning run at 5:30 am", [
            "🏃 Champions Are Made Before Sunrise",
            "5:30 AM: The Hour of Warriors",
            "Pavement Pounding: Pre-Dawn Power",
            "While They Sleep, You Conquer Miles",
            "Morning Miles: Build Mental Strength"
        ]),
        ("yoga class at 8 pm tonight", [
            "🧘 Find Your Center: Evening Zen Session",
            "8 PM Peace: Breathe, Bend, Believe",
            "Tonight's Tranquility: Yoga Flow",
            "Flexibility Is Freedom: Yoga Edition",
            "Evening Zen: Mind-Body Harmony"
        ]),
        ("gym session leg day tomorrow", [
            "🦵 The Day Your Legs Will Remember Forever",
            "No Pain, No Gains: Leg Destruction Protocol",
            "Tomorrow's Limp: Today's Badge of Honor",
            "Leg Day: Where Elevators Become Friends",
            "Stairway to Strength: Leg Edition"
        ]),
        ("track 10000 steps today", [
            "👟 Step Counter: 10K Victory Mission",
            "Every Step Counts: Movement Matters",
            "The Walk That Builds Health",
            "10,000 Steps: Daily Excellence",
            "Step by Step: Fitness Freedom"
        ])
    ])
    
    # HOBBIES & READING
    data.extend([
        ("piano practice at 5 pm", [
            "🎹 Melody Master: Keys to Harmony",
            "5 PM Symphony: Practice Makes Perfect",
            "The Hour Where Music Comes Alive",
            "Piano Progress: Key by Key",
            "Musical Excellence: Practice Edition"
        ]),
        ("read 50 pages of current novel", [
            "📖 Mind Journey: Literary Escape Time",
            "Page Turner: 50 Pages to Another World",
            "The Reading That Expands Horizons",
            "Book Lover: Chapter Progress",
            "Literary Adventure: Read and Grow"
        ]),
        ("painting session this afternoon", [
            "🎨 Creative Flow: Canvas Calling",
            "Afternoon Art: Express Your Soul",
            "The Session That Creates Beauty",
            "Artistic Excellence: Paint Edition",
            "Color Your World: Art Time"
        ]),
        ("daily journal writing tonight", [
            "📝 Reflection Time: Journal Your Journey",
            "Tonight's Writing: Capture Today",
            "The Habit That Heals: Journal Edition",
            "Daily Documentation: Write to Remember",
            "Journal Joy: Evening Reflection"
        ])
    ])
    
    # MINDFULNESS & ROUTINES
    data.extend([
        ("meditation at dawn tomorrow", [
            "🧘‍♂️ Soul Awakening: Inner Peace Protocol",
            "Dawn Meditation: Start Day with Calm",
            "The Practice That Centers You",
            "Morning Mindfulness: Breathe and Begin",
            "Meditation Mode: Peace Activation"
        ]),
        ("evening routine wind down 10 pm", [
            "🌙 Night Ritual: Prepare for Rest",
            "10 PM Protocol: Wind Down Wisdom",
            "The Routine That Ensures Sleep",
            "Evening Excellence: Bedtime Ready",
            "Wind Down Mode: Rest Preparation"
        ]),
        ("10 minute stretch routine morning", [
            "🤸 Morning Mobility: Stretch to Strength",
            "10 Minutes to Flexibility Freedom",
            "The Stretch That Starts Days Right",
            "Flexibility First: Morning Edition",
            "Stretch Success: Wake Up Routine"
        ])
    ])
    
    # ERRANDS & SOCIAL
    data.extend([
        ("bank visit discuss savings account", [
            "💰 Financial Future: Banking Wisdom",
            "Savings Strategy: Bank Meeting",
            "The Visit That Builds Wealth",
            "Banking Excellence: Account Planning",
            "Money Matters: Savings Session"
        ]),
        ("post office send return packages", [
            "📦 Shipping Mission: Package Return",
            "Post Office Power: Send and Seal",
            "The Errand That Completes Returns",
            "Mailing Mastery: Package Edition",
            "Return Victory: Shipping Success"
        ]),
        ("coffee with friend at 2 pm", [
            "☕ Friendship Fuel: Connection Time",
            "2 PM Chat: Strengthen Bonds",
            "The Coffee That Nurtures Friendship",
            "Social Excellence: Friend Edition",
            "Coffee Connection: Quality Time"
        ]),
        ("rsvp to party invitation", [
            "🎊 Social Commitment: RSVP Ready",
            "The Response That Shows You Care",
            "Party Planning: Confirm Attendance",
            "RSVP Excellence: Social Edition",
            "Social Calendar: Confirm and Commit"
        ])
    ])
    
    # CAREER & LEARNING
    data.extend([
        ("update portfolio with latest project", [
            "💼 Career Builder: Portfolio Perfection",
            "The Update That Opens Doors",
            "Portfolio Power: Showcase Success",
            "Professional Pride: Display Your Work",
            "Career Excellence: Portfolio Edition"
        ]),
        ("complete python course module today", [
            "👨‍💻 Code Warrior: Python Progress",
            "Module Mastery: Learn and Level Up",
            "The Lesson That Builds Skills",
            "Coding Excellence: Python Edition",
            "Developer Journey: Module Complete"
        ]),
        ("language lesson 15 mins duolingo", [
            "🌍 Language Explorer: Daily Practice",
            "15 Minutes to Fluency: Duolingo Edition",
            "The Lesson That Opens Worlds",
            "Linguistic Excellence: Practice Mode",
            "Language Learning: Daily Progress"
        ]),
        ("attend webinar on time management", [
            "🎓 Knowledge Seeker: Webinar Wisdom",
            "Time Management Mastery: Learn to Optimize",
            "The Session That Transforms Productivity",
            "Webinar Excellence: Skill Building",
            "Professional Development: Webinar Ready"
        ])
    ])
    
    # URGENT/IMPORTANT TASKS
    data.extend([
        ("urgent report due in 2 hours", [
            "🚨 Code Red: Report Mission Critical",
            "⏰ 2 Hour Countdown: Excellence Delivery",
            "The Sprint That Defines Character",
            "Urgent Mode: Report Domination",
            "Crisis Excellence: 2 Hour Mission"
        ]),
        ("important call at 1 pm urgent", [
            "⚡ Priority Alert: Communication Required",
            "1 PM Critical: The Call That Matters",
            "Urgent Excellence: Phone Ready",
            "High Priority: Call Connection",
            "The Call That Changes Everything"
        ]),
        ("emergency meeting right now", [
            "🔥 All Hands: Crisis Management Mode",
            "Emergency Protocol: Meeting Activated",
            "The Meeting That Solves Crisis",
            "Urgent Assembly: Team Response",
            "Crisis Mode: Meeting Now"
        ]),
        ("submit application deadline today", [
            "⚠️ Last Chance: Future Depends on This",
            "Today's Deadline: Submit and Succeed",
            "The Application That Opens Doors",
            "Final Hours: Submission Excellence",
            "Deadline Day: Application Victory"
        ])
    ])
    
    # ADD MORE VARIATIONS WITH TIME PATTERNS
    times = ["morning", "afternoon", "evening", "tonight", "tomorrow", "today"]
    
    for time in times:
        data.extend([
            (f"study session {time}", [
                f"📚 Knowledge Power: {time.title()} Study Mode",
                f"{time.title()} Excellence: Brain Training",
                f"Study Champion: {time.title()} Edition",
                f"Learning Excellence: {time.title()} Session",
                f"{time.title()} Wisdom: Study Time"
            ]),
            (f"work on project {time}", [
                f"🚀 Project Mode: {time.title()} Progress",
                f"{time.title()} Excellence: Build and Create",
                f"Project Champion: {time.title()} Edition",
                f"Creative Excellence: {time.title()} Work",
                f"{time.title()} Productivity: Project Time"
            ])
        ])
    
    return data

def train_model(model, train_loader, val_loader, num_epochs=25, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids, attention_mask, target_ids)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                outputs = model(input_ids, attention_mask, target_ids)
                total_val_loss += outputs['loss'].item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_subtitle_model.pth')
            print(f'  ✅ New best model saved! (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  📊 No improvement ({patience_counter}/{patience})')
        
        if patience_counter >= patience:
            print(f'\n⚠️ Early stopping triggered after {epoch+1} epochs')
            break
        
        print('='*60 + '\n')

def main():
    print("🚀 COMPREHENSIVE SUBTITLE GENERATOR TRAINING")
    print("="*60)
    
    print("📊 Creating massive training dataset...")
    training_data = create_massive_training_data()
    print(f"✅ Created {len(training_data)} task examples")
    
    # Calculate total subtitle variations
    total_variations = sum(len(subtitles) for _, subtitles in training_data)
    print(f"✅ Total subtitle variations: {total_variations}")
    
    # Split data
    train_data, val_data = train_test_split(training_data, test_size=0.15, random_state=42)
    print(f"✅ Train: {len(train_data)} | Validation: {len(val_data)}")
    
    # Initialize
    print("\n🤖 Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SubtitleGeneratorModel()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("\n📦 Creating datasets and dataloaders...")
    train_dataset = SubtitleDataset(train_data, tokenizer)
    val_dataset = SubtitleDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Train
    print("\n🎯 Starting training...")
    print("="*60 + "\n")
    train_model(model, train_loader, val_loader, num_epochs=30)
    
    print("\n🎉 Training completed!")
    print("💾 Best model saved as 'best_subtitle_model.pth'")
    
    # Save tokenizer
    tokenizer.save_pretrained('./subtitle_tokenizer')
    print("💾 Tokenizer saved to './subtitle_tokenizer'")
    
    print("\n" + "="*60)
    print("✅ ALL DONE! Your model is ready to generate subtitles!")
    print("="*60)

if __name__ == "__main__":
    main()