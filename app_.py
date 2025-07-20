from flask import Flask, render_template, request, redirect, url_for, session, flash
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Global variables for AI model
model_path = "ibm-granite/granite-3.0-3b-a800m-instruct"
tokenizer = None
model = None
device = None

# In-memory storage (replace with database in production)
chat_history = []
sentiment_data = {'positive': 0, 'neutral': 0, 'negative': 0}
concerns = []

def initialize_model():
    """Initialize the IBM Granite model with quantization for better performance"""
    global tokenizer, model, device
    
    print("Initializing IBM Granite model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure quantization for efficient memory usage
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # CPU inference
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            model.to(device)
        
        print("Model initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Using fallback response system...")
        return False
    
    return True

def granite_generate_response(question):
    """Generate response using IBM Granite model"""
    global tokenizer, model, device
    
    if model is None or tokenizer is None:
        return "I'm currently setting up my AI capabilities. Please try again in a moment."
    
    try:
        # Format the prompt for government services context
        prompt = f"""You are a helpful AI assistant for a government citizen engagement platform. 
        Provide clear, accurate, and helpful information about government services, policies, and civic processes.
        
        Question: {question}
        
        Answer:"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

def analyze_sentiment(text):
    """Simple sentiment analysis function"""
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Define sentiment keywords
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'outstanding', 'brilliant', 'superb', 'satisfied', 'happy', 'pleased', 'impressed', 'helpful', 'efficient']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated', 'angry', 'upset', 'poor', 'inadequate', 'useless', 'slow', 'delayed', 'problem', 'issue', 'complaint']
    
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/services')
def services():
    """Services page"""
    return render_template('services.html')

@app.route('/chat')
def chat():
    """Chat interface page"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle chat questions"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    question = request.form.get('question', '').strip()
    
    if not question:
        return render_template('chat.html', error="Please enter a question.")
    
    # Generate AI response
    response = granite_generate_response(question)
    
    # Store in chat history
    chat_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'question': question,
        'response': response
    }
    chat_history.append(chat_entry)
    
    return render_template('chat.html', question_response=response, user_question=question)

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle sentiment analysis"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    feedback_text = request.form.get('feedback', '').strip()
    
    if not feedback_text:
        return render_template('chat.html', error="Please enter feedback text.")
    
    # Analyze sentiment
    sentiment = analyze_sentiment(feedback_text)
    
    # Update sentiment counts
    sentiment_key = sentiment.lower()
    if sentiment_key in sentiment_data:
        sentiment_data[sentiment_key] += 1
    
    return render_template('chat.html', sentiment=sentiment, feedback_text=feedback_text)

@app.route('/concern', methods=['POST'])
def submit_concern():
    """Handle concern reporting"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    concern_text = request.form.get('concern', '').strip()
    
    if not concern_text:
        return render_template('chat.html', error="Please enter your concern.")
    
    # Store concern
    concern_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text': concern_text,
        'status': 'Open'
    }
    concerns.append(concern_entry)
    
    return render_template('chat.html', concern_submitted=True)

@app.route('/dashboard')
def dashboard():
    """Dashboard page with analytics"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Get recent concerns (last 10)
    recent_concerns = concerns[-10:] if concerns else []
    
    return render_template('dashboard.html', 
                         sentiment_data=sentiment_data, 
                         recent_concerns=recent_concerns,
                         total_interactions=len(chat_history))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Simple authentication (replace with proper auth in production)
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting CitizenAI Application...")
    
    # Initialize the AI model in a separate thread to avoid blocking
    try:
        model_initialized = initialize_model()
        if not model_initialized:
            print("Warning: AI model initialization failed. Using fallback responses.")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        print("Continuing with fallback responses...")
    
    print("Flask application starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
