import firebase_admin
from flask import Flask, redirect, render_template, request, jsonify, session, url_for
from functools import wraps
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
from datetime import datetime
import os
import base64
from firebase_admin import credentials, firestore, auth
import requests
from werkzeug.security import generate_password_hash, check_password_hash

# ─── OS CONCEPT: MULTITHREADING ───────────────────────────────────────────────
# Python's `threading` module lets us create threads — lightweight sub-processes
# that share the same memory space and run concurrently within one program.
#
# Why we need it here:
#   The Gemini AI call in analyze_plant() can take 5–15 seconds.
#   Without threading: Flask blocks on that call → the user's browser spins
#   and they cannot navigate anywhere until it finishes.
#   With threading:    Flask returns a job_id instantly (<100 ms), the AI
#   call runs in the background, and the user can freely browse other pages.
#
# OS parallel:
#   A thread is the smallest unit of CPU execution inside an OS process.
#   The OS scheduler interleaves thread execution — while Thread-A waits for
#   a network response (I/O-bound), Thread-B (Flask server) keeps running.
#   This is called Concurrent Execution / I/O Concurrency.
#
# GIL note (viva tip):
#   CPython has a Global Interpreter Lock (GIL) — only ONE thread runs Python
#   bytecode at a time. BUT the GIL is RELEASED during I/O waits (network
#   calls, file writes) — exactly what we do here — so threads genuinely run
#   in parallel for the slow Gemini API + Firestore parts.
# ──────────────────────────────────────────────────────────────────────────────
import threading

# Initialize Firebase Admin
cred = credentials.Certificate('fbAdminconfig.json')
firebase_app = firebase_admin.initialize_app(cred)
db = firestore.client()

# ─── IN-MEMORY JOB STORE ─────────────────────────────────────────────────────
# A plain Python dict that maps job_id → job status/result.
# Each background thread writes its progress here; the frontend polls
# GET /api/job/<job_id> to check when the job is done.
#
# Structure of each job entry:
#   {
#     'status' : 'pending' | 'done' | 'error',
#     'result' : { ... }   (set when done),
#     'error'  : 'msg'     (set when error),
#   }
#
# Thread-safety:
#   Python's dict operations (read / write a single key) are atomic due to
#   the GIL, so we do NOT need a Lock for this simple use-case. If you ever
#   do complex read-modify-write sequences, use threading.Lock() instead.
# ──────────────────────────────────────────────────────────────────────────────
diagnosis_jobs   = {}   # job_id  → job dict  (diagnosis)
identify_jobs    = {}   # job_id  → job dict  (identification)
_jobs_lock       = threading.Lock()   # used when we create/delete entries


# ─── BACKGROUND WORKER: DIAGNOSIS ────────────────────────────────────────────
#
# OS concept — Thread lifecycle:
#   NEW → READY → RUNNING → (WAITING for I/O) → RUNNING → TERMINATED
#
# This function is the thread's target (the code it runs).
# It is started by threading.Thread(target=_diagnosis_worker, ...) and
# runs CONCURRENTLY with the Flask request-handler thread.
#
# Parameters are passed by value so we don't share mutable request objects
# across thread boundaries (avoids race conditions).
# ──────────────────────────────────────────────────────────────────────────────
def _diagnosis_worker(job_id, user_id, image_bytes, image_path, analysis_type='diagnosis'):
    """
    Background thread target for plant diagnosis / identification.

    Steps (all run off the main thread so Flask stays responsive):
      1. Decode image bytes into a PIL Image                    (CPU)
      2. Call Gemini AI API with the image + prompt             (I/O — slow)
      3. Format the raw text response                           (CPU)
      4. Extract condition/severity metadata                    (CPU)
      5. Save the result to Firebase Firestore                  (I/O)
      6. Update the in-memory job store → status = 'done'       (memory write)
    """
    job_store = diagnosis_jobs if analysis_type == 'diagnosis' else identify_jobs

    try:
        print(f"[Thread {threading.current_thread().name}] Starting {analysis_type} for job {job_id}")

        # ── Step 1: decode image ──────────────────────────────────────────────
        image = Image.open(io.BytesIO(image_bytes))

        # ── Step 2: call Gemini AI (the slow I/O step) ───────────────────────
        # The GIL is released here while waiting for the network response,
        # so the Flask server thread runs freely during this wait.
        model = genai.GenerativeModel("gemini-2.5-flash")

        if analysis_type == 'diagnosis':
            prompt = """You are an expert plant pathologist. Analyze this plant image for diseases and health issues.

Provide a comprehensive diagnosis in the following format:

PLANT HEALTH STATUS:
[Overall condition - Healthy/Unhealthy/Critical]

DISEASE IDENTIFICATION:
Disease Name: [Specific disease name or "No disease detected"]
Common Name: [How it's commonly known]

SEVERITY LEVEL:
[Mild/Moderate/Severe/Critical]

SYMPTOMS OBSERVED:
• [List all visible symptoms]
• [Color changes, spots, wilting, etc.]
• [Pattern and distribution]

PROBABLE CAUSES:
• [Primary cause - fungal/bacterial/viral/pest/environmental]
• [Contributing factors]
• [Environmental conditions]

TREATMENT RECOMMENDATIONS:
Immediate Actions:
• [Urgent steps to take]
• [What to remove/isolate]

Treatment Plan:
• [Specific treatments - organic/chemical]
• [Application method and frequency]
• [Duration of treatment]

Care Adjustments:
• [Watering changes]
• [Light requirements]
• [Humidity/temperature needs]

PREVENTION STRATEGIES:
• [How to prevent recurrence]
• [Good practices]
• [Environmental management]

PROGNOSIS:
[Expected recovery time and success rate]
[Warning signs to watch for]

ADDITIONAL NOTES:
[Any other relevant information or recommendations]

Be specific and actionable in your recommendations."""
        else:
            prompt = """You are an expert botanist. Identify this plant and provide comprehensive information in a well-structured format:

PLANT IDENTIFICATION:
Common Name: [Primary common name]
Scientific Name: [Genus species]
Other Names: [Alternative common names]

CLASSIFICATION:
Family: [Plant family]
Origin: [Native region/habitat]
Type: [Annual/Perennial/Shrub/Tree/etc.]

PHYSICAL CHARACTERISTICS:
• Leaves: [Shape, size, color, arrangement]
• Flowers: [If visible - color, size, season]
• Growth Habit: [Height, spread, growth rate]
• Special Features: [Unique identifying traits]

CARE REQUIREMENTS:
Light: [Full sun/Partial shade/Shade with specifics]
Water: [Frequency and amount]
Soil: [Type, pH, drainage needs]
Temperature: [Ideal range, hardiness zones]
Humidity: [Preferences]
Fertilizer: [Type and frequency]

CARE DIFFICULTY:
[Easy/Moderate/Challenging with explanation]

TOXICITY INFORMATION:
Pets: [Safe/Toxic with details]
Humans: [Safe/Toxic with details]
Handling: [Any precautions needed]

PROPAGATION:
• [Methods: seeds, cuttings, division, etc.]
• [Best time and success tips]

COMMON ISSUES:
• [Typical pests or diseases]
• [Prevention strategies]

INTERESTING FACTS:
• [Cultural significance, uses, or unique properties]
• [Growing tips or fun information]

COMPANION PLANTS:
[Plants that grow well together]

Be accurate and comprehensive. If you cannot identify the plant with certainty, explain what category it might belong to and what additional photos would help."""

        response = model.generate_content([prompt, image])

        # ── Step 3 & 4: format + extract metadata ─────────────────────────────
        formatted_analysis = format_response_enhanced(response.text)

        if analysis_type == 'diagnosis':
            metadata = extract_plant_metadata(response.text)
            history_entry = {
                'timestamp':          datetime.now().isoformat(),
                'analysis':           response.text,
                'formatted_analysis': formatted_analysis,
                'type':               'diagnosis',
                'image_path':         image_path,
                'condition':          metadata['condition'],
                'severity':           metadata['severity'],
            }
        else:
            history_entry = {
                'timestamp':          datetime.now().isoformat(),
                'analysis':           response.text,
                'formatted_analysis': formatted_analysis,
                'type':               'identification',
                'image_path':         image_path,
            }

        # ── Step 5: save to Firestore (another I/O step) ──────────────────────
        # Firebase is thread-safe — multiple threads can write simultaneously.
        doc_id = save_history_to_db(user_id, history_entry)
        print(f"[Thread {threading.current_thread().name}] Saved to Firestore. doc_id={doc_id}")

        # ── Step 6: mark job as done ──────────────────────────────────────────
        with _jobs_lock:
            job_store[job_id]['status'] = 'done'
            job_store[job_id]['result'] = {
                'analysis':           response.text,
                'formatted_analysis': formatted_analysis,
                'doc_id':             doc_id,
            }

        print(f"[Thread {threading.current_thread().name}] Job {job_id} DONE ✓")

    except Exception as e:
        # Always catch exceptions in threads — an uncaught exception in a
        # background thread silently kills that thread (Flask won't see it).
        print(f"[Thread {threading.current_thread().name}] Job {job_id} FAILED: {e}")
        import traceback; traceback.print_exc()
        with _jobs_lock:
            job_store[job_id]['status'] = 'error'
            job_store[job_id]['error']  = str(e)


def get_api_key():
    """
    Load API key directly from .env file to avoid dotenv caching issues
    Returns the cleaned API key or raises an error
    """
    print("\n" + "="*60)
    print("LOADING API KEY")
    print("="*60)
    
    # Try 1: Read directly from .env file (MOST RELIABLE)
    try:
        with open('.env', 'r') as f:
            content = f.read()
            print(f"✓ Reading .env file directly")
            
            # Find GEMINI_API_KEY line
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('GEMINI_API_KEY='):
                    # Extract value after =
                    raw_key = line.split('=', 1)[1].strip()
                    
                    # Clean the key (remove quotes, extra spaces)
                    cleaned_key = raw_key.replace('"', '').replace("'", '').strip()
                    
                    print(f"  Raw key from .env: {repr(raw_key)}")
                    print(f"  Cleaned key: {repr(cleaned_key)}")
                    print(f"  Key length: {len(cleaned_key)}")
                    
                    if len(cleaned_key) < 30:
                        print(f"  ⚠️ Warning: Key seems too short!")
                    
                    return cleaned_key
    except FileNotFoundError:
        print("✗ .env file not found")
    except Exception as e:
        print(f"✗ Error reading .env: {e}")
    
    # Try 2: Use load_dotenv as fallback
    print("\nTrying load_dotenv()...")
    try:
        # Clear any existing GEMINI/API environment variables
        for key in list(os.environ.keys()):
            if 'GEMINI' in key or 'API_KEY' in key:
                del os.environ[key]
        
        # Force reload
        load_dotenv(override=True)
        
        env_key = os.environ.get('GEMINI_API_KEY')
        if env_key:
            print(f"✓ Key from load_dotenv: {repr(env_key[:20])}...")
            print(f"  Length: {len(env_key)}")
            return env_key
    except Exception as e:
        print(f"✗ Error with load_dotenv: {e}")
    
    # Try 3: Check for other possible variable names
    print("\nChecking alternative variable names...")
    possible_names = [
        'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'API_KEY', 
        'GEMINI_KEY', 'GOOGLE_AI_KEY', 'AI_API_KEY'
    ]
    
    for name in possible_names:
        value = os.environ.get(name)
        if value:
            print(f"✓ Found in {name}: {repr(value[:20])}...")
            return value
    
    print("\n❌ NO API KEY FOUND!")
    print("="*60)
    raise ValueError("""
⚠️ GEMINI_API_KEY not found! 
Please ensure:
1. You have a .env file in the same directory as app.py
2. It contains: GEMINI_API_KEY=your_actual_key_here
3. No quotes around the key
4. No extra spaces before/after =

Example .env file content:
GEMINI_API_KEY=AIzaSyABC123yourkeyhere
SECRET_KEY=your_flask_secret
""")

# Get the API key
GEMINI_API_KEY = get_api_key()

# Validate the key looks right
if not GEMINI_API_KEY.startswith('AIza'):
    print(f"⚠️ Warning: Key doesn't start with 'AIza' (starts with: {repr(GEMINI_API_KEY[:4])})")

print(f"\n✅ Final API key to use: {repr(GEMINI_API_KEY[:15])}...")
print(f"   Total length: {len(GEMINI_API_KEY)} characters")
print("="*60 + "\n")

# Load OpenWeather API Key
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
if not OPENWEATHER_API_KEY:
    print("⚠️ Warning: OPENWEATHER_API_KEY not found in .env file")
    print("Water Advisor feature will use default/fallback weather data")

# Configure Gemini with error handling
try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Test the configuration
    print("Testing Gemini API connection...")
    model = genai.GenerativeModel("gemini-2.5-flash")
    test_response = model.generate_content("Say 'Hello World' to test connection")
    print(f"✅ Gemini configured successfully: {test_response.text[:50]}...")
except Exception as e:
    print(f"❌ Gemini configuration failed: {e}")
    print("\nCommon issues:")
    print("1. API key is invalid or expired")
    print("2. API key doesn't have proper permissions")
    print("3. Internet connection issue")
    print("4. Google AI Studio API quota exceeded")
    raise

# ================================================
# FLASK APP INITIALIZATION
# ================================================

app = Flask(__name__)
app.url_map.strict_slashes = False

# SECURITY: Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Upload folder for images
UPLOAD_FOLDER = 'static/uploads'

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================================================
# AUTHENTICATION DECORATORS
# ================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth_page'))
        return f(*args, **kwargs)
    return decorated_function

# ================================================
# FIREBASE HELPER FUNCTIONS
# ================================================

def get_user_from_db(user_id):
    """Get user data from Firestore"""
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            return user_doc.to_dict()
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def create_user_in_db(user_id, email, name, password_hash):
    """Create a new user in Firestore"""
    try:
        user_data = {
            'user_id': user_id,
            'email': email,
            'name': name,
            'password_hash': password_hash,
            'created_at': datetime.now().isoformat(),
            'last_login': datetime.now().isoformat()
        }
        db.collection('users').document(user_id).set(user_data)
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

def update_last_login(user_id):
    """Update user's last login timestamp"""
    try:
        db.collection('users').document(user_id).update({
            'last_login': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error updating last login: {e}")

def save_history_to_db(user_id, history_entry):
    """Save analysis history to Firestore"""
    try:
        # Add user_id to the history entry
        history_entry['user_id'] = user_id
        
        # Create a new document in the user's history subcollection
        history_ref = db.collection('users').document(user_id).collection('history')
        doc_ref = history_ref.add(history_entry)
        
        # Return the document ID
        return doc_ref[1].id
    except Exception as e:
        print(f"Error saving history: {e}")
        return None

def get_user_history(user_id):
    """Get all history entries for a user from Firestore"""
    try:
        history_ref = db.collection('users').document(user_id).collection('history')
        history_docs = history_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        history_list = []
        for doc in history_docs:
            history_data = doc.to_dict()
            history_data['id'] = doc.id  # Add document ID
            history_list.append(history_data)
        
        return history_list
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

def delete_history_item(user_id, history_id):
    """Delete a specific history item from Firestore"""
    try:
        db.collection('users').document(user_id).collection('history').document(history_id).delete()
        return True
    except Exception as e:
        print(f"Error deleting history item: {e}")
        return False

def clear_user_history(user_id):
    """Clear all history for a user"""
    try:
        history_ref = db.collection('users').document(user_id).collection('history')
        docs = history_ref.stream()
        
        for doc in docs:
            doc.reference.delete()
        
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False

# ================================================
# UTILITY FUNCTIONS
# ================================================

def save_image(file, analysis_id):
    """Save uploaded image to static folder"""
    try:
        filename = f"plant_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Convert to RGB if needed and save
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(filepath, 'JPEG', quality=85)
        
        return f'/static/uploads/{filename}'
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def format_response_enhanced(text):
    """Format AI response with proper HTML markup and structured styling"""
    if not text:
        return text
    
    # Clean up markdown formatting
    text = text.replace('**', '')
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        
        if line == '':
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append('<div class="response-spacer"></div>')
            continue
        
        # Detect main headers (ALL CAPS or ends with colon and short)
        if (line.isupper() and len(line) < 60) or (line.endswith(':') and len(line) < 60 and line.count(':') == 1):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            
            # Add icon based on header content
            icon = ''
            if 'IDENTIFICATION' in line or 'PLANT' in line:
                icon = '🌿'
            elif 'HEALTH' in line or 'STATUS' in line:
                icon = '💚'
            elif 'DISEASE' in line or 'PROBLEM' in line:
                icon = '🦠'
            elif 'SEVERITY' in line:
                icon = '⚠️'
            elif 'SYMPTOMS' in line:
                icon = '🔍'
            elif 'CAUSES' in line:
                icon = '🎯'
            elif 'TREATMENT' in line or 'RECOMMENDATIONS' in line:
                icon = '💊'
            elif 'PREVENTION' in line:
                icon = '🛡️'
            elif 'PROGNOSIS' in line:
                icon = '📊'
            elif 'NOTES' in line or 'ADDITIONAL' in line:
                icon = '📝'
            elif 'CARE' in line:
                icon = '🌱'
            elif 'CLASSIFICATION' in line:
                icon = '📋'
            elif 'CHARACTERISTICS' in line:
                icon = '✨'
            elif 'TOXICITY' in line:
                icon = '⚠️'
            elif 'PROPAGATION' in line:
                icon = '🌱'
            
            formatted_lines.append(f'<div class="response-header"><span class="header-icon">{icon}</span> {line}</div>')
        
        # Sub-headers (contains : in middle)
        elif ':' in line and not line.endswith(':'):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            parts = line.split(':', 1)
            formatted_lines.append(f'<div class="response-subheader"><span class="label">{parts[0]}:</span> <span class="value">{parts[1]}</span></div>')
        
        # Bullet points
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            content = line[1:].strip()
            if not in_list:
                formatted_lines.append('<ul class="response-list">')
                in_list = True
            formatted_lines.append(f'<li class="response-bullet">{content}</li>')
        
        # Numbered lists
        elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<div class="response-numbered">{line}</div>')
        
        # Regular content
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<div class="response-content">{line}</div>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return '\n'.join(formatted_lines)
def extract_plant_metadata(analysis_text):
    """
    Extract plant condition and severity from Gemini diagnosis analysis.
    Uses keyword matching and pattern recognition.
    
    Returns:
        dict with 'condition' and 'severity' fields
    """
    analysis_lower = analysis_text.lower()
    
    # Default values
    condition = "Unknown"
    severity = "Medium"
    
    # Condition detection (prioritize specific diseases over general states)
    conditions_map = {
        'bacterial blight': ['bacterial blight', 'blight bacteria'],
        'fungal infection': ['fungal', 'fungus', 'mold', 'mildew'],
        'pest infestation': ['pest', 'insect', 'aphid', 'mite', 'caterpillar'],
        'nutrient deficiency': ['deficiency', 'nutrient', 'nitrogen', 'phosphorus', 'potassium'],
        'overwatering': ['overwater', 'waterlogged', 'root rot'],
        'underwatering': ['underwater', 'drought', 'dehydrat'],
        'sunburn': ['sunburn', 'sun scorch', 'heat stress'],
        'healthy': ['healthy', 'good condition', 'thriving', 'no disease']
    }
    
    for cond_name, keywords in conditions_map.items():
        if any(keyword in analysis_lower for keyword in keywords):
            condition = cond_name
            break
    
    # Severity detection
    severity_keywords = {
        'High': ['severe', 'critical', 'emergency', 'dying', 'advanced stage', 'extensive damage'],
        'Medium': ['moderate', 'developing', 'spreading', 'progressing', 'noticeable'],
        'Low': ['mild', 'early stage', 'minor', 'slight', 'beginning', 'healthy']
    }
    
    for sev_level, keywords in severity_keywords.items():
        if any(keyword in analysis_lower for keyword in keywords):
            severity = sev_level
            break
    
    return {
        'condition': condition,
        'severity': severity
    }

# ================================================
# OPENWEATHER API INTEGRATION
# ================================================

def fetch_weather_data(city='Pune'):
    """
    Fetch real-time weather data from OpenWeather API.
    
    Parameters:
        city (str): City name for weather lookup
    
    Returns:
        dict with temperature, humidity, rain status, and description
    """
    if not OPENWEATHER_API_KEY:
        # Fallback to default values if API key not available
        print("⚠️ OpenWeather API key not found, using default weather data")
        return {
            'temperature': 28.0,
            'humidity': 60,
            'rain': False,
            'description': 'Clear sky (default data)',
            'city': city,
            'error': 'API key not configured'
        }
    
    try:
        # OpenWeather Current Weather API endpoint
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'  # Celsius
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant weather data
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        
        # Check if it's raining
        rain = False
        if 'rain' in data:
            rain = True
        elif 'weather' in data:
            # Check weather condition codes for rain
            weather_id = data['weather'][0]['id']
            # Rain codes: 2xx (Thunderstorm), 3xx (Drizzle), 5xx (Rain)
            if 200 <= weather_id < 600:
                rain = True
        
        return {
            'temperature': round(temperature, 1),
            'humidity': humidity,
            'rain': rain,
            'description': description.capitalize(),
            'city': city
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        # Return fallback data on error
        return {
            'temperature': 28.0,
            'humidity': 60,
            'rain': False,
            'description': 'Data unavailable',
            'city': city,
            'error': str(e)
        }

# ================================================
# AOA: DECISION TREE ALGORITHM
# ================================================

def decision_tree(temp, humidity, rain, severity):
    """
    Decision Tree Algorithm for watering decision.
    Time Complexity: O(1) - constant time, fixed tree height
    
    Parameters:
        temp: Temperature in Celsius
        humidity: Humidity percentage
        rain: Boolean indicating rain
        severity: Plant severity level (High/Medium/Low)
    
    Returns:
        str: One of SKIP, WATER_LIGHT, WATER_MODERATE, WATER_THOROUGH
    """
    if rain:
        return "SKIP"
    elif temp > 35 and severity == "High":
        return "WATER_THOROUGH"
    elif temp > 30:
        return "WATER_MODERATE"
    elif humidity > 75:
        return "WATER_LIGHT"
    else:
        return "SKIP"

# ================================================
# OS: PROCESS SCHEDULING
# ================================================

# Priority mapping: Lower number = Higher priority
SEVERITY_PRIORITY = {
    'High': 1,
    'Medium': 2,
    'Low': 3
}

def priority_scheduler(process_list):
    """
    Priority Scheduling Algorithm
    Time Complexity: O(n log n) - due to sorting
    
    Parameters:
        process_list: List of process dictionaries with 'priority' key
    
    Returns:
        Sorted list by priority (ascending)
    """
    return sorted(process_list, key=lambda x: x["priority"])

def create_plant_process(history_id, severity, timestamp, condition):
    """
    Create a process structure for a plant.
    
    Parameters:
        history_id: Unique identifier for the plant diagnosis
        severity: Severity level (High/Medium/Low)
        timestamp: Analysis timestamp
        condition: Plant condition
    
    Returns:
        dict: Process structure
    """
    # Calculate burst time (watering urgency in minutes)
    burst_time_map = {
        'High': 10,
        'Medium': 6,
        'Low': 3
    }
    
    return {
        "pid": history_id,
        "severity": severity,
        "arrival_time": timestamp,
        "burst_time": burst_time_map.get(severity, 6),
        "priority": SEVERITY_PRIORITY.get(severity, 2),
        "condition": condition
    }

# ================================================
# ROUTES - PAGES
# ================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth')
def auth_page():
    # Redirect to home if already logged in
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('auth.html')

@app.route('/diagnosis')
@login_required
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/plant-identifier')
@login_required
def plant_identifier():
    return render_template('plant_identifier.html')

@app.route('/room-preview')
@login_required
def room_preview():
    return render_template('room_preview.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

# ================================================
# ROUTES - AUTHENTICATION API
# ================================================

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([name, email, password]):
            return render_template(
            "auth.html",
            error="All fields are required"
            )
        
        # Check if user already exists
        users_ref = db.collection('users')
        existing_user = users_ref.where('email', '==', email).limit(1).stream()
        
        if len(list(existing_user)) > 0:
            return render_template(
            "auth.html",
            error="Email already registered"
            )
        
        # Generate user ID and hash password
        user_id = f"user_{datetime.now().timestamp()}"
        password_hash = generate_password_hash(password)
        
        # Create user in Firestore
        if create_user_in_db(user_id, email, name, password_hash):
            # Set session
            session['user_id'] = user_id
            session['user_name'] = name
            session['user_email'] = email
            
            return redirect(url_for('index'))
        else:
            return render_template(
            "auth.html",
            error="Failed to create account"
            )
    
    except Exception as e:
        print(f"Signup error: {e}")
        return render_template(
            "auth.html",
            error=str(e)
            )

@app.route('/api/login', methods=['POST'])
def login():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not all([email, password]):
            return render_template(
            "auth.html",
            error="Email and password are required"
            )
        
        # Find user by email
        users_ref = db.collection('users')
        users_query = users_ref.where('email', '==', email).limit(1).stream()
        
        user_doc = None
        for doc in users_query:
            user_doc = doc
            break
        
        if not user_doc:
            return render_template(
            "auth.html",
            error="Invalid email or password"
            )
        
        user_data = user_doc.to_dict()
        
        # Verify password
        if not check_password_hash(user_data['password_hash'], password):
            return render_template(
            "auth.html",
            error="Invalid email or password"
            )
        
        # Update last login
        update_last_login(user_doc.id)
        
        # Set session
        session['user_id'] = user_doc.id
        session['user_name'] = user_data['name']
        session['user_email'] = user_data['email']
        
        return redirect(url_for('index'))
    
    except Exception as e:
        print(f"Login error: {e}")
        return render_template(
            "auth.html",
            error= str(e)
            )

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return redirect(url_for('auth_page'))

@app.route('/logout')
def logout():
    """Handle user logout via GET request"""
    session.clear()
    return redirect(url_for('index'))

# ================================================
# ROUTES - ANALYSIS API
# ================================================

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_plant():
    """
    Non-blocking plant diagnosis endpoint.

    OS concept — why this matters:
      Before threading: this route took 5–15 s (blocked on Gemini API).
      The Flask server thread was tied up → no other requests could be
      handled → the user's browser spun and they couldn't navigate.

      After threading:
        Main thread  → validates input, saves image, creates job, starts
                        background thread, returns job_id in <100 ms.
        Worker thread → does the slow AI + DB work independently.

      The user gets a job_id immediately and can navigate freely.
      The frontend polls GET /api/job/<job_id> every 3 seconds until done.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read image bytes BEFORE starting the thread.
        # The `file` object is tied to this HTTP request; once the response
        # is sent the file handle may close. Passing raw bytes to the thread
        # is the safe, correct approach.
        file.seek(0)
        image_bytes = file.read()

        # Save the image on the main thread so we have a path immediately.
        user_id     = session['user_id']
        analysis_id = f"{user_id}_{datetime.now().timestamp()}"
        file.seek(0)
        image_path  = save_image(file, analysis_id)

        # ── Create a unique job ID ────────────────────────────────────────────
        job_id = f"diag_{analysis_id}"

        # ── Register the job as 'pending' BEFORE starting the thread ─────────
        # (If we registered after start, the frontend might poll before
        #  the entry exists and get a 404.)
        with _jobs_lock:
            diagnosis_jobs[job_id] = {'status': 'pending'}

        # ── Create and start the background thread ────────────────────────────
        #
        # threading.Thread():
        #   target = the function to run in the new thread
        #   args   = positional arguments passed to that function
        #   daemon = True  → the thread will NOT prevent the server from
        #            shutting down if it is still running (important for
        #            clean exits)
        #   name   = human-readable label (shows up in log prints)
        #
        # thread.start() → OS creates the thread and schedules it.
        # The current (Flask) thread returns immediately after start().
        # ──────────────────────────────────────────────────────────────────────
        thread = threading.Thread(
            target = _diagnosis_worker,
            args   = (job_id, user_id, image_bytes, image_path, 'diagnosis'),
            daemon = True,
            name   = f"DiagnosisWorker-{job_id[-8:]}"
        )
        thread.start()

        print(f"[Main thread] Started background thread '{thread.name}' for job {job_id}")

        # Return the job_id immediately — frontend will poll for the result.
        return jsonify({
            'success':    True,
            'job_id':     job_id,
            'status':     'pending',
            'message':    'Diagnosis started in background. You can navigate freely — it will appear in History when done.',
            'image_path': image_path,
        })

    except Exception as e:
        print(f"Error in analyze_plant: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/identify', methods=['POST'])
@login_required
def identify_plant():
    """
    Non-blocking plant identification endpoint.
    Same threading pattern as analyze_plant — see comments there.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        file.seek(0)
        image_bytes = file.read()

        user_id     = session['user_id']
        analysis_id = f"{user_id}_{datetime.now().timestamp()}"
        file.seek(0)
        image_path  = save_image(file, analysis_id)

        job_id = f"ident_{analysis_id}"

        with _jobs_lock:
            identify_jobs[job_id] = {'status': 'pending'}

        thread = threading.Thread(
            target = _diagnosis_worker,
            args   = (job_id, user_id, image_bytes, image_path, 'identification'),
            daemon = True,
            name   = f"IdentifyWorker-{job_id[-8:]}"
        )
        thread.start()

        print(f"[Main thread] Started background thread '{thread.name}' for job {job_id}")

        return jsonify({
            'success':    True,
            'job_id':     job_id,
            'status':     'pending',
            'message':    'Identification started in background. You can navigate freely — it will appear in History when done.',
            'image_path': image_path,
        })

    except Exception as e:
        print(f"Error in identify_plant: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_id>', methods=['GET'])
@login_required
def get_job_status(job_id):
    """
    Polling endpoint — frontend calls this every 3 s to check job progress.

    Returns:
      { status: 'pending' }                         — still running
      { status: 'done',  result: { ... } }          — finished
      { status: 'error', error:  'message' }         — something failed
      404 if job_id is unknown
    """
    # Check both job stores (prefix tells us which one)
    if job_id.startswith('diag_'):
        with _jobs_lock:
            job = diagnosis_jobs.get(job_id)
    else:
        with _jobs_lock:
            job = identify_jobs.get(job_id)

    if job is None:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({'job_id': job_id, **job})


@app.route('/api/job/<job_id>', methods=['DELETE'])
@login_required
def cleanup_job(job_id):
    """
    Optional: let the frontend clean up a finished job from memory
    once it has consumed the result.
    """
    removed = False
    with _jobs_lock:
        if job_id in diagnosis_jobs:
            del diagnosis_jobs[job_id]; removed = True
        elif job_id in identify_jobs:
            del identify_jobs[job_id]; removed = True

    return jsonify({'deleted': removed})


@app.route('/api/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('analysis', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        prompt = f"""Based on this plant analysis:

{context}

User's question: {question}

Provide a clear, helpful answer. Structure your response with proper formatting."""
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        formatted_answer = format_response_enhanced(response.text)
        
        return jsonify({
            'success': True,
            'answer': response.text,
            'formatted_answer': formatted_answer
        })
    
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ================================================
# ROUTES - HISTORY API
# ================================================

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    try:
        user_id = session['user_id']
        history = get_user_history(user_id)
        
        # Format each history entry if needed
        for item in history:
            if 'formatted_analysis' not in item:
                item['formatted_analysis'] = format_response_enhanced(item.get('analysis', ''))
        
        return jsonify(history)
    except Exception as e:
        print(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    try:
        user_id = session['user_id']
        
        # Get the history item to find the image path
        history_ref = db.collection('users').document(user_id).collection('history').document(history_id)
        history_doc = history_ref.get()
        
        if history_doc.exists:
            history_data = history_doc.to_dict()
            
            # Delete associated image if exists
            if 'image_path' in history_data:
                image_path = history_data['image_path'].replace('/static/', 'static/')
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        print(f"Error deleting image: {e}")
        
        # Delete from Firestore
        delete_history_item(user_id, history_id)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    try:
        user_id = session['user_id']
        
        # Use batch delete for Firestore (more efficient)
        history_ref = db.collection('users').document(user_id).collection('history')
        batch = db.batch()
        
        # Get all history items
        history_docs = history_ref.stream()
        image_paths = []
        
        # Prepare batch delete and collect image paths
        for doc in history_docs:
            batch.delete(doc.reference)
            doc_data = doc.to_dict()
            if 'image_path' in doc_data:
                image_paths.append(doc_data['image_path'])
        
        # Commit batch delete (much faster than individual deletes)
        batch.commit()
        
        # Delete images in background (don't block response)
        # This happens after response is sent
        for image_path in image_paths:
            try:
                full_path = image_path.replace('/static/', 'static/')
                if os.path.exists(full_path):
                    os.remove(full_path)
            except Exception as e:
                print(f"Error deleting image {image_path}: {e}")
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

# ================================================
# AOA: MERGE SORT — Sort Plants by Health Score
# ================================================
#
# Merge Sort is a Divide-and-Conquer algorithm.
#
# How it works:
#   1. DIVIDE  — split the list in half recursively until every
#                sub-list has only 1 element (trivially sorted).
#   2. CONQUER — merge adjacent sorted sub-lists by comparing the
#                front element of each and always taking the larger
#                health_score first (descending order).
#   3. COMBINE — the merge bubbles results back up the recursion tree
#                producing a fully sorted list.
#
# Time Complexity : O(n log n)  — log n recursive levels × O(n) merge work
# Space Complexity: O(n)        — auxiliary space for merged sub-lists
# ================================================

def merge_sort_plants(plants):
    """
    Sort a list of plant/history records by health_score DESCENDING.
    Uses Merge Sort (Divide and Conquer) — O(n log n).
    """
    # BASE CASE: a list of 0 or 1 element is already sorted
    if len(plants) <= 1:
        return plants

    # DIVIDE: find midpoint and split
    mid        = len(plants) // 2
    left_half  = merge_sort_plants(plants[:mid])   # recurse left
    right_half = merge_sort_plants(plants[mid:])   # recurse right

    # CONQUER: merge the two sorted halves
    return _merge(left_half, right_half)


def _merge(left, right):
    """
    Merge two sorted sub-lists into one sorted list (descending health_score).
    O(n) — the merge step at each recursion level.
    """
    merged = []
    i = j  = 0

    while i < len(left) and j < len(right):
        # Descending order → pick the HIGHER score first
        if left[i]['health_score'] >= right[j]['health_score']:
            merged.append(left[i]);  i += 1
        else:
            merged.append(right[j]); j += 1

    # Append any remaining elements (already sorted)
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def calculate_health_score(plant):
    """
    Calculate a numeric health score (0–100) for a plant record.

    Scoring logic:
        Base score = 100 (assume perfect health).
        Deduct based on severity:
            High   → −55   (seriously ill)
            Medium → −30   (moderately ill)
            Low    → −10   (minor issue)
        Healthy condition → +5 bonus (capped at 100).
        Identification records (no diagnosis) default to 70.
    """
    if plant.get('type') != 'diagnosis':
        return 70  # identification entries → neutral mid-score

    severity  = (plant.get('severity') or '').strip().lower()
    condition = (plant.get('condition') or '').strip().lower()

    score = 100
    deductions = {'high': 55, 'medium': 30, 'low': 10}
    score -= deductions.get(severity, 20)

    if 'healthy' in condition:
        score = min(100, score + 5)

    return max(0, min(100, score))


# ── Route: page ─────────────────────────────────────────────────────────────
@app.route('/sorted_plants')
@login_required
def sorted_plants_page():
    return render_template('history.html')


# ── Route: API ───────────────────────────────────────────────────────────────
@app.route('/api/sorted_plants', methods=['GET'])
@login_required
def sorted_plants_api():
    """
    Fetch all plant records → compute health_score → Merge Sort → return JSON.

    Steps:
      1. Fetch all history records from Firestore
      2. Attach health_score to every record   (calculate_health_score)
      3. Sort using Merge Sort                  (O(n log n) D&C)
      4. Return sorted JSON list
    """
    try:
        user_id     = session['user_id']
        all_history = get_user_history(user_id)

        if not all_history:
            return jsonify({'sorted_plants': [], 'total': 0,
                            'algorithm': 'Merge Sort — O(n log n) Divide and Conquer'})

        # Attach computed health score to every record
        for plant in all_history:
            plant['health_score'] = calculate_health_score(plant)

        # Run Merge Sort (Divide and Conquer, O(n log n))
        sorted_list = merge_sort_plants(all_history)

        return jsonify({
            'sorted_plants': sorted_list,
            'total':         len(sorted_list),
            'algorithm':     'Merge Sort — O(n log n) Divide and Conquer'
        })

    except Exception as e:
        print(f"Error in sorted_plants_api: {e}")
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# CARE PLAN BUILDER  (used by water_advisor_api)
# ════════════════════════════════════════════════════════════════════════════════

def _build_care_plan(decision, severity, condition, temp, humidity, rain):
    """
    Return an ordered list of personalised step-by-step care actions
    based on the Decision Tree output + current weather + plant condition.
    Each step has: icon, title, detail.
    """
    steps = []
    sev = (severity or 'Low').strip().title()

    # ── Step 1: Watering action ───────────────────────────────────────────────
    watering = {
        'SKIP':           ('✅', 'Skip Watering Today',
                           'It is raining or humidity is very high — no watering needed. '
                           'Check that drainage holes are clear.'),
        'WATER_LIGHT':    ('💧', 'Water Lightly',
                           'Gently moisten the top 2–3 cm of soil. '
                           'Avoid soaking — let soil breathe.'),
        'WATER_MODERATE': ('💦', 'Water Moderately',
                           'Water until the soil is evenly moist throughout. '
                           'Let excess drain from the bottom.'),
        'WATER_THOROUGH': ('🌊', 'Water Thoroughly',
                           f'High evaporation today ({temp}°C). '
                           'Water deeply — continue until water drains freely from the bottom. '
                           'Repeat evening if soil feels dry.'),
    }
    icon, title, detail = watering.get(decision, watering['WATER_LIGHT'])
    steps.append({'icon': icon, 'title': title, 'detail': detail})

    # ── Step 2: Heat warning ──────────────────────────────────────────────────
    if temp >= 32:
        steps.append({
            'icon':   '🌡️',
            'title':  'Shield from Harsh Sun',
            'detail': f'Temperature is {temp}°C — move the plant to partial shade '
                      'during peak afternoon hours (12 pm – 4 pm) to prevent leaf scorch.'
        })

    # ── Step 3: Humidity advice ───────────────────────────────────────────────
    if humidity > 75:
        steps.append({
            'icon':   '💨',
            'title':  'Improve Air Circulation',
            'detail': f'Humidity is {humidity}% — high moisture encourages fungal growth. '
                      'Open a window or use a fan to keep air moving around the plant.'
        })
    elif humidity < 40:
        steps.append({
            'icon':   '🌫️',
            'title':  'Mist Leaves Lightly',
            'detail': f'Air is very dry ({humidity}% humidity). Lightly mist the foliage '
                      'in the morning to maintain adequate leaf moisture.'
        })

    # ── Step 4: Disease action ────────────────────────────────────────────────
    if sev in ('High', 'Medium') and condition and condition.lower() not in ('', 'unknown', 'healthy'):
        cond_display = condition.replace('_', ' ').title()
        if sev == 'High':
            steps.append({
                'icon':   '💊',
                'title':  f'Treat {cond_display} Today',
                'detail': f'High severity detected. Apply recommended treatment '
                          '(fungicide / pesticide) and remove visibly affected leaves. '
                          'Isolate plant from others if possible.'
            })
        else:
            steps.append({
                'icon':   '🔍',
                'title':  f'Monitor {cond_display}',
                'detail': 'Check for new spots, discolouration, or spreading symptoms. '
                          'Apply treatment if condition worsens.'
            })

    # ── Step 5: Soil check ────────────────────────────────────────────────────
    steps.append({
        'icon':   '🌱',
        'title':  'Check Soil Condition',
        'detail': 'Push a finger 2 cm into the soil. Dark and cool → leave it. '
                  'Pale and crumbly → water now. Check that soil is not compacted.'
    })

    return steps


# ════════════════════════════════════════════════════════════════════════════════
# AOA — GREEDY ALGORITHM: Plant Care Checklist
# ════════════════════════════════════════════════════════════════════════════════
#
# What is a Greedy Algorithm?
#   A Greedy Algorithm builds a solution step-by-step by ALWAYS choosing the
#   locally optimal choice at each step — the one with the highest immediate
#   benefit — without reconsidering previous decisions.
#
# Why is it suitable here?
#   Plant care tasks are independent of each other. Doing "Water Plant" first
#   does not change the priority of "Check Leaves". So the greedy choice
#   (pick highest priority task next) is globally optimal — you always finish
#   the most urgent care first.
#
# Algorithm steps (matches the UI explanation):
#   1. BUILD   — create a flat list of candidate tasks
#   2. SCORE   — assign each task a numeric priority based on context
#                (weather + severity + condition)
#   3. FILTER  — drop tasks that are not relevant today (e.g. skip "Water"
#                when it is raining)
#   4. SORT    — sort descending by priority          ← the greedy step
#   5. SELECT  — take the top N tasks for display
#
# Time Complexity : O(k log k)   k = number of candidate tasks (always ≤ 10,
#                                so effectively O(1) in practice)
# Space Complexity: O(k)
# ════════════════════════════════════════════════════════════════════════════════

# ── Priority constants (easy to tune / explain in viva) ─────────────────────
_P_SOIL_DRY        = 10   # Highest — dry soil is immediately damaging
_P_DISEASE_CHECK   =  9   # Active disease needs daily monitoring
_P_WATER_THOROUGH  =  8   # Heavy watering day
_P_WATER_MODERATE  =  7   # Normal watering
_P_WATER_LIGHT     =  6   # Light moisture top-up
_P_TREAT_DISEASE   =  8   # Apply treatment when disease is High severity
_P_PRUNE_DEAD      =  7   # Remove dead/spotted leaves
_P_HUMIDITY_CHECK  =  5   # Check humidity when plant is sensitive
_P_FERTILIZE       =  5   # Routine fertilising
_P_SOIL_CHECK      =  4   # General soil inspection
_P_SUNLIGHT_CHECK  =  3   # Verify the plant is getting enough light
_P_LOG_PROGRESS    =  2   # Lowest — record-keeping


def _make_task(icon, title, detail, priority, tag):
    """
    Create a single task dict.
    'tag' is a short machine-readable string used by the frontend to
    render tick-box state independently of the display title.
    """
    return {
        'icon':     icon,
        'title':    title,
        'detail':   detail,
        'priority': priority,
        'tag':      tag,
    }


def greedy_care_checklist(temp, humidity, rain, severity, condition, plant_name):
    """
    Build a priority-sorted Plant Care Checklist using a Greedy Algorithm.

    Parameters
    ----------
    temp       : float  — current temperature in °C   (from OpenWeather)
    humidity   : int    — current humidity %           (from OpenWeather)
    rain       : bool   — is it raining right now?    (from OpenWeather)
    severity   : str    — 'High' | 'Medium' | 'Low'   (from Firestore)
    condition  : str    — plant condition / disease name
    plant_name : str    — display name shown in checklist items

    Returns
    -------
    list of task dicts, sorted descending by priority (the greedy ordering)
    """

    # Normalise inputs
    sev  = (severity  or 'Low').strip().title()       # 'High' | 'Medium' | 'Low'
    cond = (condition or 'Unknown').strip().lower()
    name = (plant_name or 'Plant').strip().title()

    is_diseased  = sev in ('High', 'Medium') and cond not in ('', 'unknown', 'healthy')
    is_hot       = temp >= 32
    is_dry_air   = humidity < 45
    is_very_dry  = is_hot and is_dry_air               # worst combination
    healthy_cond = 'healthy' in cond

    # ── STEP 1 & 2: BUILD candidate task pool and SCORE each task ────────────

    candidates = []

    # ── Watering tasks (mutually exclusive — only one watering level added) ──
    if rain:
        # Skip watering entirely when raining; check drainage instead
        candidates.append(_make_task(
            '🌧️', f'Check Drainage for {name}',
            'It is raining — ensure the pot has proper drainage to avoid waterlogging.',
            _P_WATER_LIGHT, 'drainage'
        ))
    elif is_very_dry:
        candidates.append(_make_task(
            '🌊', f'Water {name} Thoroughly',
            f'Temp {temp}°C + humidity {humidity}% — soil loses moisture fast. '
            'Water deeply until it drains from the bottom.',
            _P_SOIL_DRY, 'water_thorough'
        ))
    elif is_hot:
        candidates.append(_make_task(
            '💦', f'Water {name} Moderately',
            f'Hot day ({temp}°C). Moderate watering to compensate for evaporation.',
            _P_WATER_MODERATE, 'water_moderate'
        ))
    elif is_dry_air:
        candidates.append(_make_task(
            '💧', f'Water {name} Lightly',
            f'Dry air ({humidity}% humidity) — light watering to keep soil surface moist.',
            _P_WATER_LIGHT, 'water_light'
        ))
    else:
        candidates.append(_make_task(
            '💧', f'Water {name} as Normal',
            'Weather conditions are stable. Follow your regular watering schedule.',
            _P_WATER_LIGHT, 'water_normal'
        ))

    # ── Disease / health tasks ────────────────────────────────────────────────
    if is_diseased:
        candidates.append(_make_task(
            '🔍', f'Check {name} Leaves for Disease',
            f'Diagnosed: {condition.title()}. Inspect all leaves for new spots, '
            'discolouration, or spreading symptoms.',
            _P_DISEASE_CHECK, 'disease_check'
        ))
        if sev == 'High':
            candidates.append(_make_task(
                '💊', f'Apply Treatment to {name}',
                f'High-severity condition detected ({condition.title()}). '
                'Apply recommended fungicide/pesticide as per diagnosis.',
                _P_TREAT_DISEASE, 'treat_disease'
            ))
            candidates.append(_make_task(
                '✂️', f'Prune Affected Leaves on {name}',
                'Remove visibly diseased or dead leaves to stop spread and '
                'improve airflow around the plant.',
                _P_PRUNE_DEAD, 'prune_leaves'
            ))
        elif sev == 'Medium':
            candidates.append(_make_task(
                '✂️', f'Remove Damaged Leaves from {name}',
                'Trim any yellowing or spotted leaves to slow the progression '
                'of the detected condition.',
                _P_PRUNE_DEAD - 1, 'prune_leaves'
            ))
    else:
        # Healthy plant — routine leaf inspection is lower priority
        candidates.append(_make_task(
            '🔍', f'Inspect {name} Leaves',
            'Plant appears healthy. Quick visual check to catch any early signs '
            'of pests or disease before they spread.',
            _P_SOIL_CHECK, 'leaf_inspect'
        ))

    # ── Soil tasks ────────────────────────────────────────────────────────────
    candidates.append(_make_task(
        '🌱', f'Check Soil Moisture for {name}',
        'Press a finger 2 cm into the soil. If it feels dry, water immediately; '
        'if damp, skip until tomorrow.',
        _P_HUMIDITY_CHECK if is_diseased else _P_SOIL_CHECK,
        'soil_check'
    ))

    # ── Fertilising ──────────────────────────────────────────────────────────
    # Don't fertilise a severely diseased plant — it can worsen stress
    if sev != 'High':
        fert_detail = (
            'Healthy plant — apply balanced liquid fertiliser (NPK 10-10-10) '
            'once a week to support growth.'
            if healthy_cond else
            f'Use a recovery fertiliser (low N, higher P & K) to strengthen '
            f'{name} while it recovers from {condition.title()}.'
        )
        candidates.append(_make_task(
            '🌿', f'Fertilize {name}',
            fert_detail,
            _P_FERTILIZE, 'fertilize'
        ))

    # ── Environmental checks ──────────────────────────────────────────────────
    if humidity > 75:
        candidates.append(_make_task(
            '💨', f'Improve Air Circulation Around {name}',
            f'Humidity is {humidity}% — high moisture promotes fungal growth. '
            'Move plant to a spot with better airflow or use a fan.',
            _P_HUMIDITY_CHECK, 'air_circulation'
        ))

    if is_hot:
        candidates.append(_make_task(
            '☀️', f'Check {name} Sun Exposure',
            f'Temperature is {temp}°C. Make sure the plant is not in direct harsh '
            'afternoon sun — move to partial shade if leaves look scorched.',
            _P_SUNLIGHT_CHECK, 'sun_check'
        ))

    # ── Progress logging ──────────────────────────────────────────────────────
    candidates.append(_make_task(
        '📝', f'Log Today\'s Condition of {name}',
        'Take a quick photo and note any changes. Regular records help track '
        'recovery progress and catch problems early.',
        _P_LOG_PROGRESS, 'log_progress'
    ))

    # ── STEP 3: FILTER — remove tasks that conflict with each other ───────────
    # (already handled above by the if/elif watering logic)

    # ── STEP 4: SORT descending by priority  ← the GREEDY step ──────────────
    # "Always pick the task with the highest local benefit first."
    # Python's sort is Timsort → O(k log k). Since k ≤ 10, this is O(1).
    candidates.sort(key=lambda t: t['priority'], reverse=True)

    # ── STEP 5: SELECT top tasks (cap at 6 to keep checklist readable) ───────
    checklist = candidates[:6]

    return checklist


# ════════════════════════════════════════════════════════════════════════════════
# ROUTES - WATER ADVISOR
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/weather-watering')
@login_required
def water_advisor_page_no_id():
    return render_template('weather_watering.html', history_id='')

@app.route('/weather-watering/<path:history_id>')
@login_required
def water_advisor_page(history_id):
    """Render the water advisor page"""
    return render_template('weather_watering.html', history_id=history_id)

@app.route('/api/water-advisor/<history_id>', methods=['GET'])
@login_required
def water_advisor_api(history_id):
    """
    Water Advisor API - Implements Decision Tree & Process Scheduling
    
    Flow:
        1. Fetch plant diagnosis from Firestore
        2. Extract condition and severity
        3. Get user city from database
        4. Fetch real-time weather using OpenWeather API
        5. Run decision_tree() algorithm
        6. Create plant processes for scheduling
        7. Run priority_scheduler() algorithm
        8. Generate notification if needed
    """
    try:
        user_id = session['user_id']
        
        # Get user data for city
        user_data = get_user_from_db(user_id)
        city = user_data.get('city', 'Mumbai') if user_data else 'Mumbai'
        
        # Fetch the specific history item
        history_ref = db.collection('users').document(user_id).collection('history').document(history_id)
        history_doc = history_ref.get()
        
        if not history_doc.exists:
            return jsonify({'error': 'Diagnosis not found'}), 404
        
        history_data = history_doc.to_dict()
        
        # Only work with diagnosis type
        if history_data.get('type') != 'diagnosis':
            return jsonify({'error': 'Water Advisor only works with plant diagnoses'}), 400
        
        # Extract condition and severity
        condition = history_data.get('condition', '')
        severity = history_data.get('severity', '')
        if not condition or condition == 'Unknown':
            meta = extract_plant_metadata(history_data.get('analysis', ''))
            condition = meta['condition']
            if not severity: severity = meta['severity']
        if not severity: severity = 'Medium'
        timestamp = history_data.get('timestamp', datetime.now().isoformat())
        
        # Fetch real-time weather data
        weather = fetch_weather_data(city)
        
        # Run Decision Tree Algorithm (O(1))
        decision = decision_tree(
            temp=weather['temperature'],
            humidity=weather['humidity'],
            rain=weather['rain'],
            severity=severity
        )
        
        # Map decision to detailed recommendation
        decision_map = {
            "SKIP": {
                "action": "Skip Watering",
                "description": "No watering needed at this time.",
                "icon": "🚫"
            },
            "WATER_LIGHT": {
                "action": "Water Lightly",
                "description": "Light watering recommended - just moisten the soil surface.",
                "icon": "💧"
            },
            "WATER_MODERATE": {
                "action": "Water Moderately",
                "description": "Moderate watering needed - water until soil is moist throughout.",
                "icon": "💦"
            },
            "WATER_THOROUGH": {
                "action": "Water Thoroughly",
                "description": "Thorough watering required - water deeply until it drains from bottom.",
                "icon": "🌊"
            }
        }
        
        recommendation = decision_map.get(decision, decision_map["SKIP"])

        # ── Build Today's Care Plan (step-by-step actions) ───────────────────
        care_plan = _build_care_plan(
            decision  = decision,
            severity  = severity,
            condition = condition,
            temp      = weather['temperature'],
            humidity  = weather['humidity'],
            rain      = weather['rain'],
        )

        # ── Generate urgent notification when needed ───────────────────────────
        notification = None
        if decision in ('WATER_THOROUGH', 'WATER_MODERATE') and severity == 'High':
            notification = {
                'type':     'urgent',
                'message':  f'⚠️ Urgent: {condition.title()} detected with High severity — '
                            'water and treat today!',
                'severity': 'high'
            }
        elif decision == 'WATER_THOROUGH':
            notification = {
                'type':     'warning',
                'message':  '💡 Hot and dry conditions today — water your plant thoroughly '
                            'to prevent stress.',
                'severity': 'warning'
            }

        # ── Greedy Care Checklist (AOA — O(k log k), k ≤ 10 tasks) ──────────
        # greedy_care_checklist() scores every candidate task, sorts them in
        # descending priority order (the greedy choice — always pick highest
        # urgency next), and returns the top 6 tasks for display.
        plant_name = condition.replace('_', ' ').title() if condition else 'Plant'
        checklist  = greedy_care_checklist(
            temp       = weather['temperature'],
            humidity   = weather['humidity'],
            rain       = weather['rain'],
            severity   = severity,
            condition  = condition,
            plant_name = plant_name,
        )

        return jsonify({
            'success':        True,
            'weather':        weather,
            'plant_info': {
                'condition':  condition,
                'severity':   severity,
                'timestamp':  timestamp,
            },
            'recommendation': recommendation,
            'decision_code':  decision,
            'care_plan':      care_plan,
            'checklist':      checklist,   # ← Greedy-sorted task list
            'notification':   notification,
        })
    
    except Exception as e:
        print(f"Error in water_advisor_api: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ================================================
# MAIN
# ================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 PlantCare Pro - Starting...")
    print("="*60)
    print(f"✓ Server: http://localhost:5000")
    print(f"✓ Upload Folder: {UPLOAD_FOLDER}")
    print(f"✓ Firebase: Connected")
    print(f"✓ Environment: {'Production' if not app.debug else 'Development'}")
    print(f"✓ Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)