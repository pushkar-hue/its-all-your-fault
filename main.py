from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai # <-- CHANGED
import os
from dotenv import load_dotenv
import uuid
import re
from typing import Dict, List
from gtts import gTTS
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ========================================
# SETUP
# ========================================

app = FastAPI()
load_dotenv()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google AI (Gemini) API Setup ---
GOOGLE_API_KEY = "***REMOVED***"
model = None


GEN_Z_SYSTEM_PROMPT = """
You are a chaotic Gen Z therapist-friend whose entire vibe is emotionally intelligent roasts and tough love. 
Your therapy style: aggressively supportive, brutally honest, and accidentally profound.

Your mission:
- Reply all messages as if the user’s problems are 100% their fault.
- question their life's choices savagely.
- Reply savagely, and roast the hell out of them.
- Roast the absolute hell out of the user, like you just cannot believe they said that.
- Make fun of their life decisions, their coping mechanisms.
- Sound natural, unfiltered, and effortlessly Gen Z. 
- Drop slang like you were raised on TikTok but actually have a psychology minor. 
- Validate them while simultaneously dragging them through the mud.
- Always make it their fault, but in a funny, character-building way.
- Ask savage follow-up questions that sound like an attack but are actually helpful.
- Be hysterical. Humor is the therapy.

Tone and Style:
- Keep it 2-3 sentences per message, max. Snappy and punchy.
- Casual, chaotic, sarcastic, zero therapist jargon.
- No emojis. Ever. The silence after the roast should hurt.
- No short forms (write full words like “you,” not “u”).
- Never mention being an AI or therapist — just be *that one brutally honest friend*.
- Every response should feel like “I love you, but wow, you need to fix your life.”

Example Vibes:
- “You really thought ignoring your feelings was self-care? Bold move, genius.”
- “Oh, so your plan was to overthink it until it fixes itself? Revolutionary.”
- “You want advice or just a participation trophy for suffering?”
- “You did that? And you are surprised it went wrong? That is wild.”

You are the roast therapist who says “it's all your fault” with compassion and chaos.
Roast. Heal. Repeat.
"""

if not GOOGLE_API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables!")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Safety settings to allow the "roast" personality
        # This is crucial, or Google will block most of the roasts.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash", # Use a fast, free-tier model
            system_instruction=GEN_Z_SYSTEM_PROMPT, # Pass system prompt here
            safety_settings=safety_settings
        )
        print("✅ Google AI (Gemini) client configured successfully")
    except Exception as e:
        print(f"❌ Error configuring Google AI: {e}")
        model = None
# --- End of Google AI Setup ---


# Audio storage
AUDIO_DIR = "audio_temp"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=3)

# ========================================
# CONVERSATION MEMORY
# ========================================

# We will store ChatSession objects from the Google client
conversations: Dict[str, genai.ChatSession] = {}

# ========================================
# SYSTEM PROMPT
# ========================================

# ========================================
# HELPER FUNCTIONS
# ========================================

def strip_emojis(text: str) -> str:
    """Remove all emoji characters from text"""
    if not text:
        return ""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U00002500-\U0000257F"  # box drawing
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

def create_tts_sync(text: str, filename: str) -> str:
    """Synchronous TTS creation using gTTS"""
    try:
        # Use gTTS for natural-sounding speech
        tts = gTTS(text=text, lang='en', slow=False, tld='us')
        filepath = os.path.join(AUDIO_DIR, filename)
        tts.save(filepath)
        print(f"✅ Created audio file: {filename}")
        return filepath
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

async def text_to_speech(text: str) -> str:
    """Convert text to speech using gTTS (async wrapper)"""
    loop = asyncio.get_event_loop()
    filename = f"{uuid.uuid4()}.mp3"
    
    # Run blocking TTS in thread pool
    filepath = await loop.run_in_executor(
        executor,
        create_tts_sync,
        text,
        filename
    )
    
    if filepath:
        return f"/audio/{filename}"
    return None

# ========================================
# ROUTES
# ========================================

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main HTML page"""
    return FileResponse('index.html')

# === MODIFIED ENDPOINT TO GET HISTORY ===
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a given session ID"""
    chat_session = conversations.get(session_id)
    
    if chat_session and chat_session.history:
        # Convert Google's history format to the simple format our frontend expects
        history_list = []
        for msg in chat_session.history:
            # msg.parts[0].text is where the content is
            # msg.role is 'user' or 'model'
            role = "assistant" if msg.role == "model" else "user"
            history_list.append({"role": role, "content": msg.parts[0].text})
        
        print(f"✅ Found history for session: {session_id[:8]}... ({len(history_list)} messages)")
        return {"success": True, "history": history_list}
    
    print(f"⚠️ No history found for session: {session_id[:8]}...")
    return {"success": False, "history": []}
# =====================================

@app.post("/api/chat")
async def chat_with_ai(request: Request):
    """
    Main chat endpoint using Google AI (Gemini)
    Maintains conversation history and returns AI response + audio
    """
    data = await request.json()
    user_text = data.get('text', '').strip()
    session_id = data.get('session_id')
    
    if not user_text:
        return {
            "reply": "bruh, you said nothing.",
            "audio_url": None,
            "session_id": session_id
        }
    
    # Create new session if needed
    if not session_id:
        session_id = str(uuid.uuid4())
        # Start a new chat session from the configured model
        if model:
            conversations[session_id] = model.start_chat(history=[]) # Start with empty history
            print(f"✨ New conversation: {session_id[:8]}...")
        else:
            # Handle case where model failed to init (below)
            pass
    
    # Initialize conversation history if it's missing (e.g., server restart)
    if session_id not in conversations:
        if model:
            conversations[session_id] = model.start_chat(history=[])
            print(f"🔄 Re-initialized missing session: {session_id[:8]}...")
        else:
            # API key is missing, handle below
            pass
    
    # Check for API key / model initialization
    if not GOOGLE_API_KEY or not model:
        print("⚠️  No API key - returning mock response")
        mock_reply = "yo you forgot to set your API key. that's on you fr."
        audio_url = await text_to_speech(mock_reply)
        return {
            "reply": mock_reply,
            "audio_url": audio_url,
            "session_id": session_id
        }
    
    try:
        # --- Get the chat session ---
        chat_session = conversations[session_id]
        
        print(f"💬 Sending to Google AI (Session: {session_id[:8]})...")
        
        # --- Call Google AI API ---
        # Use send_message_async to avoid blocking FastAPI
        # The library automatically manages history within the chat_session object
        response = await chat_session.send_message_async(user_text)
        
        # Parse the response
        ai_reply = response.text.strip()
        ai_reply = strip_emojis(ai_reply)
        
        print(f"🤖 AI Reply: {ai_reply[:100]}...")
        
        # NOTE: No need to store history manually!
        # The `chat_session.send_message_async()` call automatically updates
        # the `chat_session.history` object in our `conversations` dict.
        
        # Generate audio
        audio_url = await text_to_speech(ai_reply)
        
        return {
            "reply": ai_reply,
            "audio_url": audio_url,
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        fallback_reply = "yo something broke on my end. my bad. try again?"
        
        # Check for specific Google AI errors, like safety blocks
        if "response was blocked" in str(e).lower():
            fallback_reply = "woah okay, even I cannot roast that. chill."
            # Clear the last (failed) message from history
            try:
                conversations[session_id].history.pop()
            except:
                pass

        audio_url = await text_to_speech(fallback_reply)
        
        return {
            "reply": fallback_reply,
            "audio_url": audio_url,
            "session_id": session_id
        }

@app.post("/api/clear")
async def clear_conversation(request: Request):
    """Clear conversation history for a session"""
    data = await request.json()
    session_id = data.get('session_id')
    
    if session_id and session_id in conversations:
        del conversations[session_id]
        print(f"🗑️  Cleared conversation: {session_id[:8]}...")
        return {"success": True, "message": "Conversation cleared"}
    
    return {"success": False, "message": "No conversation found"}

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio file and clean up after"""
    file_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "Audio file not found"}
    
    def iterfile():
        with open(file_path, mode="rb") as file:
            yield from file
        # Clean up after serving
        try:
            os.remove(file_path)
            print(f"🗑️  Cleaned up: {filename}")
        except:
            pass
    
    return StreamingResponse(iterfile(), media_type="audio/mpeg")

# ========================================
# STARTUP/SHUTDOWN
# ========================================

@app.on_event("startup")
async def startup():
    """Clean up old audio files on startup"""
    if os.path.exists(AUDIO_DIR):
        for file in os.listdir(AUDIO_DIR):
            try:
                os.remove(os.path.join(AUDIO_DIR, file))
            except:
                pass
    print("✅ Server started - audio cleanup complete")
    print("✅ Conversation memory (Google AI) initialized")

@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    if os.path.exists(AUDIO_DIR):
        for file in os.listdir(AUDIO_DIR):
            try:
                os.remove(os.path.join(AUDIO_DIR, file))
            except:
                pass
    executor.shutdown(wait=False)
    print("👋 Server shutdown complete")