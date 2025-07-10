import streamlit as st
import vosk
import json
import threading
import time
import sounddevice as sd
import queue

# Path to your vosk model
VOSK_MODEL_PATH = "vosk-model-small-hi-0.22"

# Initialize Vosk model
model = vosk.Model(VOSK_MODEL_PATH)
samplerate = 16000
chunk_size = 4000  # 0.25 sec
rec = vosk.KaldiRecognizer(model, samplerate)

# Global flags
keep_listening = False
final_text = ""

# Audio queue
q = queue.Queue()

# Audio callback function for sounddevice
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Audio thread function
def listen_audio():
    global keep_listening, final_text
    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, dtype='int16'):
        while keep_listening:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    final_text += result["text"] + " "
            else:
                _ = json.loads(rec.PartialResult())

# Set page config
st.set_page_config(page_title="🎤 Speech-En-IND", layout="centered")

# Custom Tailwind-like styling (via embedded HTML)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body {
        font-family: 'Inter', sans-serif;
        background: #0f172a;
        color: #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(to right, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(to right, #7c3aed, #a855f7);
    }
    .mic-glow {
        animation: pulse 1.2s infinite;
        color: #ef4444;
        font-size: 1.2rem;
    }
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .box {
        background: #1e293b;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #334155;
        min-height: 250px;
        color: #fff;
        margin-bottom: 40px;    box-shadow: 0px 4px 16px 10px #a5a5a5;
    }
    .btn-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .btn-row button {
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .start-btn {
        background: linear-gradient(to right, #10b981, #22d3ee);
        color: white;
    }
    .stop-btn {
        background: linear-gradient(to right, #ef4444, #f87171);
        color: white;
    }
    .clear-btn {
        background: linear-gradient(to right, #6b7280, #9ca3af);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# UI Heading
st.markdown("<h1 style='text-align: center; color: #a5b4fc;'>🎤 Live Speech Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Using Vosk + Streamlit | Hindi (India)</p>", unsafe_allow_html=True)
st.markdown("---")

# Session state
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Placeholders
output_placeholder = st.empty()
mic_animation_placeholder = st.empty()

# Custom HTML Buttons Row
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Listening", key="start", disabled=st.session_state.is_listening):
        st.session_state.is_listening = True
        keep_listening = True
        final_text = ""
        st.session_state.transcript = ""
        threading.Thread(target=listen_audio).start()
        st.success("Started listening... Speak into your mic.")

with col2:
    if st.button("⏹️ Stop Listening", key="stop", disabled=not st.session_state.is_listening):
        keep_listening = False
        st.session_state.is_listening = False
        st.info("Stopped listening.")

with col3:
    if st.button("🗑️ Clear Transcript", key="clear"):
        st.session_state.transcript = ""
        final_text = ""
        st.info("Transcript cleared.")

# Mic animation + transcript display
if st.session_state.is_listening:
    blink = True
    while keep_listening:
        mic_status = "🎙️ <span class='mic-glow'>🔴 Listening...</span>" if blink else "🎙️ Listening..."
        mic_animation_placeholder.markdown(f"<h3 style='text-align:center;'>{mic_status}</h3>", unsafe_allow_html=True)
        blink = not blink
        st.session_state.transcript = final_text
        output_placeholder.markdown(f"<div class='box'><strong>Recognized Speech:</strong><br>{st.session_state.transcript}</div>", unsafe_allow_html=True)
        time.sleep(0.8)
else:
    mic_animation_placeholder.empty()
    output_placeholder.markdown(f"<div class='box'><strong>Recognized Speech:</strong><br>{st.session_state.transcript}</div>", unsafe_allow_html=True)
