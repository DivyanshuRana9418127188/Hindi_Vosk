import streamlit as st
import streamlit.components.v1 as components
import os
import json
import tempfile
import wave
import numpy as np
from datetime import datetime
import time
# --- VOSK OFFLINE RECOGNITION SETUP ---
# This app can optionally use the Vosk library for offline speech recognition.
# We check for its availability and warn the user if it's not installed.

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


class LiveVoskRecognizer:
    def __init__(self, model):
        self.model = model
        self.rec = vosk.KaldiRecognizer(model, 16000)
        self.rec.SetWords(True)
        self.transcript = ""

    def process(self, audio_bytes):
        if self.rec.AcceptWaveform(audio_bytes):
            result = json.loads(self.rec.Result())
            self.transcript += result.get("text", "") + " "
        else:
            partial = json.loads(self.rec.PartialResult())
        return self.transcript

class VoskSpeechRecognizer:
    """
    Handles offline speech recognition by processing audio files using a Vosk model.
    """
    def __init__(self, model_path: str):
        """Initializes the recognizer with the path to the Vosk model."""
        if not VOSK_AVAILABLE:
            raise ImportError("Vosk library not found. Please install it using 'pip install vosk'")
        
        self.model_path = model_path
        self.model = None
        self.sample_rate = 16000 # Vosk models generally expect 16kHz audio

    def load_model(self) -> bool:
        """
        Loads the Vosk language model from the specified path.
        Returns True if successful, False otherwise.
        """
        try:
            if not os.path.exists(self.model_path):
                st.error(f"Vosk model path does not exist: {self.model_path}")
                return False
            self.model = vosk.Model(self.model_path)
            return True
        except Exception as e:
            st.error(f"Error loading Vosk model: {e}")
            return False

    def transcribe_audio_file(self, audio_file) -> str:
        """
        Transcribes an uploaded audio file. It handles various audio formats
        by converting them to the required WAV format using librosa if available.
        """
        if not self.model:
            return "Error: Vosk model is not loaded."

        tmp_file_path = None
        try:
            # Determine file extension to ensure NamedTemporaryFile uses correct suffix
            file_extension = os.path.splitext(audio_file.name)[1] if audio_file.name else ".tmp"
            
            # Create a temporary file to store the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name

            # Use librosa to load and resample the audio to 16kHz mono, which is what Vosk expects.
            # This provides robust support for various audio formats (MP3, FLAC, etc.).
            if LIBROSA_AVAILABLE:
                audio_data, _ = librosa.load(tmp_file_path, sr=self.sample_rate, mono=True)
                # Convert audio data to 16-bit PCM format
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                # Fallback to standard 'wave' module if librosa is not installed (supports only WAV)
                with wave.open(tmp_file_path, 'rb') as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                        return "Error: Without 'librosa', only 16-bit mono PCM WAV files are supported."
                    if wf.getframerate() != self.sample_rate:
                        return f"Error: Audio must be {self.sample_rate}Hz. This file is {wf.getframerate()}Hz."
                    audio_int16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

            # Initialize the recognizer and process the audio data in chunks
            rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
            rec.SetWords(True)
            
            transcript = ""
            chunk_size = 4000
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                if rec.AcceptWaveform(chunk.tobytes()):
                    result = json.loads(rec.Result())
                    transcript += result.get("text", "") + " "
            
            # Get the final part of the transcript
            final_result = json.loads(rec.FinalResult())
            transcript += final_result.get("text", "")
            
            return transcript.strip()

        except Exception as e:
            st.error(f"An error occurred during audio processing: {e}")
            return "Error during transcription."
        
        finally:
            # Clean up the temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="🎤 Hybrid Speech-to-Text",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- CUSTOM CSS STYLING ---
    # Load custom styles from an external file for better organization.
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    # This ensures that variables persist across user interactions.
    if "vosk_recognizer" not in st.session_state:
        st.session_state.vosk_recognizer = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "vosk_transcript" not in st.session_state:
        st.session_state.vosk_transcript = ""
    if "web_transcript" not in st.session_state:
        st.session_state.web_transcript = ""
    if "web_is_listening" not in st.session_state:
        st.session_state.web_is_listening = False
    if "web_error" not in st.session_state:
        st.session_state.web_error = ""


    # --- HEADER SECTION ---
    st.markdown("""
    <div class="title-section">
        <h1 class="title">🎤 Hybrid Speech-to-Text</h1>
        <p class="subtitle">Real-time Browser API + Offline Vosk Model</p>
    </div>
    """, unsafe_allow_html=True)


    # --- SIDEBAR FOR VOSK CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Offline Model Settings")
        
        if VOSK_AVAILABLE:
            model_path = st.text_input(
                "Vosk Model Path",
                value="vosk-model-small-en-us-0.15", # Default English model
                help="Path to your downloaded Vosk model folder."
            )
            
            if st.button("🚀 Load Vosk Model"):
                if os.path.exists(model_path):
                    with st.spinner("Loading model into memory..."):
                        recognizer = VoskSpeechRecognizer(model_path)
                        if recognizer.load_model():
                            st.session_state.vosk_recognizer = recognizer
                            st.session_state.model_loaded = True
                            st.success("✅ Model loaded successfully!")
                            # Important: Rerun to update the main app UI with model_loaded state
                            st.rerun() 
                        else:
                            st.session_state.model_loaded = False
                            st.error("❌ Failed to load model.")
                else:
                    st.error("❌ Path not found. Please check the model path.")

            if st.session_state.model_loaded:
                st.success("🟢 Vosk model is ready.")
            else:
                st.info("⚪ Vosk model not loaded.")
        else:
            st.error("❌ Vosk not installed.")
            st.code("pip install vosk", language="bash")

        st.markdown("---")
        st.subheader("📥 Download Models")
        st.markdown("""
        1. Visit [Vosk Models Page](https://alphacephei.com/vosk/models)
        2. Download a model (e.g., small English or Hindi model).
        3. Extract the ZIP file.
        4. Copy the folder path into the input box above.
        """)

    # --- MAIN CONTENT TABS ---
    tab_web, tab_vosk = st.tabs(["🌐 Real-time (Browser)", "📁 Offline (File Upload)"])

    # --- TAB 1: WEB SPEECH API (REAL-TIME) ---
    with tab_web:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-title">🌐 Real-time Recognition</div>
            <div class="mode-description">
                Uses your browser's built-in Speech Recognition API for instant transcription. 
                Requires an active internet connection and works best in Chrome or Edge.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            web_language_options = [
                ("en-US", "English (US)"), ("en-GB", "English (UK)"), ("en-IN", "English (India)"),
                ("hi-IN", "हिंदी (Hindi)"), ("es-ES", "Español (España)"),
                ("fr-FR", "Français (France)"), ("de-DE", "Deutsch (Deutschland)"),
            ]
            web_language = st.selectbox(
                "Select Language",
                options=web_language_options,
                format_func=lambda x: x[1],
                key="web_language_select" # Added a key for selectbox consistency
            )
        with col2:
            continuous_mode = st.checkbox("Continuous Recognition", value=True, help="Keep listening until stopped.", key="continuous_mode_checkbox")

        # This HTML component contains the JavaScript for handling the Web Speech API.
        # It communicates with Streamlit by passing its state as a JSON object.
        web_speech_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .speech-controls {{ display: flex; justify-content: center; gap: 1rem; margin: 1.5rem 0; }}
                .speech-btn {{ 
                    padding: 10px 20px; border-radius: 50px; border: none; font-weight: 600;
                    font-size: 1rem; cursor: pointer; transition: all 0.2s ease;
                    display: flex; align-items: center; gap: 8px;
                }}
                #startBtn {{ background: #22c55e; color: white; }}
                #startBtn:hover:not(:disabled) {{ background: #16a34a; }}
                #stopBtn {{ background: #ef4444; color: white; }}
                #stopBtn:hover:not(:disabled) {{ background: #dc2626; }}
                #copyBtn, #clearBtn {{ background: #6b7280; color: white; }}
                #copyBtn:hover, #clearBtn:hover {{ background: #4b5563; }}
                .speech-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
                #status-toast {{ 
                    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                    background-color: #333; color: white; padding: 10px 20px; border-radius: 8px;
                    z-index: 1000; font-size: 0.9rem; opacity: 0; transition: opacity 0.3s;
                }}
            </style>
        </head>
        <body>
            <div class="speech-controls">
                <button id="startBtn" class="speech-btn">🎙️ Start</button>
                <button id="stopBtn" class="speech-btn" disabled>⏹️ Stop</button>
                <button id="clearBtn" class="speech-btn">🗑️ Clear</button>
                <button id="copyBtn" class="speech-btn">📋 Copy</button>
            </div>

            <div id="statusIndicator" class="listening-indicator" style="display: none;">
                <span class="pulse">🔴 Listening... Please speak.</span>
            </div>
            
            <div class="transcript-container" id="transcript-box">
                <span id="final-transcript">{st.session_state.web_transcript or "Your speech will appear here..."}</span>
                <span id="interim-transcript" style="color: #9ca3af;"></span>
            </div>

            <div id="status-toast"></div>

            <script>
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const clearBtn = document.getElementById('clearBtn');
                const copyBtn = document.getElementById('copyBtn');
                const finalSpan = document.getElementById('final-transcript');
                const interimSpan = document.getElementById('interim-transcript');
                const statusIndicator = document.getElementById('statusIndicator');
                const toast = document.getElementById('status-toast');

                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                let recognition;
                let finalTranscript = finalSpan.innerText === "Your speech will appear here..." ? "" : finalSpan.innerText;

                // Function to send component state back to Streamlit
                function sendStateToStreamlit() {{
                    const state = {{
                        transcript: finalTranscript,
                        isListening: !stopBtn.disabled, // Correctly reflect listening state
                    }};
                    window.parent.postMessage({{ type: 'streamlit:setComponentValue', value: state }}, '*');
                }}

                if (SpeechRecognition) {{
                    recognition = new SpeechRecognition();
                    recognition.lang = '{web_language[0]}';
                    recognition.continuous = {str(continuous_mode).lower()};
                    recognition.interimResults = true;

                    recognition.onstart = () => {{
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        statusIndicator.style.display = 'block';
                        sendStateToStreamlit();
                    }};

                    recognition.onend = () => {{
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        statusIndicator.style.display = 'none';
                        sendStateToStreamlit();
                    }};

                    recognition.onresult = (event) => {{
                        let interim = '';
                        for (let i = event.resultIndex; i < event.results.length; i++) {{
                            const transcriptPart = event.results[i][0].transcript;
                            if (event.results[i].isFinal) {{
                                finalTranscript += transcriptPart + ' ';
                            }} else {{
                                interim += transcriptPart;
                            }}
                        }}
                        finalSpan.innerText = finalTranscript;
                        interimSpan.innerText = interim;
                        // Only send state to Streamlit when a final result is available or on stop
                        // For interim results, continuously sending can cause too many reruns.
                        // However, since Streamlit only reads the component's value on a full rerun,
                        // this approach works fine for updating the `st.session_state.web_transcript`
                        // to reflect the *final* transcript when recognition ends or if the user stops it.
                        // For real-time updates of interim results *within* Streamlit, a different
                        // component strategy (like an input field updated by JS) would be needed.
                        // For now, this is acceptable as the visible HTML updates real-time.
                    }};

                    recognition.onerror = (event) => {{
                        showToast(`Error: ${{event.error}}`);
                        // Send state to Streamlit on error to update listening status
                        sendStateToStreamlit(); 
                        stopRecognition();
                    }};
                }} else {{
                    finalSpan.innerText = "❌ Speech recognition not supported in this browser. Please use Chrome or Edge.";
                    startBtn.disabled = true;
                    // Inform Streamlit about the error state
                    window.parent.postMessage({{ type: 'streamlit:setComponentValue', value: {{ transcript: finalSpan.innerText, isListening: false, error: finalSpan.innerText }} }}, '*');
                }}

                function startRecognition() {{
                    if (recognition) {{
                        if (finalSpan.innerText === "Your speech will appear here...") {{
                            finalTranscript = "";
                            finalSpan.innerText = "";
                        }}
                        recognition.start();
                    }}
                }}
                
                function stopRecognition() {{
                    if (recognition) {{
                        recognition.stop();
                    }}
                    // Ensure the final transcript is sent to Streamlit when stopping
                    sendStateToStreamlit(); 
                }}

                function clearTranscript() {{
                    finalTranscript = "";
                    finalSpan.innerText = "Your speech will appear here...";
                    interimSpan.innerText = "";
                    sendStateToStreamlit();
                }}

                function copyTranscript() {{
                    const textToCopy = finalSpan.innerText;
                    if (!textToCopy || textToCopy === "Your speech will appear here...") {{
                        showToast("Nothing to copy.");
                        return;
                    }}
                    navigator.clipboard.writeText(textToCopy).then(() => {{
                        showToast("✅ Copied to clipboard!");
                    }}, (err) => {{
                        showToast("❌ Copy failed. Please try again.");
                    }});
                }}
                
                function showToast(message) {{
                    toast.innerText = message;
                    toast.style.opacity = 1;
                    setTimeout(() => {{ toast.style.opacity = 0; }}, 3000);
                }}

                startBtn.addEventListener('click', startRecognition);
                stopBtn.addEventListener('click', stopRecognition);
                clearBtn.addEventListener('click', clearTranscript);
                copyBtn.addEventListener('click', copyTranscript);

                // Initial state update when component loads (useful on app rerun)
                sendStateToStreamlit();

            </script>
        </body>
        </html>
        """
        
        # This component gets a key, so its state is stored in st.session_state
        component_value = components.html(web_speech_html, height=400)

        # Update session state based on the value returned from the JS component
        if isinstance(component_value, dict):
            # Use .get with a default to prevent KeyError if a key is missing
            st.session_state.web_transcript = component_value.get("transcript", st.session_state.web_transcript)
            st.session_state.web_is_listening = component_value.get("isListening", st.session_state.web_is_listening)
            st.session_state.web_error = component_value.get("error", "") # Capture JS errors if any
        
        # Display listening indicator outside the HTML component
        if st.session_state.web_is_listening:
            st.markdown('<div class="listening-indicator"><span class="pulse">🔴 Listening... Please speak.</span></div>', unsafe_allow_html=True)
        elif st.session_state.web_error:
            st.error(st.session_state.web_error)

        # Add a download button for the transcript
        if st.session_state.web_transcript and st.session_state.web_transcript != "Your speech will appear here...":
            st.download_button(
                "⬇️ Download Transcript",
                st.session_state.web_transcript,
                f"web_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # --- TAB 2: VOSK OFFLINE PROCESSING ---
    with tab_vosk:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-title">📁 Offline File Processing</div>
            <div class="mode-description">
                Upload an audio file (WAV, MP3, etc.) for transcription using the loaded Vosk model. 
                This runs entirely on the server, ensuring privacy and offline capability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.model_loaded:
            if not LIBROSA_AVAILABLE:
                st.markdown('<div class="warning-banner">⚠️ <b>Warning:</b> `librosa` is not installed. Only 16-bit, 16kHz mono WAV files are supported. Install with `pip install librosa` for broader format support.</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                st.audio(uploaded_file, format=uploaded_file.type)
                if st.button("🔄 Transcribe File"):
                    with st.spinner("Processing audio file... This may take a moment."):
                        transcript = st.session_state.vosk_recognizer.transcribe_audio_file(uploaded_file)
                        st.session_state.vosk_transcript = transcript

                if st.session_state.vosk_transcript:
                    st.subheader("📝 Transcription Result")
                    st.markdown(f'<div class="transcript-container">{st.session_state.vosk_transcript}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "⬇️ Download Transcript",
                        st.session_state.vosk_transcript,
                        f"vosk_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.markdown('<div class="warning-banner">⚠️ Please load a Vosk model from the sidebar to enable file transcription.</div>', unsafe_allow_html=True)
    


    # # --- TAB 3: LIVE MICROPHONE TRANSCRIPTION (VOSK) ---
    # with tab_live_vosk:
    #     st.markdown("""
    #     <div class="mode-card">
    #         <div class="mode-title">\U0001F3A4 Live Vosk Recognition</div>
    #         <div class="mode-description">
    #             Use your device's microphone for real-time transcription using the offline Vosk model. Works fully offline.
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    #     if not VOSK_AVAILABLE:
    #         st.error("❌ Vosk not installed. Please install it using `pip install vosk`.")
    #     elif "model_loaded" not in st.session_state or not st.session_state.model_loaded:
    #         st.warning("⚠️ Please load a Vosk model from the sidebar to start live transcription.")
    #     else:
    #         if "vosk_transcript" not in st.session_state:
    #             st.session_state.vosk_transcript = ""

    #         vosk_model = st.session_state.vosk_recognizer.model
    #         live_recognizer = LiveVoskRecognizer(vosk_model)

    #         # Store updated transcript
    #         placeholder = st.empty()

    #         def audio_frame_callback(frame: av.AudioFrame):
    #             pcm_data = frame.to_ndarray().flatten().astype(np.int16).tobytes()
    #             live_recognizer.process(pcm_data)
    #             return frame

    #         webrtc_ctx = webrtc_streamer(
    #             key="vosk_live_stream",
    #             mode=WebRtcMode.SENDONLY,
    #             audio_receiver_size=1024,
    #             audio_frame_callback=audio_frame_callback,
    #             media_stream_constraints={"audio": True, "video": False},
    #         )

    #         # Live polling loop
    #         if webrtc_ctx.state.playing:
    #             st.success("🔴 Listening... Speak into your mic.")
    #             while webrtc_ctx.state.playing:
    #                 st.session_state.vosk_transcript = live_recognizer.transcript
    #                 placeholder.markdown(f'<div class="transcript-container">{st.session_state.vosk_transcript}</div>', unsafe_allow_html=True)
    #                 time.sleep(1)





if __name__ == "__main__":
    main()