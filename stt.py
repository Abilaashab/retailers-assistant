import streamlit as st
from streamlit_mic_recorder import mic_recorder
import requests
import io
from pydub import AudioSegment
import os

from pydub import AudioSegment
from pydub.utils import which

# Manually set the ffmpeg and ffprobe path
AudioSegment.converter = which("C:\Users\Administrator\Downloads\ffmpeg-7.1.1\\bin\\ffmpeg.exe")   # Adjust if needed
AudioSegment.ffprobe = which("C:\\ffmpeg\\bin\\ffprobe.exe")   # Add this too!

# Load your API key from environment variable or .env
SARVAM_AI_API = os.getenv("SARVAM_AI_API")

# Sarvam API config
API_URL = "https://api.sarvam.ai/speech-to-text-translate"
HEADERS = {
    "api-subscription-key": SARVAM_AI_API
}
MODEL = "saaras:v2"

# Language code mapping
LANGUAGE_CODE_MAP = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "English": "en"
}

st.title("🎙️ Indic Speech Translator using Sarvam AI")

# Language selection
selected_lang = st.selectbox("Select language spoken in audio:", list(LANGUAGE_CODE_MAP.keys()))
lang_code = LANGUAGE_CODE_MAP[selected_lang]

# Record audio
audio_bytes = mic_recorder(
    start_prompt="Click to record",
    stop_prompt="Stop recording",
    key="recorder",
    use_container_width=True
)


if audio_bytes:
    st.success("Audio recorded successfully!")

    # ✅ Extract raw bytes
    raw_audio = audio_bytes["bytes"]

    with st.expander("🔊 Play Audio"):
        st.audio(raw_audio, format="audio/wav")

    if st.button("🎯 Translate to English"):
        with st.spinner("Translating using Saaras API..."):

            # Convert to AudioSegment and export to WAV in memory
            audio = AudioSegment.from_file(io.BytesIO(raw_audio), format="wav")
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)

            # Prepare payload
            data = {
                "model": MODEL,
                "with_diarization": False,
                "language": lang_code
            }
            files = {
                'file': ('audio.wav', buffer, 'audio/wav')
            }

            # Make API request
            response = requests.post(API_URL, headers=HEADERS, files=files, data=data)

            if response.status_code in [200, 201]:
                res_json = response.json()
                st.subheader("📜 Translated Transcript:")
                st.markdown(res_json.get("transcript", "*No transcript returned*"))
                st.caption(f"Detected language: `{res_json.get('language_code', 'N/A')}`")
            else:
                st.error(f"API call failed: {response.status_code}")
                st.code(response.text)


else:
    st.info("Please record audio to begin.")
