import streamlit as st
from streamlit_mic_recorder import mic_recorder
import torch
import numpy as np
import soundfile as sf
import imageio_ffmpeg
import subprocess
import tempfile
import os
import io
import json
import random
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from peft import PeftModel

# ── FFmpeg ─────────────────────────────────────────────────────
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="VibeCheck SER", page_icon="🎤", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #ff4b4b;
        background-color: transparent;
        color: white;
        width: 100%;
    }
    .stButton>button:hover { background-color: #ff4b4b; color: white; }
    h1 { font-family: 'Courier New', Courier, monospace; letter-spacing: -1px; }
    </style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
MODEL_BASE  = "facebook/wav2vec2-base"
ADAPTER_DIR = "./ser_final_model"
MAX_LENGTH  = 16000 * 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROASTS = {
    "angry":   [
        "Whoa, calm down Hulk. It's just a microphone, not your ex.",
        "Hydrate or diedrate, because that temper is crusty.",
        "Sir/Ma'am this is a Wendy's.",
    ],
    "sad": [
        "You sound like a damp paper towel in a thunderstorm.",
        "Is everything okay at home, or do you just enjoy being a main character?",
        "Crying is just your face leaking. You'll be fine.",
    ],
    "happy": [
        "The toxic positivity is radiant. Chill out.",
        "You sound suspiciously like you just found $20 in an old pair of jeans.",
        "Nobody is this happy. What are you hiding.",
    ],
    "neutral": [
        "Giving us absolutely nothing. Groundbreaking.",
        "A cardboard box has more personality than your current tone.",
        "Beige. You sound completely beige.",
    ],
    "fear": [
        "Scared of a microphone? Really?",
        "Whatever it is, it probably won't kill you. Probably.",
        "Your voice just did a little jumpscare. Respect.",
    ],
    "disgust": [
        "Same, honestly.",
        "You sound like you opened the fridge and regretted it immediately.",
        "The audacity of your face right now is palpable even through audio.",
    ],
}

EMOTION_EMOJI = {
    "angry": "😡", "sad": "😢", "happy": "😊",
    "neutral": "😐", "fear": "😨", "disgust": "🤢",
}

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_ser_model():
    with open(f"{ADAPTER_DIR}/ser_config.json") as f:
        cfg = json.load(f)
    label2id = cfg["label2id"]
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    feature_extractor = AutoFeatureExtractor.from_pretrained(ADAPTER_DIR)
    base = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval().to(DEVICE)
    return model, feature_extractor, id2label

# ── Audio decode ───────────────────────────────────────────────
def decode_audio(audio_bytes):
    # Try 1: soundfile directly (clean WAV)
    try:
        arr, sr = sf.read(io.BytesIO(audio_bytes))
        return arr, sr
    except Exception:
        pass

    # Try 2: ffmpeg pipe stdin → stdout
    try:
        result = subprocess.run(
            [FFMPEG, "-y", "-i", "pipe:0",
             "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1", "-vn"],
            input=audio_bytes,
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout:
            arr, sr = sf.read(io.BytesIO(result.stdout))
            return arr, sr
    except Exception:
        pass

    # Try 3: ffmpeg via temp files
    inp_path, out_path = None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as f:
            f.write(audio_bytes)
            inp_path = f.name
        out_path = inp_path + ".wav"
        subprocess.run(
            [FFMPEG, "-y", "-i", inp_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=30
        )
        if os.path.exists(out_path):
            arr, sr = sf.read(out_path)
            return arr, sr
    except Exception:
        pass
    finally:
        for p in [inp_path, out_path]:
            if p and os.path.exists(p):
                os.remove(p)

    return None, None

# ── Predict ────────────────────────────────────────────────────
def predict(audio_bytes):
    arr, sr = decode_audio(audio_bytes)
    if arr is None:
        return None

    if arr.ndim == 2:
        arr = arr[:, 0]
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)

    arr = arr.astype(np.float32)
    inputs = feature_extractor(
        arr, sampling_rate=16000,
        max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    return probs

# ── Results ────────────────────────────────────────────────────
def show_results(probs):
    top_idx   = int(np.argmax(probs))
    top_label = id2label[top_idx]
    top_prob  = probs[top_idx]
    emoji     = EMOTION_EMOJI.get(top_label, "🎯")
    roast     = random.choice(ROASTS.get(top_label, ["...interesting."]))

    st.markdown(f"## {emoji} **{top_label.upper()}** — {top_prob*100:.0f}% sure")
    st.info(f"💬 {roast}")
    st.write("")
    st.caption("Full breakdown:")
    for i in np.argsort(probs)[::-1]:
        label = id2label[i]
        st.write(f"{EMOTION_EMOJI.get(label,'')} **{label}**")
        st.progress(float(probs[i]), text=f"{probs[i]*100:.1f}%")

# ── App ────────────────────────────────────────────────────────
st.title("VibeCheck.ai 🎤")
st.caption("Don't speak if you can't handle the truth.")
st.write("---")

with st.spinner("Booting the judgment machine..."):
    model, feature_extractor, id2label = load_ser_model()

tab_record, tab_upload = st.tabs(["🎤 Record", "📂 Upload"])

# ── Record tab ─────────────────────────────────────────────────
with tab_record:
    st.subheader("Expose Your Feelings")
    st.caption("Aim for 2–6 seconds of speech.")

    audio_data = mic_recorder(
        start_prompt="⏺️ Start Ranting",
        stop_prompt="⏹️ Stop & Get Judged",
        just_once=False,
        use_container_width=True,
        key="recorder"
    )

    if audio_data and audio_data.get("bytes"):
        raw = audio_data["bytes"]
        st.audio(raw, format="audio/wav")
        st.caption(f"Format check — bytes: {len(raw)} | hex: {raw[:8].hex()}")

        with st.status("Judging your tone...", expanded=True) as status:
            st.write("Extracting the drama from your voice...")
            probs = predict(raw)
            status.update(label="Judgment complete.", state="complete", expanded=False)

        if probs is not None:
            st.write("---")
            show_results(probs)
        else:
            st.error("Could not decode audio. Paste the hex above in chat!")
    else:
        st.write("Click the button above. If you're scared, just say that.")

# ── Upload tab ─────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Your Baggage")
    uploaded = st.file_uploader(
        "Drop a WAV, MP3, OGG or FLAC file",
        type=["wav", "mp3", "ogg", "flac"]
    )

    if uploaded:
        st.audio(uploaded)
        with st.status("Judging your tone...", expanded=True) as status:
            st.write("Extracting the drama from your voice...")
            probs = predict(uploaded.read())
            status.update(label="Judgment complete.", state="complete", expanded=False)

        if probs is not None:
            st.write("---")
            show_results(probs)
        else:
            st.error("Could not decode audio. Try a WAV file.")

st.write("---")
st.caption("Minimalist. Savage. Slightly Unnecessary.")