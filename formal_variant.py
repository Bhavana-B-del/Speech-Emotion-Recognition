import streamlit as st
import torch
import numpy as np
import soundfile as sf
import librosa
import json
import io
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from peft import PeftModel
from streamlit_mic_recorder import mic_recorder  # pip install streamlit-mic-recorder

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognizer",
    page_icon="🎙️",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────
MODEL_BASE  = "facebook/wav2vec2-base"
ADAPTER_DIR = "./ser_final_model"
MAX_LENGTH  = 16000 * 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_EMOJI = {
    "angry":   "😡",
    "disgust": "🤢",
    "fear":    "😨",
    "happy":   "😊",
    "neutral": "😐",
    "sad":     "😢",
}

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
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

# ── Inference ──────────────────────────────────────────────────
def predict(audio_array, sr):
    if sr != 16000:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
        )
    if audio_array.ndim == 2:
        audio_array = audio_array[:, 0]

    audio_array = audio_array.astype(np.float32)

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return probs

# ── Results display ────────────────────────────────────────────
def show_results(probs):
    top_idx   = int(np.argmax(probs))
    top_label = id2label[top_idx]
    top_prob  = probs[top_idx]
    top_emoji = EMOTION_EMOJI.get(top_label, "🎯")

    st.subheader("Result")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f"<h1 style='text-align:center;margin-top:10px'>{top_emoji}</h1>",
            unsafe_allow_html=True
        )
    with col2:
        st.metric("Detected Emotion", top_label.upper())
        st.metric("Confidence", f"{top_prob * 100:.1f}%")

    st.subheader("All Probabilities")
    for i in np.argsort(probs)[::-1]:
        label = id2label[i]
        prob  = probs[i]
        emoji = EMOTION_EMOJI.get(label, "")
        st.write(f"{emoji} **{label}**")
        st.progress(float(prob), text=f"{prob * 100:.1f}%")

    st.caption(
        "⚠️ Model accuracy: ~65% multilingual, ~66% zero-shot German. "
        "Works best on clean speech under 6 seconds."
    )

# ── App ────────────────────────────────────────────────────────
st.title("🎙️Cross-Lingual Speech Emotion Recognizer")
st.caption("Multilingual · English / Persian / Italian / Urdu / Chinese-accented English")

with st.spinner("Loading model... (first run downloads ~360MB base model)"):
    model, feature_extractor, id2label = load_model()
st.success("Model ready!", icon="✅")
st.divider()

# ── Input mode tabs ────────────────────────────────────────────
tab_upload, tab_record = st.tabs(["📂 Upload File", "🎤 Record Voice"])

# ── TAB 1: File upload ─────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a WAV, MP3, OGG or FLAC file (max 6s used)",
        type=["wav", "mp3", "ogg", "flac"]
    )

    if uploaded:
        st.audio(uploaded)
        with st.spinner("Analyzing emotion..."):
            audio_array, sr = sf.read(io.BytesIO(uploaded.read()))
            probs = predict(audio_array, sr)
        st.divider()
        show_results(probs)

# ── TAB 2: Live recording ──────────────────────────────────────
# ── TAB 2: Live recording ──────────────────────────────────────
with tab_record:
    st.write("Click **Start** to record, **Stop** when done (aim for 2–6 seconds).")

    audio_data = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="mic"
    )

    if audio_data and audio_data.get("bytes"):
        raw_bytes = audio_data["bytes"]
        st.audio(raw_bytes, format="audio/wav")

        with st.spinner("Analyzing emotion..."):
            try:
                # Browser gives WebM/Opus — convert to numpy via pydub
                from pydub import AudioSegment
                import tempfile, os

                # Write raw bytes to a temp file so pydub can detect format
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name

                audio_seg = AudioSegment.from_file(tmp_path)   # auto-detects format
                os.unlink(tmp_path)                             # cleanup

                # Convert to 16kHz mono numpy float32
                audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
                samples   = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
                samples  /= np.iinfo(audio_seg.array_type).max   # normalize to [-1, 1]

                probs = predict(samples, 16000)
                st.divider()
                show_results(probs)

            except Exception as e:
                st.error(f"Could not process recording: {e}")
                st.info("Make sure FFmpeg is installed and added to PATH.")