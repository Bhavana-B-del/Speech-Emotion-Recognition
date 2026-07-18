# VibeCheck.ai - Multilingual Speech Emotion Recognition

A speech emotion recognition system that classifies 6 emotions (angry, disgust, fear, happy, neutral, sad) from short voice clips, fine-tuned to generalize across multiple languages and deployed as an interactive Streamlit app.

🎤 Record or upload audio → get an emotion prediction with a full confidence breakdown, wrapped in a playful "VibeCheck" UI.

## Highlights

- **Multilingual by design**: trained across English, Chinese-accented English, Persian, Italian, and Urdu speech — not just a single-language model
- **Zero-shot generalization tested**: evaluated on German (EMODB), a language the model never saw during training, to measure true generalization rather than memorization
- **Parameter-efficient fine-tuning**: LoRA adapters on top of `facebook/wav2vec2-base` instead of full fine-tuning — ~4.7MB of trained weights instead of retraining the full ~95M parameter model
- **Deployed as a live demo app**: Streamlit interface supporting both live microphone recording and file upload

## Approach

### Base model & fine-tuning strategy

- **Base model:** `facebook/wav2vec2-base`
- **Method:** LoRA (r=16, alpha=32, dropout=0.1) applied to attention projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`)
- **Layer strategy:** LoRA adapters applied only to the **top 12 of 24 encoder layers**; the bottom 12 layers were kept fully frozen to preserve the base model's general phonetic representations while adapting higher-level layers to emotion-specific features
- **Learning rates:** dual LR setup — encoder layers at 5e-5, classification head at 2e-4 (10x higher), since earlier training runs with a single shared LR (or too large a gap) caused the head to destabilize
- **Training:** 15 epochs, effective batch size 32 (batch size 8 × gradient accumulation 4), cosine LR schedule with 1000-step warmup, early stopping on macro F1

### Datasets

Combined and balanced across 8 public emotion-speech datasets, with per-dataset sampling caps to prevent English from dominating the training distribution:

| Dataset | Language | Sampling |
|---|---|---|
| RAVDESS | English | capped 1500 |
| CREMA-D | English | capped 2000 |
| TESS | English | capped 1500 |
| SAVEE | English | full (480) |
| ESD | Chinese-accented English | capped 3500 (from ~35k) |
| ShEMO | Persian | full (~3000) |
| EMOVO | Italian | full (588) |
| URDU | Urdu | full (400) |

German (EMODB) was deliberately **excluded from training** and used only as a zero-shot evaluation set.

### Why LoRA over full fine-tuning or other approaches

Full fine-tuning of a wav2vec2-scale model was unnecessary for this task and riskier for a limited, imbalanced, multilingual dataset — LoRA kept the trainable parameter count small, made experimentation faster, and reduced overfitting risk on the smaller non-English subsets.

## Results

**Multilingual test set** (held-out mix of all trained languages):
- Accuracy: **64.9%**
- Macro F1: **0.619**

Per-class accuracy varied notably - `neutral` (0.83) and `angry` (0.78) were the strongest classes, while `happy` (0.28) was the weakest, frequently confused with `angry` and `fear` (see confusion matrix in `training.ipynb`).

**Zero-shot generalization - German (EMODB, unseen during training):**
- Accuracy: **65.8%**
- The model generalized surprisingly well to a language it never saw, particularly on `angry` (97.6% recall) and `sad` (79%), though `disgust` and `fear` recall dropped sharply - likely due to class imbalance in training data and cross-lingual differences in how these emotions are vocally expressed.

Full classification reports and confusion matrices are in `training.ipynb`.

## App

Two Streamlit UI variants were built on top of the same fine-tuned model:

- **`app.py`** (main) - "VibeCheck," a more playful, roast-styled interface. Records or uploads audio, predicts emotion, and returns a lighthearted "roast" alongside the full probability breakdown.
- **`formal_variant.py`** - an earlier, more clinical UI presentation of the same model, framed explicitly around the cross-lingual evaluation results.

Both support:
- Live microphone recording (via `streamlit-mic-recorder`)
- File upload (WAV/MP3/OGG/FLAC)
- Robust audio decoding across formats (direct `soundfile` read, falling back to `ffmpeg` piping, falling back to temp-file conversion)

## Repository Structure

```
speech-emotion-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                          # Main app (VibeCheck)
├── formal_variant.py               # Earlier formal UI variant
├── training.ipynb                  # Full training pipeline, data prep, evaluation
└── ser_final_model/                # Trained LoRA adapter
    ├── README.md
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── preprocessor_config.json
    └── ser_config.json
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the LoRA adapter from `./ser_final_model` at startup (cached after first load). Note: `ffmpeg` must be available on your system for audio format conversion - `imageio-ffmpeg` bundles a portable binary automatically.

## Tech Stack

- **Model:** `facebook/wav2vec2-base` + PEFT/LoRA
- **Training:** PyTorch, HuggingFace Transformers, PEFT, Datasets, Evaluate
- **App:** Streamlit, `streamlit-mic-recorder`
- **Audio processing:** `soundfile`, `librosa`, `pydub`, `ffmpeg`

## Limitations

- `happy` is the weakest-performing class across both the multilingual test set and zero-shot evaluation, likely due to class imbalance and high acoustic overlap with `angry`/`fear` in some datasets
- Optimized for short clips (2–6 seconds); longer inputs are truncated
- Zero-shot performance on languages/accents further from the training distribution (e.g. tonal languages, non-Indo-European languages) is untested

## License

MIT
