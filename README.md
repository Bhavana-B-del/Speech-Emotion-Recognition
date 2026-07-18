---
base_model: facebook/wav2vec2-base
library_name: peft
tags:
- lora
- transformers
- speech-emotion-recognition
- audio-classification
---

# SER LoRA Adapter (wav2vec2-base, multilingual)

LoRA adapter fine-tuned on top of `facebook/wav2vec2-base` for 6-class speech emotion recognition (angry, disgust, fear, happy, neutral, sad) across English, Chinese-accented English, Persian, Italian, and Urdu speech, with zero-shot evaluation on German.

See the [main repository README](../README.md) for full training details, dataset sources, and evaluation results.

## Quick facts

- **Base model:** facebook/wav2vec2-base
- **Method:** LoRA (r=16, alpha=32, dropout=0.1) applied to `q_proj`, `k_proj`, `v_proj`, `out_proj` on the top 12 of 24 encoder layers; bottom 12 layers frozen
- **Task type:** Sequence classification (6 emotion classes)
- **Multilingual test accuracy:** 64.9% (macro F1: 0.619)
- **Zero-shot German (EMODB) accuracy:** 65.8%

## Usage

```python
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from peft import PeftModel
import json

with open("ser_final_model/ser_config.json") as f:
    cfg = json.load(f)

feature_extractor = AutoFeatureExtractor.from_pretrained("ser_final_model")
base = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(cfg["label2id"]),
    label2id=cfg["label2id"],
    id2label={int(k): v for k, v in cfg["id2label"].items()},
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(base, "ser_final_model")
model.eval()
```

### Framework versions
- PEFT 0.18.1
