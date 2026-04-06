# Remote Sensing Image Captioning (RSIC)

Comparative study of three deep learning architectures for automatically generating captions from satellite and aerial imagery.

## Architectures

| Model | Encoder | Decoder |
|---|---|---|
| ViT-BiLSTM | ViT-B/16 (TF-Hub, frozen) | Bidirectional LSTM (2 layers) |
| ViT-GPT2 | ViT-B/16 (HuggingFace, frozen) | GPT-2 (fine-tuned) |
| RemoteCLIP-GPT2 | RemoteCLIP ViT-B/32 (frozen) | GPT-2 (fine-tuned) |

## Results

Evaluated on the [Kaggle Satellite Image Caption Generation](https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation) dataset (test set, 1,093 images).

| Metric | ViT-BiLSTM | ViT-GPT2 | RemoteCLIP-GPT2 |
|---|---|---|---|
| BLEU-1 | 55.62 | 55.65 | 57.02 |
| BLEU-2 | 38.71 | 38.67 | 40.37 |
| BLEU-3 | 28.47 | 28.67 | 30.18 |
| BLEU-4 | 21.40 | 22.14 | 22.88 |
| METEOR | 23.84 | 25.46 | 26.11 |
| ROUGE-L | 41.36 | 40.90 | 42.41 |
| CIDEr | 36.03 | 37.53 | 44.90 |

## Setup

```bash
pip install -r requirements.txt
```

## Inference

```bash
python inference.py --image "path/to/image.jpg"
```

Add `--no-display` to print captions only without the visualization window:
```bash
python inference.py --image "path/to/image.jpg" --no-display
```

## Tech Stack

- **TensorFlow 2.15** — ViT-BiLSTM training & inference
- **PyTorch** — ViT-GPT2 & RemoteCLIP-GPT2 training & inference
- **HuggingFace Transformers** — GPT-2, ViT
- **OpenCLIP** — RemoteCLIP
- **TF-Hub** — ViT-B/16 feature extractor
