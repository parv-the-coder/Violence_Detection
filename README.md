# Violence Detection in Surveillance Video

A violence detection system for CCTV footage. A frozen **DINOv2 ViT-B/14**
encoder turns each frame into an embedding, and a small **temporal Transformer**
reads a 32 frame window of those embeddings and scores it for violence.

Around the model sits everything needed to actually run it: a Streamlit
dashboard, attention heatmaps that show where the model is looking, PDF incident
reports, CSV timelines, and email alerts.

Trained and evaluated on **UCF-Crime**.

## Results

| Metric | Value |
| --- | --- |
| Test ROC AUC | **0.9436** |
| Test accuracy | 86.72% |
| Test loss | 0.3087 |
| Train accuracy | 96.53% |
| Train loss | 0.1272 |
| Best epoch | 13 of 22 |

The checkpoint is chosen by test ROC AUC, not accuracy, because the two classes
are not balanced.

Confusion matrix over 4,557 test windows:

```text
                 Predicted Normal   Predicted Violence
Actual Normal          2086                 338
Actual Violence         267                1866
```

Please read these numbers as segment classification, not precise event
timing. The labels are per video, not per frame, so a quiet moment inside a
violent clip is still labelled violent. The limitations section of
[`report.md`](report.md) explains this and the other caveats in full.

## How it works

1. **Spatial encoder.** Meta's `dinov2_vitb14` (86M parameters, frozen) maps
   each frame to a 768 dimensional CLS embedding.
2. **Temporal classifier.** A 2 layer pre-norm Transformer encoder with 16
   attention heads, a learnable CLS token, and both learnable and sinusoidal
   position encodings. It reads 32 consecutive embeddings and outputs a violence
   score. About 4.7M trainable parameters.

Only the temporal head is trained. The backbone runs once and its output is
cached, so a full training run fits comfortably on a single GPU.

## Project layout

```
config.py                     Paths, hyperparameters, and env driven settings
models/
  spatial_extractor.py        Frozen DINOv2 backbone wrapper
  temporal_extractor.py       Temporal Transformer classifier
utils/preprocessing/
  conv_video_to_img.py        Videos to frames (blur filter, pHash dedup, sampling)
  conv_img_to_npy.py          Frames to .npy chunks, split by video
  forward_pass_extr.py        Batched DINOv2 feature extraction
train/train.py                Training loop, early stopping, checkpointing
test/test.py                  Full pipeline evaluation with latency measurement
modules/
  surveillance_analysis.py    Shared analysis engine (files and live sources)
  gradcam.py                  DINOv2 attention rollout heatmaps
  timeline.py                 Event merging and CSV export
  report_generator.py         PDF incident reports
  email_service.py            SMTP alert and report delivery
  alerts.py                   Snapshot capture and alert payloads
dashboard/streamlit_app.py    Streamlit monitoring UI
inference.py                  CLI entry point for a single video
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put a trained checkpoint at `model/best_model.pt`, or point
`SURVEILLANCE_MODEL_PATH` at one. Checkpoints are gitignored because of their
size, so you need this before running any inference.

For email alerts, copy the template and fill it in:

```bash
cp .env.example .env
```

Gmail needs an **App Password**, not your normal account password. `.env` is
gitignored, so never commit real credentials.

## Training from scratch

Download UCF-Crime into `Data/`, keeping the folder structure it ships with,
then run the four steps in order:

```bash
python utils/preprocessing/conv_video_to_img.py   # videos to clean frames
python utils/preprocessing/conv_img_to_npy.py     # frames to .npy chunks
python utils/preprocessing/forward_pass_extr.py   # frames to DINOv2 features
python train/train.py                             # train the temporal head
```

The train and test split happens at the **video** level, so no two frames from
the same clip can land on opposite sides of the split. Sliding windows also
never cross from one video into the next.

The best checkpoint goes to `train_output/models/best_model.pt`, picked on test
ROC AUC with early stopping after 5 epochs without improvement.

## Running inference

One video, annotated output:

```bash
python inference.py Data/my_test_video.mp4 --output annotated.mp4
```

Dashboard, for an uploaded video, a webcam, or an RTSP or IP stream:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard gives you a detection threshold slider, attention heatmaps, a
confidence over time chart, an event timeline, incident snapshots, and CSV and
PDF downloads.

## Outputs

Every run writes into `outputs/`:

| Path | Contents |
| --- | --- |
| `outputs/snapshots/` | Middle frame of each detected window, as JPEG |
| `outputs/heatmaps/` | Raw and overlaid attention heatmaps |
| `outputs/reports/` | Generated PDF incident reports |
| `outputs/timeline.csv` | Merged event timeline |

## Dataset

[UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/), which has 13 anomaly
categories plus normal footage. All 13 are currently treated as one positive
class, including a few that are not violent. See the limitations section of
`report.md`.
