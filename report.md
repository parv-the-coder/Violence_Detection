# Violence Detection in Videos using Vision Transformers

This report describes the whole project as it stands in the repository: how the
data is prepared, how features are extracted, how the model is built and
trained, how inference runs, and how the dashboard, reports, and email alerts
work.

The aim of the system is to spot violent activity in surveillance video,
produce annotated output, generate reports, and optionally send an email when
something is detected.

---

## 1. Project overview

The project is built on a two stage pipeline:

1. A frozen DINOv2 backbone turns individual frames into spatial embeddings.
2. A temporal Transformer reads short sequences of those embeddings and labels
   each window as normal or violent.

On top of that sits a Streamlit dashboard for uploading video or watching a live
camera, plus PDF reports, CSV timeline export, snapshot and heatmap saving, and
email notifications.

### What the system can do

- Train a violence classifier from pre-extracted visual features.
- Run inference on raw video files and write an annotated output video.
- Monitor a live webcam or an RTSP or IP camera.
- Save incident snapshots, heatmaps, and CSV timelines.
- Generate PDF incident reports.
- Send alert and summary emails.
- Show metrics and detection history in a dashboard.

---

## 2. Technology stack

- Python
- PyTorch for the model definition and inference
- torchvision for image preprocessing and normalisation
- OpenCV for video decoding, writing, snapshots, and overlays
- NumPy and Pandas for data handling
- scikit-learn for splitting and metrics
- scikit-image for the optional SSIM duplicate check
- ReportLab for PDF generation
- Streamlit for the dashboard
- Plotly and Matplotlib for charts
- SMTP (Gmail by default) for email delivery

The workspace keeps models, modules, preprocessing, training, inference,
dashboard, outputs, and tests in separate folders.

---

## 3. Architecture

Spatial and temporal modelling are kept separate, which is what makes the
training cost manageable.

```text
Raw video
  -> frame extraction and cleanup
  -> centre crop and resize
  -> DINOv2 spatial embeddings
  -> temporal windowing
  -> Temporal Transformer
  -> violence probability
```

### 3.1 Spatial feature extractor

The backbone is `dinov2_vitb14`, wrapped by `models/spatial_extractor.py`.

- The backbone is frozen, so features stay stable across runs.
- Each frame becomes a 768 dimensional CLS embedding.
- Input is 224x224 with ImageNet normalisation.
- The same wrapper is reused in preprocessing and at inference time.

Why this works:

- DINOv2 captures strong general visual semantics without any fine tuning.
- Freezing it makes training far cheaper.
- Caching the features means the backbone runs once, not once per epoch.

### 3.2 Temporal classifier

The classifier is a small Transformer encoder in
`models/temporal_extractor.py`.

- Sequence length: 32 frames
- Feature dimension: 768
- Attention heads: 16
- Encoder layers: 2
- Dropout: 0.3
- Output: 2 classes, violence and non violence

Details worth noting:

- A learnable CLS token is placed at the front of each sequence.
- Both learnable position embeddings and fixed sinusoidal encodings are added.
- Each block is pre-norm: LayerNorm, then attention or feed forward, then the
  residual add. Post-norm was tried first and the loss diverged to NaN within a
  few epochs at a learning rate of 1e-3.
- The final CLS output goes through a linear head and a sigmoid.

### 3.3 Result object

A run returns a structured result containing total incidents, highest and
average confidence, duration and processing speed, timeline events, the
annotated video path, snapshot paths, heatmap paths, and the PDF report path.

---

## 4. Data preparation

The preprocessing pipeline cleans up surveillance footage, builds training ready
arrays, and keeps the train and test sets properly separated.

### 4.1 Video to frames

`utils/preprocessing/conv_video_to_img.py` turns raw videos into frame folders.

- Videos are processed in parallel with `ProcessPoolExecutor`, one worker per
  core minus one. Processes rather than threads, because OpenCV decoding is not
  thread safe.
- Each video gets a frame budget based on its duration, from 14 frames for a
  clip under 10 seconds up to 70 for anything over a minute.
- Blurry frames are dropped using Laplacian variance.
- Near duplicate frames are removed with a pHash fingerprint by default. SSIM is
  available as a more accurate but roughly 10 times slower alternative.
- Frames are saved as JPEG at quality 95.

Two bugs were fixed in this stage during development. OpenCV decodes to BGR
while PIL writes RGB, so early extractions had their red and blue channels
swapped and had to be redone. Separately, a few clips report 0 fps in their
metadata, which made the sampling interval zero and crashed the run.

### 4.2 Frames to NumPy chunks

`utils/preprocessing/conv_img_to_npy.py` loads the frames and writes chunked
`.npy` files.

- Frames are centre cropped. An early version cropped from the top left, which
  cut away the part of the frame where the action usually is.
- Output is resized to 224x224 and scaled to float32 in the `[0, 1]` range.
- Images load on a thread pool, written back by index so frame order survives.
- The index where each video starts is recorded, so later stages can keep
  sequence windows inside a single video.
- The split is 80/20 at the video level.

The video level split matters. An earlier version split the flat list of frames,
which put frames from the same clip on both sides of the split. Since
consecutive frames are almost identical, the test score was largely measuring
memorisation.

### 4.3 DINOv2 feature extraction

`utils/preprocessing/forward_pass_extr.py` runs batched forward passes through
the backbone.

- Loads the normalised arrays from the previous stage.
- Runs 64 frames per batch. One frame at a time left the GPU mostly idle and
  turned a few minutes of work into hours.
- Applies ImageNet mean and standard deviation only. An earlier version divided
  by 255 a second time, on data that was already scaled, so everything reached
  the backbone at roughly 1/255 of its intended intensity.
- Saves train and test features into `output_features_dinov2/`.

### 4.4 Statistics helper

`check_stats.py` reports how many videos and frames exist per category, which is
a quick way to sanity check the extracted dataset and its class balance.

---

## 5. Training

Training lives in `train/train.py`.

### 5.1 Building the dataset

Training runs on the cached DINOv2 features, not on raw images.

- Sliding windows of 32 frames are cut from the feature sequence.
- No window is allowed to cross a video boundary. Before this was enforced,
  windows could mix frames from two unrelated clips and then take their label
  from whichever video the last frame belonged to.
- Each window is labelled by its last frame.

### 5.2 Optimisation

- Batch size: 32
- Optimiser: SGD
- Loss: binary cross entropy
- Initial learning rate: 0.001
- Learning rate decay: 0.9 per epoch
- Random seed: 42
- Early stopping: 5 epochs without an improvement in test ROC AUC

### 5.3 What training produces

- Per epoch metrics in `train_output/training_log.csv`
- The best checkpoint in `train_output/models/best_model.pt`
- A confusion matrix printed each epoch
- A loss curve in `train_output/loss_curve.png`

Checkpoints are selected on ROC AUC rather than accuracy, so the saved model is
the one that ranks violent windows best rather than the one that happens to get
the most predictions right on an unbalanced test set. Optimizer state is saved
alongside the weights so a run can be resumed.

---

## 6. Inference

`inference.py` is the command line path.

- Takes a video path as an argument.
- Runs the shared analysis engine.
- Writes an annotated output video.
- Leaves heatmaps, PDF generation, and email off by default, to keep the CLI
  light.

The real work happens in `modules/surveillance_analysis.py`, which the dashboard
uses as well. Having one engine behind both paths means a fix in either place
benefits the other.

Outputs from a run: the annotated video, a timeline CSV, snapshots of violent
windows, and optionally heatmaps and a PDF report.

Three bugs in this engine are worth recording, because each one changed the
output in a visible way:

- Checkpoints saved by `train.py` are dictionaries holding the model state, the
  optimizer state, and metrics. The loader originally passed the whole
  dictionary to `load_state_dict` and failed on every checkpoint.
- The annotated video wrote one still, the middle frame of the window, once per
  buffered frame. Every detection played back as a 32 frame freeze.
- Frames left in the buffer at the end of a video never fill a window, so they
  were scored and then dropped. The output video came out short by up to 31
  frames.

---

## 7. Dashboard

`dashboard/streamlit_app.py` is the main interface.

### 7.1 Input modes

- Upload a video
- Live webcam
- RTSP or IP camera

### 7.2 Controls

Detection threshold, confidence threshold, attention heatmap toggle, incident
report toggle, email alert toggle, and a recipient email field.

The UI deliberately asks only for the recipient address. The sender address and
app password are never shown or entered there.

### 7.3 Behaviour

For an uploaded video the app saves the file to a temporary folder, runs
`analyze_video`, shows progress and alert banners, transcodes for browser
playback where it can, displays the original and annotated videos side by side,
and renders the charts and timeline once processing finishes.

For live monitoring it opens the camera source, keeps a live preview updated,
shows current status, confidence, and latency, captures a snapshot on each
detection, and can email alerts and periodic summaries.

---

## 8. Alerts, reports, and email

### 8.1 Alerts

`modules/alerts.py` defines the alert payload and the snapshot helper. On a
detection the system builds a short readable summary, saves a snapshot of the
incident frame, and passes the alert to whatever callback the caller supplied.

### 8.2 Timeline

`modules/timeline.py` provides the `TimelineEvent` record, timestamp
formatting, merging of consecutive violent windows, and CSV export. Merging
matters because one long incident would otherwise appear as a run of separate 32
frame windows.

### 8.3 PDF reports

`modules/report_generator.py` builds the incident report: project title, video
metadata, processing date, model used, confidence summary, the event timeline as
a table, incident snapshots, heatmaps where available, and a recommendations
section.

### 8.4 Email

`modules/email_service.py` handles SMTP delivery over STARTTLS.

- Sender credentials come from the environment, never from the UI.
- Only the recipient address is collected in the dashboard.
- App passwords are stripped of spaces before login, since they are usually
  copied with them.
- Snapshots and PDF reports are attached with the correct MIME type.
- A failed send warns and returns `False` instead of raising. Before that, an
  unreachable host or a rejected login would kill an analysis halfway through a
  video.

---

## 9. Explainability

`modules/gradcam.py` produces attention rollout heatmaps from the DINOv2
backbone. Forward pre-hooks capture the input to each block's attention module,
attention is reconstructed from the qkv projection, and the CLS to patch
attention is averaged across blocks and heads before being rendered as a JET
colormap.

The implementation is deliberately defensive. Blocks without a usable attention
module are skipped, and if nothing can be captured at all it falls back to an
edge based map rather than failing. Contrast is stretched to the 5th and 95th
percentile with a 0.55 gamma, because attention concentrates in a handful of
patches and plain min and max scaling left most maps looking flat.

Explainability is treated as an extra, not a dependency. If rollout throws, the
analysis logs a warning and carries on.

---

## 10. Output artifacts

Runtime output goes into `outputs/`:

- `outputs/heatmaps/` for attention visualisations
- `outputs/reports/` for generated PDFs
- `outputs/snapshots/` for incident frames
- `outputs/temp/` for uploads and transcoding

Other generated files:

- `outputs/timeline.csv` for the event summary
- `outputs/email_state.json` for throttling periodic reports
- `train_output/training_log.csv` for training history
- `train_output/models/best_model.pt` for the best checkpoint

---

## 11. Configuration

`config.py` is the single place for project paths, preprocessing directories,
model settings, temporal hyperparameters, training hyperparameters, dashboard
thresholds, and email settings.

Credentials are read from the environment or from a gitignored `.env` file; see
`.env.example`. Nothing sensitive is stored in source. The only thing written to
disk is the chosen recipient address, in `outputs/email_recipient.json`.

Key values:

- DINOv2 model: `dinov2_vitb14`
- Feature dimension: 768
- Image size: 224
- Sequence length: 32
- Test split ratio: 0.2
- Default detection threshold: 0.80

---

## 12. Usage

### 12.1 Install

```bash
pip install -r requirements.txt
```

### 12.2 Preprocess

```bash
python utils/preprocessing/conv_video_to_img.py
python utils/preprocessing/conv_img_to_npy.py
python utils/preprocessing/forward_pass_extr.py
```

### 12.3 Train

```bash
python train/train.py
```

### 12.4 CLI inference

```bash
python inference.py path/to/video.mp4 --output output_annotated.mp4
```

### 12.5 Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

### 12.6 Email setup

- Copy `.env.example` to `.env` and set `SURVEILLANCE_EMAIL_SENDER` and
  `SURVEILLANCE_EMAIL_PASSWORD`. For Gmail that password must be an App
  Password, not the account password. `.env` is gitignored.
- Enter only the recipient address in the dashboard.
- Turn on email alerts before starting the analysis.

---

## 13. Results

Best run, epoch 13 of 22:

| Metric | Value |
| --- | --- |
| Train accuracy | 96.53% |
| Train loss | 0.1272 |
| Test accuracy | 86.72% |
| Test loss | 0.3087 |
| Test ROC AUC | 0.9436 |

Confusion matrix over 4,557 test windows:

```text
                 Predicted Normal   Predicted Violence
Actual Normal          2086                 338
Actual Violence         267                1866
```

Recall on the violent class is 1866 / 2133, or 87.5%, and precision is
1866 / 2204, or 84.7%. For an alerting workflow, missing an event costs more
than raising a false one, so leaning towards recall is the right trade. The ROC
AUC of 0.9436 says the model separates the two classes well across thresholds,
which is what makes the adjustable threshold in the dashboard useful.

---

## 14. Limitations

These are open issues, written down so the numbers above are read in context.

**Labels are per video, not per frame.** Every frame of an anomaly video gets
label 1, including the ordinary footage before and after the event. UCF-Crime
ships frame level annotations in
`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`, but the preprocessing
pipeline does not use them yet. The model therefore learns scene appearance
alongside the act itself, and the reported accuracy is optimistic compared to
true event localisation.

**Training and inference sample at different rates.** Training windows are built
from uniformly sampled frames, 14 to 70 per video, so 32 of them can span tens
of seconds. Inference buffers 32 consecutive frames, which is about one second
at 30 fps. The classifier sees much faster apparent motion in deployment than it
did in training. Lining these two up is the single most valuable improvement
left to make.

**Violence is defined broadly.** All 13 UCF-Crime anomaly categories are
collapsed into one positive class, including ones that are not violent at all,
such as Shoplifting and RoadAccidents.

**The head overfits the frozen features.** Train accuracy reaches 96.5% against
86.7% on test. Heavily overlapping stride 1 windows within each video push the
training figure up further than it should be.

---

## 15. Conclusion

This is a complete violence detection system rather than just a model file. It
pairs careful preprocessing with a frozen DINOv2 encoder, a temporal Transformer
head, a shared inference engine, a monitoring dashboard, PDF reporting, and
email alerting.

The design holds together because of a few decisions made early:

- Preprocessing is built for scale, with parallel extraction and chunked output.
- The split is at the video level, so the metrics are not inflated by leakage.
- Inference works on raw video, not on preprocessed arrays.
- Live monitoring and uploaded video share one backend.
- Reporting is generated from real runtime detections.

The clearest next step is to use the frame level annotations and match the
training and inference sampling rates. Both go after the same weakness, which is
that the model currently learns what a violent scene looks like more than it
learns what violence looks like.
