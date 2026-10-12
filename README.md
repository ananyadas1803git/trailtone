---
title: TrailTone
emoji: 🐦
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
python_version: 3.11
pinned: false
---

# TrailTone

**Listen closer. Look up. Keep walking.**

TrailTone is a local-first bird-call identification demo for short walks. Record a
clip or choose an audio file, get BirdNET's likely matches and confidence scores,
then put the screen away and see whether you can spot or hear the bird yourself.

## Why open AI

TrailTone uses the open-source BirdNET Python library and its acoustic model. The
model runs on the machine running this app, so recordings do not need to be sent to
a hosted inference API. You can inspect and change the code, select a different
BirdNET model backend, or build on the detections for a local species guide.

The first launch downloads and caches the model. After the model is cached, analysis
can run offline. The app itself binds to `127.0.0.1` by default and does not enable
Gradio's public sharing feature.

> **Model license:** BirdNET's code is MIT licensed, while BirdNET model files are
> CC BY-NC-SA 4.0. Review the model terms before redistributing model files or using
> the model commercially. See the [BirdNET project](https://github.com/birdnet-team/birdnet/).

## Run locally

Use Python 3.11–3.13 for the smoothest package compatibility.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>. On first use, connect to the internet so BirdNET can
download its model files. Then you can disconnect and use the cached model offline.
The app accepts audio from your microphone or a file upload. It limits input files
to 25 MB and does not intentionally retain recordings after the Gradio session.

## Deploy a demo

For a shareable demo, create a Hugging Face Space with the **Gradio** SDK and upload
`app.py`, `requirements.txt`, and this README. The app detects the Space environment
and binds to the host required by its container. A hosted Space processes uploaded
audio on its server, so use the local version for private or sensitive recordings.
The model license is non-commercial; check current hosting and model terms before
publishing or using a Space commercially.

## Project structure

```text
app.py             Gradio interface and BirdNET inference
requirements.txt   Python dependencies
README.md          Setup, open-AI rationale, and deployment notes
```

## Research and next steps

- Build a small labeled evaluation set from recordings you have permission to use.
- Report top-1/top-5 accuracy, false positives, and inference latency on a laptop.
- Compare model backends and confidence thresholds on the same held-out clips.
- Add an optional local walk journal with user-confirmed sightings.
- Add a region-specific shortlist while keeping model predictions clearly labeled.

## Acknowledgment

TrailTone builds on [BirdNET](https://github.com/birdnet-team/birdnet/). Please cite
the BirdNET research paper when presenting results based on its model.
