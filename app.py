"""TrailTone: local-first bird-call identification with BirdNET."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd


MODEL_VERSION = os.getenv("TRAILTONE_MODEL_VERSION", "3.0")
MODEL_BACKEND = os.getenv("TRAILTONE_MODEL_BACKEND", "onnx")
MAX_AUDIO_BYTES = 25 * 1024 * 1024


@lru_cache(maxsize=1)
def load_model():
    """Load once per process. BirdNET caches downloaded model files locally."""
    try:
        import birdnet
    except ImportError as exc:
        raise RuntimeError(
            "BirdNET is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return birdnet.load("acoustic", MODEL_VERSION, MODEL_BACKEND)


def _common_name(species_name: str) -> str:
    """BirdNET labels are typically 'scientific_name_Common name'."""
    return species_name.split("_", maxsplit=1)[-1].strip()


def identify(audio_path: str | None, top_k: int, min_confidence: float):
    if not audio_path:
        raise gr.Error("Record a short clip or choose an audio file first.")

    path = Path(audio_path)
    if not path.is_file():
        raise gr.Error("The selected audio file could not be read.")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise gr.Error("Please use a clip smaller than 25 MB.")

    try:
        predictions = load_model().predict(str(path))
    except Exception as exc:  # Surface a readable message for offline/model errors.
        raise gr.Error(
            "Could not analyze this recording. On first use, connect once so the "
            "BirdNET model can be downloaded; later runs can use the cached model "
            "offline. Details: " + str(exc)
        ) from exc

    if predictions is None or predictions.empty:
        return "No bird detections found. Try a clearer recording with less wind.", pd.DataFrame(
            columns=["Start", "End", "Bird", "Confidence"]
        )

    detections = predictions.copy()
    detections["confidence"] = pd.to_numeric(detections["confidence"], errors="coerce")
    detections = detections.dropna(subset=["confidence"])
    detections = detections[detections["confidence"] >= min_confidence]
    if detections.empty:
        return (
            "No detections met that confidence threshold. Lower it a little or try "
            "another recording.",
            pd.DataFrame(columns=["Start", "End", "Bird", "Confidence"]),
        )

    # A species may appear in several 3-second windows. Keep its strongest window.
    strongest = (
        detections.sort_values("confidence", ascending=False)
        .drop_duplicates("species_name")
        .head(int(top_k))
    )
    lines = ["## Likely birds in this recording", ""]
    for rank, row in enumerate(strongest.itertuples(index=False), start=1):
        name = _common_name(str(row.species_name))
        lines.append(f"{rank}. **{name}** — {float(row.confidence):.0%} confidence")
    lines += ["", "*These are model predictions, not confirmed sightings. Listen and look around to verify.*"]

    table = detections.copy()
    table["Bird"] = table["species_name"].map(lambda name: _common_name(str(name)))
    table = table.rename(
        columns={"start_time": "Start", "end_time": "End", "confidence": "Confidence"}
    )
    table["Confidence"] = table["Confidence"].map(lambda value: f"{float(value):.0%}")
    wanted = [column for column in ("Start", "End", "Bird", "Confidence") if column in table]
    table = table[wanted].sort_values("Start", kind="stable")
    return "\n".join(lines), table


with gr.Blocks(title="TrailTone · Listen closer") as demo:
    gr.Markdown(
        """
        # TrailTone
        ### A little help hearing what’s around you.

        Record a short birdsong or upload a clip. TrailTone runs BirdNET on this
        machine and shows possible matches. Check the result, then put your phone
        away and keep listening.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                label="Birdsong recording",
                sources=["microphone", "upload"],
                type="filepath",
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Birds to show")
            min_confidence = gr.Slider(
                0.05, 0.95, value=0.20, step=0.05, label="Minimum confidence"
            )
            identify_button = gr.Button("Identify birds", variant="primary")
        with gr.Column(scale=1):
            result = gr.Markdown("Your likely matches will appear here.")
            timeline = gr.Dataframe(
                label="Detections by moment",
                headers=["Start", "End", "Bird", "Confidence"],
                interactive=False,
                wrap=True,
            )

    identify_button.click(
        identify,
        inputs=[audio, top_k, min_confidence],
        outputs=[result, timeline],
        api_name="identify",
    )
    gr.Markdown(
        "**Privacy:** this app does not save recordings or send them to an inference "
        "API. Run it on your own machine for local processing. First launch downloads "
        "the model; after that, the cached model works offline."
    )


if __name__ == "__main__":
    # Hugging Face Spaces requires the app to listen on all container interfaces.
    # Keep local development private by default when running outside Spaces.
    default_host = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    enable_share = os.getenv("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", default_host),
        server_port=int(os.getenv("PORT", "7860")),
        share=enable_share,
    )
