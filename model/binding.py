"""Declare the inference operation in both containers; load science only in the worker."""

import hashlib
import json
import logging
import pathlib
from typing import TYPE_CHECKING

import provider_core.scientific as sdk
from provider_core.attempt_record import MAX_EVIDENCE_BYTES
from provider_core.diagnostics import describe_core_failure, failure_evidence_json

if __package__:
    from .wire import PredictionBatch, PredictionBatchResult, PredictionResult, ModelFailure
else:
    from wire import PredictionBatch, PredictionBatchResult, PredictionResult, ModelFailure

if TYPE_CHECKING:
    from torch.nn import Module

log = logging.getLogger(__name__)


class Inference(sdk.Contribution):
    def __init__(self, *, checkpoint_path: str, device: str) -> None:
        super().__init__()
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.device = device
        self.model: Module | None = None
        self.extrema: dict | None = None

    def initialize(self, *, assets: sdk.AssetAccess) -> None:
        """Verify the checkpoint, then initialize the model before readiness."""
        import torch
        from moldetr import inference
        from moldetr.postprocess import load_extrema

        device = torch.device(self.device)

        if not (device.type in {"cpu", "cuda"}):
            raise ValueError(f"Unsupported {device=}: MolDeTr requires CPU or CUDA execution")

        checkpoint_path = self.checkpoint_path
        marker_path = checkpoint_path.with_name(checkpoint_path.name + ".sanitized.json")
        expected_sha256 = json.loads(marker_path.read_text())["files"][checkpoint_path.name]

        with checkpoint_path.open("rb") as checkpoint:
            actual_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()

            if not (actual_sha256 == expected_sha256):
                raise ValueError(f"Cannot load {checkpoint_path=}: {actual_sha256=} differs from {expected_sha256=} in {marker_path=}")

            checkpoint.seek(0)
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

        model = inference.build_model()

        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        extrema = load_extrema(pathlib.Path(inference.__file__).with_name("assets") / "extrema.txt")
        self.extrema = extrema
        self.model = model

        log.info(f"Model ready on {self.device=}")

    @sdk.operation()
    def predict_batch(self, *, batch: PredictionBatch) -> PredictionBatchResult:
        """Run entries sequentially and retain each completed result or model exception."""
        from moldetr.inference import run as run_inference
        from moldetr.postprocess import decode_predictions

        replies = []

        for (index, prediction) in enumerate(batch.requests):
            log.info(f"Starting batch entry {index=} with {prediction.noise_seed=}")

            try:
                if not (0 < prediction.points_per_hz):
                    raise ValueError(f"Expected positive sampling density; {prediction.points_per_hz=}")

                # Upstream run normalizes on CPU. Move its input to the selected
                # device while retaining its preprocessing and output reshaping.
                output = run_inference(
                    lambda spectrum: self.model(spectrum.to(self.device)),
                    prediction.amplitudes,
                    noise_seed=prediction.noise_seed,
                    noise_frac=prediction.noise_frac,
                )

                decoded_predictions = decode_predictions(
                    output,
                    self.extrema,
                    prediction.points_per_hz,
                    ppm_left=prediction.ppm_left,
                    ppm_right=prediction.ppm_right,
                    threshold=prediction.threshold,
                )

                prediction_reply = PredictionResult(predictions=decoded_predictions)
            except Exception as error:
                # todo: Review exception text for partial scientific-input echoes;
                # complete spectrum encodings are redacted; locals are never captured.
                diagnostic = json.loads(
                    failure_evidence_json(
                        error,
                        secrets=(
                            str(prediction.amplitudes),
                            json.dumps(prediction.amplitudes),
                        ),
                        maximum_bytes=MAX_EVIDENCE_BYTES // 2,
                        describe_owner=lambda failure, text, window: {
                            **describe_core_failure(failure, text, window),
                            "message": str(failure),
                        },
                    ),
                )

                log.error(f"Batch entry failed; {index=}; {diagnostic=}")

                prediction_reply = ModelFailure(
                    exception_type=diagnostic["exception_type"],
                    message=diagnostic["message"],
                    diagnostic=diagnostic,
                )
            else:
                prediction_count = len(decoded_predictions)

                log.info(f"Batch entry completed: {index=}; {prediction_count=}")

            replies.append(prediction_reply)

        return PredictionBatchResult(replies=replies)

    def close(self) -> None:
        """Release model references before the SDK retires the worker process."""
        self.model = None
        self.extrema = None

        log.info("Model closed")


def construct(*, startup: dict) -> Inference:
    return Inference(**startup)
