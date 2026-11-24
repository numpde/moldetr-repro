"""Batched worker requests and replies for MolDeTr multiplet detection.

PredictionRequest supplies amplitudes and noise settings to moldetr.inference.run,
then calibration and threshold settings to moldetr.postprocess.decode_predictions.
MultipletPrediction mirrors the dictionaries returned by that decoder. These
declarations are shared by the provider and worker without scientific imports.
"""

from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict
from pydantic import JsonValue


@dataclass
class PredictionRequest:
    """One real, finite 6144-point spectrum sampled at 5.12 points/Hz.

    Provide both ppm endpoints to obtain calibrated ppm shifts. Without them,
    shifts are reported in points and Hz. Noise settings retain upstream's
    seeded preprocessing defaults; threshold retains its prediction CLI default.

    Upstream input, preprocessing, and decode contracts:
    https://github.com/numpde/fork-of-MolDeTr/blob/cdbbb8c667ced8eb1d7c547d4cdce93a14987af4/moldetr/validation.py#L28
    https://github.com/numpde/fork-of-MolDeTr/blob/cdbbb8c667ced8eb1d7c547d4cdce93a14987af4/moldetr/inference.py#L149
    https://github.com/numpde/fork-of-MolDeTr/blob/cdbbb8c667ced8eb1d7c547d4cdce93a14987af4/moldetr/postprocess.py#L51
    """

    amplitudes: list[float]
    noise_seed: int = 0
    noise_frac: float = 0.005
    points_per_hz: float = 5.12
    ppm_left: float | None = None
    ppm_right: float | None = None
    threshold: float = 0.3


class MultipletPrediction(TypedDict):
    """A decoded detection with upstream's field names and physical units.

    coupling_constants_hz contains only the largest inferred coupling, or is
    empty when that coupling is below the decoder's cutoff. It does not describe
    the complete coupling pattern.

    Upstream prediction dictionary:
    https://github.com/numpde/fork-of-MolDeTr/blob/cdbbb8c667ced8eb1d7c547d4cdce93a14987af4/moldetr/postprocess.py#L83
    """

    proton_count: Literal[1, 2, 3, 4, 6]
    confidence: float
    chemical_shift_in_points: float
    chemical_shift_hz: float
    chemical_shift_ppm: float | None
    linewidth_hz: float
    coupling_constants_hz: list[float]


@dataclass
class PredictionResult:
    """Decoded detections in ascending chemical_shift_in_points order."""

    predictions: list[MultipletPrediction]


@dataclass
class ModelFailure:
    """A completed model call raised; its diagnostic belongs to this batch entry."""

    exception_type: str
    message: str
    diagnostic: dict[str, JsonValue]


@dataclass
class PredictionBatch:
    """Requests processed sequentially in their supplied order."""

    requests: list[PredictionRequest]


@dataclass
class PredictionBatchResult:
    """One reply per request, in the same order; model failures do not stop later entries.

    A process or transport failure can prevent the entire reply from arriving.
    That does not authorize replaying a batch whose earlier entries may have run.
    """

    replies: list[PredictionResult | ModelFailure]
