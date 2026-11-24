"""The performance models, each a module whose run() takes inputs.ModelInputs.

gpuperf: GPUPerf as published (Sim et al.), plus the SFU fix. gasperf: our model.
"""

from . import gasperf, gpuperf

# by name, in the order results are shown
MODELS = {"GPUPerf": gpuperf, "GASPerf": gasperf}
DEFAULT_MODEL = "GPUPerf"

# how the tables name each model
DISPLAY_NAMES = {"GPUPerf": "GPUPerf", "GASPerf": "GASP"}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)
