"""The performance models. Each one is a module with run(), which takes the
shared inputs (inputs.ModelInputs) to the model's derived parameters and times.

gpuperf: GPUPerf as published (Sim et al.), plus the SFU fix.
gasperf: our model. It starts as a copy of GPUPerf and is ours to change.
Only GASPerf adds launch overhead to T_exec for T_GPU."""

from . import gasperf, gpuperf

# by name, in the order results are shown
MODELS = {"GPUPerf": gpuperf, "GASPerf": gasperf}
DEFAULT_MODEL = "GPUPerf"

# how the tables name each model
DISPLAY_NAMES = {"GPUPerf": "GPUPerf", "GASPerf": "GASP"}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)
