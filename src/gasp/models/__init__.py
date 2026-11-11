"""The performance models. Each one is a module with run(), which takes the
shared inputs (inputs.ModelInputs) to the model's derived parameters and times.

gpuperf: GPUPerf as published (Sim et al.), plus the SFU fix.
gasperf: our model. It starts as a copy of GPUPerf and is ours to change."""

from . import gasperf, gpuperf

# by name, in the order results are shown
MODELS = {"GPUPerf": gpuperf, "GASPerf": gasperf}
DEFAULT_MODEL = "GPUPerf"
