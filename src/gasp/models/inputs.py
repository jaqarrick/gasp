"""The inputs every performance model (gasp.models.gpuperf, gasperf) runs on."""

from dataclasses import dataclass

# bytes per thread per memory instruction, for MWP_peak_bw
WORD_BYTES = 4


@dataclass
class ModelInputs:
    """Every input the model takes from static analysis, per warp where it's
    a count. predict() fills this from the static analyses, applies the
    overrides to it once, and the models derive everything else from the result."""

    total_inst_count: float  # every instruction, SFU ones included
    mem_inst_count: float
    sfu_inst_count: float
    sync_inst_count: float
    divergent_inst_count: float
    ilp: float
    mlp: float
    average_inst_latency: float  # over the non-SFU instructions
    average_trans_warp: float
    bank_conflict_count: float
    l1_hit_ratio: float
    l2_hit_ratio: float
