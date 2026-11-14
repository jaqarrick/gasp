from gasp.common.constants import GPU_TO_ARCH


def test_known_gpus():
    assert GPU_TO_ARCH["b200"] == "sm_100"
    assert GPU_TO_ARCH["h100"] == "sm_90"
    assert GPU_TO_ARCH["a100"] == "sm_80"
    assert GPU_TO_ARCH["l4"] == "sm_89"
    assert GPU_TO_ARCH["v100"] == "sm_70"
    assert GPU_TO_ARCH["rtx2080ti"] == "sm_75"
    assert GPU_TO_ARCH["titanv"] == "sm_70"
    assert GPU_TO_ARCH["rtx4070"] == "sm_89"


def test_unknown_gpu():
    assert GPU_TO_ARCH.get("foo") is None
