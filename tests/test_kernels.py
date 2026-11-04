"""Tests for gasp.kernels — kernel names and selection."""

import pytest

from gasp.kernels import (
    KernelSelectionError,
    kernel_base_name,
    resolve_kernel,
    source_base_name,
)

# names as cuobjdump -res-usage prints them
PLAIN = "_Z5plainPfS_"
NAMESPACED = "_ZN2ns3nskEPf"
TEMPLATE_INT = "_Z2tkIiEvPT_"
TEMPLATE_FLOAT = "_Z2tkIfEvPT_"
C_NAME = "cname"


@pytest.mark.parametrize(
    "mangled,expected",
    [
        (PLAIN, "plain"),
        (NAMESPACED, "nsk"),
        ("_ZL3fooPf", "foo"),
        (TEMPLATE_FLOAT, "tk"),
        ("_ZN2ns2tkIfEEvPT_", "tk"),
        (C_NAME, "cname"),
    ],
)
def test_kernel_base_name(mangled, expected):
    assert kernel_base_name(mangled) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("plain", "plain"),
        ("ns::nsk", "nsk"),
        ("tk<float>", "tk"),
        ("ns::tk<a::b>", "tk"),
    ],
)
def test_source_base_name(name, expected):
    assert source_base_name(name) == expected


def test_resolve_single_kernel_without_selector():
    assert resolve_kernel(None, [PLAIN]) == PLAIN


def test_resolve_several_kernels_without_selector_lists_them():
    with pytest.raises(KernelSelectionError) as e:
        resolve_kernel(None, [PLAIN, C_NAME])
    assert PLAIN in str(e.value)
    assert C_NAME in str(e.value)


@pytest.mark.parametrize(
    "selector,expected",
    [
        (PLAIN, PLAIN),
        ("plain", PLAIN),
        ("nsk", NAMESPACED),
        ("ns::nsk", NAMESPACED),
        (C_NAME, C_NAME),
        (TEMPLATE_INT, TEMPLATE_INT),
    ],
)
def test_resolve_selector(selector, expected):
    kernels = [PLAIN, NAMESPACED, C_NAME, TEMPLATE_INT, TEMPLATE_FLOAT]
    assert resolve_kernel(selector, kernels) == expected


def test_resolve_unknown_selector_lists_kernels():
    with pytest.raises(KernelSelectionError) as e:
        resolve_kernel("missing", [PLAIN])
    assert PLAIN in str(e.value)


def test_resolve_ambiguous_selector_lists_mangled_names():
    with pytest.raises(KernelSelectionError) as e:
        resolve_kernel("tk", [TEMPLATE_INT, TEMPLATE_FLOAT, PLAIN])
    assert TEMPLATE_INT in str(e.value)
    assert TEMPLATE_FLOAT in str(e.value)
    assert PLAIN not in str(e.value)


def test_resolve_no_kernels():
    with pytest.raises(KernelSelectionError):
        resolve_kernel(None, [])
