import re
from collections.abc import Collection

_MANGLED_PREFIX = re.compile(r"_Z(N?)L?")
_IDENTIFIER_LENGTH = re.compile(r"\d+")


class KernelSelectionError(ValueError):
    pass


def kernel_base_name(kernel: str) -> str:
    """Source name of a kernel, without namespaces or template arguments.

    _Z5plainPfS_ -> plain, _ZN2ns3nskEPf -> nsk, _Z2tkIfEvPT_ -> tk, cname -> cname
    """
    prefix = _MANGLED_PREFIX.match(kernel)
    if prefix is None:
        return kernel

    nested = prefix.group(1) == "N"
    pos = prefix.end()
    name = kernel
    while (length := _IDENTIFIER_LENGTH.match(kernel, pos)) is not None:
        start = length.end()
        pos = start + int(length.group())
        name = kernel[start:pos]
        if not nested:
            break
    return name


def source_base_name(name: str) -> str:
    """Base name of a source-style name, as Pico reports: ns::tk<float> -> tk"""
    return name.split("<", 1)[0].rsplit("::", 1)[-1]


def _listing(kernels: Collection[str]) -> str:
    return ", ".join(sorted(kernels))


def resolve_kernel(selector: str | None, kernels: Collection[str]) -> str:
    """Mangled name of the kernel a selector picks out of a compiled file.

    The selector is a mangled name or a source name. None is only accepted when
    the file holds a single kernel.
    """
    if not kernels:
        raise KernelSelectionError("no kernels found in the compiled file")

    if selector is None:
        if len(kernels) > 1:
            raise KernelSelectionError(
                f"the file holds several kernels, select one: {_listing(kernels)}"
            )
        return next(iter(kernels))

    if selector in kernels:
        return selector

    base = source_base_name(selector)
    matches = [k for k in kernels if kernel_base_name(k) == base]
    if not matches:
        raise KernelSelectionError(
            f"no kernel named {selector!r}, the file holds: {_listing(kernels)}"
        )
    if len(matches) > 1:
        raise KernelSelectionError(
            f"kernel name {selector!r} is ambiguous, "
            f"select one by its mangled name: {_listing(matches)}"
        )
    return matches[0]
