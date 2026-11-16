import os
import re
import shutil
import subprocess
import sys
from typing import TypedDict

from gasp.logger import Logger

REQUIRED: list[str] = ["nvcc", "nvdisasm", "cuobjdump"]
# a CUDA install outside PATH, the usual place on Linux
DEFAULT_CUDA_BIN = "/usr/local/cuda/bin"


def cuda_hint() -> str:
    """How to get the CUDA tools onto PATH on this machine."""
    # Lmod and Environment Modules both set MODULESHOME, e.g. on HPC clusters
    if os.environ.get("MODULESHOME") or os.environ.get("LMOD_CMD"):
        return (
            "Load the CUDA module first, e.g. 'module load cuda' "
            "('module avail cuda' lists the versions)"
        )
    if os.path.isfile(os.path.join(DEFAULT_CUDA_BIN, "nvcc")):
        return f"CUDA is installed but not on PATH: add {DEFAULT_CUDA_BIN} to PATH"
    return "Install the CUDA Toolkit and add its bin/ to PATH"


# kernel name -> {"reg" | "shared" | "stack" | "local": value}
ResourceUsage = dict[str, dict[str, int]]


# label (.L_x_4, or a kernel name) -> (source file, line) of its first instruction
LabelLines = dict[str, tuple[str, int]]


class KernelInfo(TypedDict):
    raw_sass: str
    raw_cfg: str
    resources: ResourceUsage
    label_lines: LabelLines


_LABEL = re.compile(r"^([.\w$]+):\s*$")
_LINE_INFO = re.compile(r'//## File "([^"]+)", line (\d+)')
_INSTRUCTION = re.compile(r"^\s*/\*[0-9a-f]+\*/")


def parse_label_lines(disassembly: str) -> LabelLines:
    """Source line of each label's first instruction, from `nvdisasm -g` output.

    nvdisasm writes a `//## File` comment only when the line changes, so a label
    with none before its first instruction is still on the line seen last.
    """
    lines: LabelLines = {}
    current: tuple[str, int] | None = None
    pending: list[str] = []
    for text in disassembly.splitlines():
        if (info := _LINE_INFO.search(text)) is not None:
            current = (info.group(1), int(info.group(2)))
            for label in pending:
                lines[label] = current
            pending = []
        elif (label := _LABEL.match(text)) is not None:
            pending.append(label.group(1))
        elif _INSTRUCTION.match(text):
            if current is not None:
                for label in pending:
                    lines[label] = current
            pending = []
    return lines


class CUDAExtractor:
    def __init__(
        self,
        input_file: str,
        arch: str,
        logger: Logger,
        include_dirs: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> None:
        self.input_file = input_file
        self.logger = logger
        self.arch = arch
        self.include_dirs = include_dirs or []
        self.extra_flags = extra_flags or []
        self.cubin_path = input_file.replace(".cu", ".cubin")

    def check_packages(self) -> None:
        missing = [t for t in REQUIRED if not shutil.which(t)]
        if missing:
            self.logger.error(f"Missing packages: {', '.join(missing)}")
            self.logger.error(cuda_hint())
            sys.exit(1)

    def compile(self) -> None:
        cmd = [
            "nvcc",
            "-cubin",
            f"-arch={self.arch}",
            # source lines for the loop table; the SASS is the same without it
            "-lineinfo",
            *(f"-I{d}" for d in self.include_dirs),
            *self.extra_flags,
            self.input_file,
            "-o",
            self.cubin_path,
        ]
        self.logger.info(f"Compiling: {' '.join(cmd)}")
        self.logger.trace("cuda", f"nvcc command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            self.logger.error(f"nvcc failed:\n{result.stderr}")
            sys.exit(1)

    def disassemble(self) -> str:
        result = subprocess.run(
            ["nvdisasm", self.cubin_path], capture_output=True, check=False, text=True
        )

        if result.returncode != 0:
            self.logger.error(f"nvdisasm failed:\n{result.stderr}")
            sys.exit(1)
        return result.stdout

    def get_label_lines(self) -> LabelLines:
        """Source lines of the labels. Only used to describe loops, so a failure
        only warns."""
        result = subprocess.run(
            ["nvdisasm", "-g", self.cubin_path],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            self.logger.warning(
                f"nvdisasm -g failed, no source lines:\n{result.stderr}"
            )
            return {}
        return parse_label_lines(result.stdout)

    def get_cfg(self) -> str:
        result = subprocess.run(
            ["nvdisasm", "-bbcfg", self.cubin_path],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(f"nvdisasm -bbcfg failed:\n{result.stderr}")
            sys.exit(1)
        return result.stdout

    def get_resource_usage(self) -> ResourceUsage:
        result = subprocess.run(
            ["cuobjdump", "-res-usage", self.cubin_path],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(f"cuobjdump failed:\n{result.stderr}")
            sys.exit(1)

        kernels: ResourceUsage = {}
        current_kernel: str | None = None
        for line in result.stdout.splitlines():
            fn_match = re.search(r"Function\s+(\S+)", line)
            if fn_match:
                current_kernel = fn_match.group(1).rstrip(":")
                kernels[current_kernel] = {}

            if current_kernel:
                for field in ["REG", "SHARED", "STACK", "LOCAL"]:
                    m = re.search(rf"{field}:(\d+)", line)
                    if m:
                        kernels[current_kernel][field.lower()] = int(m.group(1))
        for kernel, usage in kernels.items():
            self.logger.trace("cuda", f"resource usage {kernel} {usage}")
        return kernels

    def process(self) -> KernelInfo:
        self.logger.info("Checking CUDA toolkit")
        self.check_packages()
        self.logger.info("Compiling CUDA to cubin")
        self.compile()
        self.logger.info("Extracting raw SASS")
        sass = self.disassemble()
        self.logger.info("Extracting raw CFG")
        raw_cfg = self.get_cfg()
        self.logger.info("Extracting kernel resource usage")
        resources = self.get_resource_usage()
        self.logger.info("Extracting source lines")
        label_lines = self.get_label_lines()
        kernel_info: KernelInfo = {
            "raw_sass": sass,
            "raw_cfg": raw_cfg,
            "resources": resources,
            "label_lines": label_lines,
        }

        return kernel_info
