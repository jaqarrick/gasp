import re
import shutil
import subprocess
import sys
from typing import TypedDict

from gasp.logger import Logger

REQUIRED: list[str] = ["nvcc", "nvdisasm", "cuobjdump"]
# kernel name -> {"reg" | "shared" | "stack" | "local": value}
ResourceUsage = dict[str, dict[str, int]]


class KernelInfo(TypedDict):
    raw_sass: str
    raw_cfg: str
    resources: ResourceUsage


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
            self.logger.error("Install the CUDA Toolkit (nvidia-cuda-toolkit)")
            sys.exit(1)

    def compile(self) -> None:
        cmd = [
            "nvcc",
            "-cubin",
            f"-arch={self.arch}",
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
        kernel_info: KernelInfo = {
            "raw_sass": sass,
            "raw_cfg": raw_cfg,
            "resources": resources,
        }

        return kernel_info
