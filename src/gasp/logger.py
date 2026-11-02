from datetime import datetime
from typing import TextIO

from rich.console import Console
from rich.table import Table

TRACE_MODULES: tuple[str, ...] = (
    "config",
    "cuda",
    "cfg",
    "sass",
    "occupancy",
    "pico",
    "dependency",
    "predict",
    "profiler",
)
DEFAULT_TRACE_FILE = "traces.log"


def format_table(columns: list[str], rows: list[list[str]]) -> str:
    """Plain-text table with left-aligned columns."""
    widths = [len(c) for c in columns]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths)).rstrip()

    out = [line(columns), line(["-" * w for w in widths])]
    out += [line(row) for row in rows]
    return "\n".join(out)


class Logger:
    def __init__(
        self,
        verbose: bool = False,
        traces: set[str] | None = None,
        trace_file: str = DEFAULT_TRACE_FILE,
    ) -> None:
        self.err = Console(stderr=True, force_terminal=True)
        self.out = Console(force_terminal=True)
        # currently info only prints in verbose mode, but could change later
        self.verbose = verbose
        self.traces: set[str] = set(traces or ())
        self.trace_file = trace_file
        # opened on the first trace so untraced runs leave no file behind
        self._trace_fh: TextIO | None = None

    def info(self, msg: str) -> None:
        if self.verbose:
            self.err.print(f"[blue]ℹnfo[/] {msg}")

    def success(self, msg: str) -> None:
        self.out.print(f"[green]success[/] {msg}")

    def error(self, msg: str) -> None:
        self.err.print(f"[red]error[/] {msg}")

    def warning(self, msg: str) -> None:
        self.err.print(f"[yellow]warn[/] {msg}")

    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
        force: bool = False,
        trace: str | None = None,
    ) -> None:
        # trace: module whose trace also gets this table
        if trace is not None:
            self.trace_table(trace, title, columns, rows)

        t = Table(title=title.upper(), title_style="bold cyan")
        for col in columns:
            t.add_column(col)
        for row in rows:
            t.add_row(*row)

        if self.verbose or force:
            self.out.print(t)

    def trace_enabled(self, module: str) -> bool:
        return "all" in self.traces or module in self.traces

    def trace(self, module: str, msg: str) -> None:
        if not self.trace_enabled(module):
            return

        if self._trace_fh is None:
            # stays open for the whole run, see the flush below
            self._trace_fh = open(self.trace_file, "w", encoding="utf-8")  # noqa: SIM115
            enabled = ", ".join(sorted(self.traces))
            self._trace_fh.write(
                f"# gasp trace {datetime.now().isoformat(timespec='seconds')}"
                f" (modules: {enabled})\n"
            )

        prefix = f"[{module}]"
        lines = msg.splitlines() or [""]
        self._trace_fh.write("".join(f"{prefix} {line}\n" for line in lines))
        # sys.exit is called throughout, so never rely on a close
        self._trace_fh.flush()

    def trace_table(
        self, module: str, title: str, columns: list[str], rows: list[list[str]]
    ) -> None:
        if self.trace_enabled(module):
            self.trace(module, f"=== {title} ===\n{format_table(columns, rows)}")
