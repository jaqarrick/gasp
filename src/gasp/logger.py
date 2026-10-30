from rich.console import Console
from rich.table import Table


class Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.err = Console(stderr=True, force_terminal=True)
        self.out = Console(force_terminal=True)
        # currently info only prints in verbose mode, but could change later
        self.verbose = verbose

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
        self, title: str, columns: list[str], rows: list[list[str]], force: bool = False
    ) -> None:
        t = Table(title=title.upper(), title_style="bold cyan")
        for col in columns:
            t.add_column(col)
        for row in rows:
            t.add_row(*row)

        if self.verbose or force:
            self.out.print(t)
