"""Tests for gasp.logger — module-scoped traces."""

from gasp.logger import TRACE_MODULES, Logger, format_table


def test_trace_disabled_writes_no_file(tmp_path):
    trace_file = tmp_path / "traces.log"
    logger = Logger(trace_file=str(trace_file))

    logger.trace("sass", "IMAD R6, R6, c[0x0][0x0], R3")

    assert not logger.trace_enabled("sass")
    assert not trace_file.exists()


def test_trace_all_enables_every_module(tmp_path):
    logger = Logger(traces={"all"}, trace_file=str(tmp_path / "traces.log"))

    assert all(logger.trace_enabled(m) for m in TRACE_MODULES)


def test_trace_filters_by_module(tmp_path):
    trace_file = tmp_path / "traces.log"
    logger = Logger(traces={"cfg", "sass"}, trace_file=str(trace_file))

    logger.trace("cfg", "cfg line")
    logger.trace("sass", "sass line")
    logger.trace("profiler", "profiler line")

    text = trace_file.read_text()
    assert "[cfg] cfg line" in text
    assert "[sass] sass line" in text
    assert "profiler line" not in text


def test_trace_prefixes_every_line(tmp_path):
    trace_file = tmp_path / "traces.log"
    logger = Logger(traces={"cfg"}, trace_file=str(trace_file))

    logger.trace("cfg", "first\nsecond")

    lines = trace_file.read_text().splitlines()
    assert lines[0].startswith("# gasp trace")
    assert lines[1:] == ["[cfg] first", "[cfg] second"]


def test_trace_file_is_overwritten_per_logger(tmp_path):
    trace_file = tmp_path / "traces.log"
    Logger(traces={"all"}, trace_file=str(trace_file)).trace("cfg", "old run")
    Logger(traces={"all"}, trace_file=str(trace_file)).trace("cfg", "new run")

    text = trace_file.read_text()
    assert "old run" not in text
    assert "new run" in text


def test_table_with_trace_writes_plain_table(tmp_path):
    trace_file = tmp_path / "traces.log"
    logger = Logger(traces={"occupancy"}, trace_file=str(trace_file))

    logger.table(
        "Occupancy", ["Parameter", "Value"], [["MWP", "8.00"]], trace="occupancy"
    )

    text = trace_file.read_text()
    assert "[occupancy] === Occupancy ===" in text
    assert "[occupancy] MWP        8.00" in text


def test_format_table_aligns_columns():
    assert format_table(["Name", "V"], [["a", "100"], ["long name", "1"]]) == (
        "Name       V\n---------  ---\na          100\nlong name  1"
    )
