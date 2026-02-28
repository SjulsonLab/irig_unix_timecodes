"""Tests for MATLAB ClockTable loader and decode_irig wrapper.

Tests call MATLAB via subprocess (-batch mode). Each test generates a
.clocktable.npz fixture in Python, then verifies that the MATLAB code
produces matching results.

Requires MATLAB R2025b at /Applications/MATLAB_R2025b.app/bin/matlab.
All tests are skipped if MATLAB is not found.
"""

import json
import os
import subprocess
import textwrap

import numpy as np
import pytest

from neurokairos.clock_table import ClockTable

# -- MATLAB path and skip logic -----------------------------------------------

MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "matlab"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Skip entire module if MATLAB is not installed
pytestmark = pytest.mark.skipif(
    not os.path.isfile(MATLAB_BIN),
    reason=f"MATLAB not found at {MATLAB_BIN}",
)


def matlab_run(code, timeout=120):
    """Run MATLAB code via -batch and return stdout.

    MATLAB -batch silently drops output when the command contains literal
    newlines, so we join all lines with spaces to form a single-line command.
    """
    # Join multi-line code into a single line
    oneliner = " ".join(line.strip() for line in code.splitlines() if line.strip())
    full_code = f"addpath('{MATLAB_DIR}'); {oneliner}"
    result = subprocess.run(
        [MATLAB_BIN, "-batch", full_code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
    return result.stdout


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def clock_table_npz(tmp_path):
    """Create a .clocktable.npz file with known values for MATLAB to load."""
    source = np.array([0.0, 30000.0, 60000.0, 90000.0, 120000.0])
    reference = np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    metadata = {
        "source_file": "test_recording.dat",
        "n_channels": 3,
        "recording_start": "2025-01-16T02:40:00Z",
    }
    stratum = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
    dispersion = np.array([0.25, 0.25, 1.0, 1.0, 0.5])

    ct = ClockTable(
        source=source,
        reference=reference,
        nominal_rate=30000.0,
        source_units="samples",
        metadata=metadata,
        sync_stratum=stratum,
        sync_dispersion_upperbound_ms=dispersion,
    )
    path = tmp_path / "test.clocktable.npz"
    ct.save(path)

    return {
        "path": str(path),
        "ct": ct,
        "source": source,
        "reference": reference,
        "metadata": metadata,
        "stratum": stratum,
        "dispersion": dispersion,
    }


# -- ClockTable loading tests --------------------------------------------------


def test_load_clock_table(clock_table_npz):
    """MATLAB can load a .clocktable.npz and read source/reference/rate."""
    npz_path = clock_table_npz["path"]
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        fprintf('source: %s\\n', mat2str(ct.source', 15));
        fprintf('reference: %s\\n', mat2str(ct.reference', 15));
        fprintf('nominal_rate: %.1f\\n', ct.nominal_rate);
        fprintf('source_units: %s\\n', ct.source_units);
    """)
    stdout = matlab_run(code)
    assert "source: [0 30000 60000 90000 120000]" in stdout
    assert "nominal_rate: 30000.0" in stdout
    assert "source_units: samples" in stdout


def test_source_to_reference(clock_table_npz):
    """MATLAB source_to_reference matches Python output."""
    npz_path = clock_table_npz["path"]
    ct = clock_table_npz["ct"]

    # Test at exact anchor and interpolated points
    test_values = [0.0, 15000.0, 30000.0, 75000.0, 120000.0]
    py_results = ct.source_to_reference(np.array(test_values))

    values_str = mat_array_str(test_values)
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        vals = {values_str};
        result = ct.source_to_reference(vals);
        fprintf('result: %s\\n', mat2str(result, 15));
    """)
    stdout = matlab_run(code)
    matlab_result = parse_mat_array(stdout, "result")
    np.testing.assert_allclose(matlab_result, py_results, atol=1e-9)


def test_reference_to_source(clock_table_npz):
    """MATLAB reference_to_source matches Python output."""
    npz_path = clock_table_npz["path"]
    ct = clock_table_npz["ct"]

    test_values = [1000.0, 1000.5, 1002.0, 1003.5, 1004.0]
    py_results = ct.reference_to_source(np.array(test_values))

    values_str = mat_array_str(test_values)
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        vals = {values_str};
        result = ct.reference_to_source(vals);
        fprintf('result: %s\\n', mat2str(result, 15));
    """)
    stdout = matlab_run(code)
    matlab_result = parse_mat_array(stdout, "result")
    np.testing.assert_allclose(matlab_result, py_results, atol=1e-9)


def test_extrapolation_within_limit(clock_table_npz):
    """MATLAB extrapolation within 1.5 s matches Python."""
    npz_path = clock_table_npz["path"]
    ct = clock_table_npz["ct"]

    # 1.0 s below the first anchor (30000 samples below source[0]=0)
    # source starts at 0, so query at -30000 = 1.0 s below
    test_val = -30000.0
    py_result = ct.source_to_reference(test_val)

    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        result = ct.source_to_reference({test_val});
        fprintf('result: %.15g\\n', result);
    """)
    stdout = matlab_run(code)
    matlab_result = float(extract_value(stdout, "result"))
    np.testing.assert_allclose(matlab_result, py_result, atol=1e-9)


def test_extrapolation_beyond_limit_nan(clock_table_npz):
    """MATLAB returns NaN for queries beyond the 1.5 s extrapolation limit."""
    npz_path = clock_table_npz["path"]

    # 5.0 s below: source[0]=0, query at -150000 = 5 s below
    test_val = -150000.0
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        result = ct.source_to_reference({test_val});
        fprintf('isnan: %d\\n', isnan(result));
    """)
    stdout = matlab_run(code)
    assert "isnan: 1" in stdout


def test_sync_arrays_loaded(clock_table_npz):
    """MATLAB loads sync_stratum and sync_dispersion arrays correctly."""
    npz_path = clock_table_npz["path"]
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        fprintf('stratum: %s\\n', mat2str(ct.sync_stratum', 15));
        fprintf('dispersion: %s\\n', mat2str(ct.sync_dispersion_upperbound_ms', 15));
    """)
    stdout = matlab_run(code)
    assert "stratum: [1 1 2 2 1]" in stdout
    assert "dispersion: [0.25 0.25 1 1 0.5]" in stdout


def test_metadata_loaded(clock_table_npz):
    """MATLAB loads metadata as a struct with correct fields."""
    npz_path = clock_table_npz["path"]
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        fprintf('source_file: %s\\n', ct.metadata.source_file);
        fprintf('n_channels: %d\\n', ct.metadata.n_channels);
        fprintf('recording_start: %s\\n', ct.metadata.recording_start);
    """)
    stdout = matlab_run(code)
    assert "source_file: test_recording.dat" in stdout
    assert "n_channels: 3" in stdout
    assert "recording_start: 2025-01-16T02:40:00Z" in stdout


def test_display(clock_table_npz):
    """disp(ct) runs without error."""
    npz_path = clock_table_npz["path"]
    code = textwrap.dedent(f"""\
        ct = ClockTable.load('{npz_path}');
        disp(ct);
        fprintf('disp_ok: 1\\n');
    """)
    stdout = matlab_run(code)
    assert "disp_ok: 1" in stdout


def test_decode_sglx(sglx_nidq, tmp_path):
    """decode_irig decodes a SpikeGLX .bin file and matches Python output."""
    from neurokairos.decoders.sglx import decode_sglx_irig

    bin_path = str(sglx_nidq["bin_path"])

    # Get Python reference
    py_ct = decode_sglx_irig(bin_path, irig_channel="sync")

    # Get the path of the .clocktable.npz that Python saved
    npz_path = bin_path.replace(".bin", ".clocktable.npz")

    code = textwrap.dedent(f"""\
        ct = decode_irig('{bin_path}', 'format', 'sglx', 'irig_channel', 'sync', 'python', '{VENV_PYTHON}');
        fprintf('n_entries: %d\\n', length(ct.source));
        fprintf('rate: %.1f\\n', ct.nominal_rate);
        fprintf('source_first: %.1f\\n', ct.source(1));
        fprintf('ref_first: %.6f\\n', ct.reference(1));
    """)
    stdout = matlab_run(code, timeout=180)
    assert f"n_entries: {len(py_ct.source)}" in stdout
    assert f"rate: {py_ct.nominal_rate:.1f}" in stdout


def test_decode_dat(generate_test_dat, tmp_path):
    """decode_irig decodes a .dat file with explicit n_channels/irig_channel."""
    from neurokairos.decoders.irig import decode_dat_irig

    dat_path = generate_test_dat["path"]
    n_ch = generate_test_dat["n_channels"]
    irig_ch = generate_test_dat["irig_channel"]

    # Get Python reference
    py_ct = decode_dat_irig(dat_path, n_ch, irig_ch)

    code = textwrap.dedent(f"""\
        ct = decode_irig('{dat_path}', 'format', 'dat', 'n_channels', {n_ch}, 'irig_channel', {irig_ch}, 'python', '{VENV_PYTHON}');
        fprintf('n_entries: %d\\n', length(ct.source));
        fprintf('rate: %.1f\\n', ct.nominal_rate);
    """)
    stdout = matlab_run(code, timeout=180)
    assert f"n_entries: {len(py_ct.source)}" in stdout
    assert f"rate: {py_ct.nominal_rate:.1f}" in stdout


# -- Helpers -------------------------------------------------------------------


def mat_array_str(values):
    """Format a list of floats as a MATLAB row vector string."""
    return "[" + " ".join(f"{v}" for v in values) + "]"


def parse_mat_array(stdout, label):
    """Parse a MATLAB mat2str output line into a numpy array."""
    for line in stdout.splitlines():
        if line.strip().startswith(f"{label}:"):
            arr_str = line.split(":", 1)[1].strip()
            # Handle both "[1 2 3]" and "1" (scalar)
            arr_str = arr_str.strip("[]")
            # mat2str uses semicolons for row separators
            arr_str = arr_str.replace(";", " ")
            return np.array([float(x) for x in arr_str.split()])
    raise ValueError(f"Label '{label}' not found in MATLAB output")


def extract_value(stdout, label):
    """Extract a single value after 'label:' from MATLAB stdout."""
    for line in stdout.splitlines():
        if line.strip().startswith(f"{label}:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Label '{label}' not found in MATLAB output")
