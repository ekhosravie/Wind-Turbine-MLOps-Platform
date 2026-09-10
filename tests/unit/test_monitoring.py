"""
Regression test for a bug caught while assembling this repo: an earlier,
regex-based extraction of src/monitoring/drift_detector.py accidentally
pulled in orchestration try/except blocks alongside the intended pure
functions, so importing the module printed "... skipped" messages and
attempted to reference notebook-only globals. Re-extracted with `ast`
parsing instead of regex; this test guards against that regressing.
"""

import io
import contextlib


def test_drift_detector_import_has_no_side_effects():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import importlib
        import src.monitoring.drift_detector as dd
        importlib.reload(dd)
    assert buf.getvalue() == "", f"Import printed unexpected output: {buf.getvalue()!r}"


def test_drift_detector_only_exports_pure_functions():
    import src.monitoring.drift_detector as dd
    expected = {
        "calculate_psi", "detect_data_drift", "detect_concept_drift",
        "detect_prediction_drift", "detect_model_drift",
    }
    exported_funcs = {
        name for name in dir(dd)
        if callable(getattr(dd, name)) and not name.startswith("_")
        and getattr(getattr(dd, name), "__module__", None) == "src.monitoring.drift_detector"
    }
    assert exported_funcs == expected
