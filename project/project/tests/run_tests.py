"""Minimal runner for environments without pytest:  python tests/run_tests.py"""
import importlib, inspect, sys, tempfile, traceback, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
failed = 0
for mod_name in ("test_features", "test_pairs", "test_collect", "test_users", "test_models"):
    mod = importlib.import_module(mod_name)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d)) if "tmp_path" in inspect.signature(fn).parameters else fn()
            print(f"PASS {mod_name}.{name}")
        except unittest.SkipTest as e:
            print(f"SKIP {mod_name}.{name} ({e})")
        except Exception:
            failed += 1
            print(f"FAIL {mod_name}.{name}"); traceback.print_exc()
sys.exit(1 if failed else 0)
