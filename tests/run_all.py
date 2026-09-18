"""Run every test module and the smoke test.  Exit code is the number of failures."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ["tests/test_plucker.py", "tests/test_injection.py",
           "tests/test_ngi.py", "tests/test_model.py", "tests/test_pipeline.py",
           "scripts/smoke_test.py"]


def main() -> int:
    failures, summary = 0, []
    for m in MODULES:
        print(f"\n{'=' * 72}\n{m}\n{'=' * 72}")
        r = subprocess.run([sys.executable, os.path.join(ROOT, m)], cwd=ROOT,
                           capture_output=True, text=True)
        out = "\n".join(l for l in r.stdout.splitlines()
                        if "UserWarning" not in l and "warnings.warn" not in l)
        print(out)
        if r.returncode != 0:
            failures += 1
            tail = r.stderr.strip().splitlines()[-5:]
            if tail:
                print("stderr:", "\n".join(tail))
        last = [l for l in out.splitlines() if "passed" in l or "SMOKE" in l]
        summary.append((m, r.returncode, last[-1] if last else ""))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for m, rc, note in summary:
        print(f"  [{'ok ' if rc == 0 else 'FAIL'}] {m:32} {note}")
    print(f"\n{len(MODULES) - failures}/{len(MODULES)} modules green")
    return failures


if __name__ == "__main__":
    sys.exit(main())
