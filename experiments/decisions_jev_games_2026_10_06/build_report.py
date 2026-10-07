"""Build the audited report, executed notebook, and self-contained source bundle.

Offline only. Publishing is an explicit copy of ``report/`` to the report host;
this script never deploys a page or invokes a model.
"""
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    subprocess.run([sys.executable, str(HERE / "analyze.py")], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(HERE / "build_notebook.py")], cwd=ROOT, check=True)
    notebook_path = ROOT / "notebooks/decisions_jev_games.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, notebook_path)
    report = HERE / "report"
    shutil.copy2(notebook_path, report / notebook_path.name)
    shutil.copy2(HERE / "data/summary.json", report / "summary.json")
    shutil.copy2(ROOT / "docs/sysone.md", report / "sysone.md")

    # Explicit inclusion: exclude unrelated working-tree edits, data, credentials,
    # environments, bytecode, and the archive itself.
    files = [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "docs/sysone.md", notebook_path]
    files += sorted((ROOT / "lludens").glob("*.py"))
    files += sorted((ROOT / "lludens/sysone").glob("*.py"))
    files += [ROOT / "tests/test_decision_games.py", ROOT / "tests/test_sysone.py"]
    files += sorted(HERE.glob("*.py"))
    files += [HERE / "README.md", HERE / "requirements.txt", HERE / "report_template.html"]
    for pattern in ("*.json", "*.jsonl.gz", "matches/*.jsonl"):
        files += sorted((HERE / "data").glob(pattern))
    for pattern in ("*.html", "*.png", "*.json"):
        files += sorted(report.glob(pattern))
    with ZipFile(report / "replication.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in replication bundle: {path}")
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("README.md", (HERE / "README.md").read_text())
        # Keep the report's relative downloads usable after extraction.
        for name in (notebook_path.name, "sysone.md"):
            archive.write(report / name, str((report / name).relative_to(ROOT)))
    print("Report, executed notebook, opening examples, and replication bundle built offline.")


if __name__ == "__main__":
    main()
