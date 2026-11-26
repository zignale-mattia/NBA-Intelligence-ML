"""Fast checks: parse committed code without running data collectors or ML."""
import ast
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/NBA_Data_Science_Analytics_V4_sanitized.ipynb"


class RepositoryTests(unittest.TestCase):
    def test_python_files_parse(self):
        for path in [*ROOT.joinpath("src").glob("*.py"),
                     *ROOT.joinpath("dashboard").glob("*.py")]:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebook_is_clean_and_syntactically_valid(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(notebook["cells"]), 13)
        seen = set()
        for index, cell in enumerate(notebook["cells"], 1):
            with self.subTest(cell=index):
                self.assertNotIn(cell["id"], seen)
                seen.add(cell["id"])
                if cell["cell_type"] != "code":
                    continue
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                source = "".join(cell["source"])
                # IPython shell magics are only valid inside notebook kernels.
                clean = "\n".join(
                    line for line in source.splitlines()
                    if not line.lstrip().startswith(("!", "%"))
                )
                ast.parse(clean, filename=f"notebook-cell-{index}")

    def test_data_directory_can_be_overridden_in_notebook(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) for c in notebook["cells"]]
        self.assertIn('NBA_DATA_DIR', sources[0])
        self.assertIn('BASE_DIR', sources[1])
        self.assertIn('BASE_DIR', sources[2])
        self.assertIn('BASE_DIR', sources[11])
        self.assertIn('BASE_DIR', sources[12])

    def test_no_data_or_models_tracked(self):
        files = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], text=True
        ).splitlines()
        restricted = (".db", ".sqlite", ".sqlite3", ".joblib",
                      ".pkl", ".csv", ".xlsx", ".parquet", ".env")
        for filename in files:
            with self.subTest(filename=filename):
                self.assertFalse(filename.endswith(restricted)
                                 and filename != ".env.example")


if __name__ == "__main__":
    unittest.main()
