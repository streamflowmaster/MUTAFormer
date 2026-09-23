"""Data-free release checks; no model training or external network access."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'gene_with_order' / 'web'
sys.path[:0] = [str(WEB), str(ROOT), str(ROOT / 'gene_with_order')]


class ReleaseChecks(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    def test_import_legacy_engine(self):
        import legacy_engine
        self.assertTrue(callable(legacy_engine.run))

    def test_json_api_errors(self):
        from server import app
        response = app.test_client().get('/api/missing-release-check')
        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.is_json)

    def test_synthetic_input(self):
        import pipeline
        frame = pipeline.validate_database(pipeline.read_table(WEB / 'data/demo_cohort.csv'))
        self.assertEqual(frame.Class.nunique(), 2)

    def test_source_release_has_no_weights_or_workbooks(self):
        for path in ROOT.rglob('*'):
            self.assertNotIn(path.suffix.lower(), {'.xlsx', '.xls', '.pth', '.joblib', '.zip'})


if __name__ == '__main__':
    unittest.main()
