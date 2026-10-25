import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_all_notebooks_are_valid_and_clean(self):
        notebooks = sorted((ROOT / 'notebooks').glob('*.ipynb'))
        self.assertEqual(len(notebooks), 6)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            self.assertIn('cells', notebook)
            for cell in notebook['cells']:
                source = ''.join(cell.get('source', []))
                self.assertNotIn('METADATA_PAYLOAD', source)
                self.assertNotIn('metadata_payload =', source)
                self.assertNotIn('MODULE_SOURCE =', source)
                if cell['cell_type'] == 'code':
                    self.assertEqual(cell.get('outputs', []), [])
                    self.assertIsNone(cell.get('execution_count'))

    def test_private_metadata_is_not_published(self):
        self.assertFalse((ROOT / 'artifacts' / 'kion_reward_item_features.csv').exists())
        manifest = json.loads((ROOT / 'artifacts' / 'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['published'])
        self.assertEqual(len(manifest['sha256']), 64)

    def test_feature_store_accepts_explicit_splits(self):
        import ast
        tree = ast.parse((ROOT / 'src' / 'kion_reward_model.py').read_text(encoding='utf-8'))
        feature_store = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'FeatureStore'
        )
        ingest = next(
            node for node in feature_store.body
            if isinstance(node, ast.FunctionDef) and node.name == 'ingest'
        )
        self.assertIn('splits', [argument.arg for argument in ingest.args.args])


if __name__ == '__main__':
    unittest.main()
