"""CPU-only checks for the EGR-FS naming migration and legacy result reading."""
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.protocol import prepare, read_csv, write_csv, VARIANTS
from experiments.paper import config_for, aggregate


class ProjectNamingTests(unittest.TestCase):
    def test_new_protocol_contains_only_new_method_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = prepare(root)
            self.assertEqual(len(rows), 50)
            self.assertIn('egr_fs', VARIANTS)
            self.assertNotIn('v4', VARIANTS)
            main = [r for r in rows if r['experiment'] == 'egr_fs']
            self.assertEqual(len(main), 3)
            for row in main:
                cfg = yaml.safe_load(Path(row['config']).read_text())
                self.assertEqual(cfg['SESSIONAME'], 'egr_fs_s' + str(row['seed']))
            cfg, path = config_for(root, 'v4', 200)
            self.assertEqual(path.name, 'egr_fs_s200.yaml')

    def test_legacy_config_and_results_keep_raw_files_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'configs').mkdir()
            path = root/'configs/v4_s200.yaml'
            path.write_text('SESSIONAME: v4_s200\nMODEL: robofire_graph\n')
            original = path.read_bytes()
            cfg, selected = config_for(root, 'egr_fs', 200)
            self.assertEqual(selected, path)
            self.assertEqual(cfg['SESSIONAME'], 'v4_s200')
            samples = []
            for name, score in [('rffnet', .8), ('v4', .85)]:
                dest = root/'results'/name/'test_clean/summary.csv'
                write_csv(dest, [dict(experiment=name, seed='200', dataset='FLAME2',
                    split='test', condition='clean', full_split='True',
                    output='prediction' if name == 'rffnet' else 'refined', miou=score)])
                samples.append((dest, dest.read_bytes()))
            aggregate(root)
            rows = read_csv(root/'paired_seed_deltas.csv')
            self.assertEqual(rows[0]['comparison'], 'egr_fs_minus_rffnet')
            self.assertAlmostEqual(float(rows[0]['mean_delta']), .05)
            self.assertEqual(path.read_bytes(), original)
            for dest, data in samples:
                self.assertEqual(dest.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
