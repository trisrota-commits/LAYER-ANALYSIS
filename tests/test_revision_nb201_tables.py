"""Complete NB201 primary/table projections, including joint Holm-six."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NB201TableTests(unittest.TestCase):
    def test_three_dataset_primary_projection_has_joint_holm_and_supporting_rows(self):
        self.assertTrue((ROOT / 'analysis/revision_nb201_tables.py').is_file())
        from analysis.revision_nb201_tables import project_primary
        specs = {'cifar10':'nb201-qwen-primary-v1/cifar10',
                 'cifar100':'nb201-qwen-primary-v1/cifar100',
                 'imagenet16':'nb201-imagenet16-primary-v1'}
        reports = {ds:json.loads((ROOT/'evidence'/p/'analysis/results.json').read_text()) for ds,p in specs.items()}
        main, descriptive, supporting = project_primary(reports, {'cifar10':45,'cifar100':44,'imagenet16':43})
        self.assertEqual(len(main),3)
        self.assertEqual(len(descriptive),6)
        self.assertEqual(len(supporting),18)
        self.assertAlmostEqual(main[0]['record_holm6_p'],4.705583981533387e-6)
        self.assertAlmostEqual(main[2]['seed_holm6_p'],.00038084169488598505)
        self.assertEqual({r['dataset'] for r in supporting},set(specs))


if __name__ == '__main__':
    unittest.main()
