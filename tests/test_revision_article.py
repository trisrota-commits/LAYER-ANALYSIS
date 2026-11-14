"""Focused article adapter contract and canonical-source checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis import revision_article as article

REPO = Path(__file__).resolve().parents[1]

class ArticleTests(unittest.TestCase):
    def test_frozen_input_tamper_is_rejected_before_output(self):
        from unittest.mock import patch
        self.assertTrue(hasattr(article, 'verify_inputs'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            with patch.object(article, 'sha', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'Frozen article input changed'):
                    article.run(REPO, out)
            self.assertFalse(out.exists())

    def test_existing_output_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); marker=out/'sentinel'; marker.write_text('keep')
            with self.assertRaises(ValueError): article.run(Path(tmp)/'missing',out)
            self.assertEqual(marker.read_text(),'keep')
    def test_preserved_sources(self):
        manifest=article.verify_sources(REPO)
        self.assertEqual(len(manifest['files']),16)
        self.assertEqual(manifest['files']['primary_endpoint.py']['sha256'],'ec93db019513a06e9fa5a44f3c655f3ec5f7c9b8581895f876ca142d11a943c0')
    def test_canonical_identity_and_missing_seeds(self):
        cond, sources, keys=article.load_conditions(REPO)
        self.assertEqual(len(keys),115)
        self.assertEqual(len(set(keys)),115)
        for k in [(3,'on','no-layer'),(2,'on','all-seven'),(3,'off','taylor-sensitivity')]:
            self.assertEqual(set(cond[k]),set(range(42,48)))
            self.assertTrue(all(cond[k][s][0]==keys for s in cond[k]))
        self.assertTrue(any('third-on-no_layer-s42' in r['member'] for r in sources))
    def test_bad_identity_rejected(self):
        from analysis.thesis_lemur import matrix
        a=b'idx,dataset,gt_best_accuracy,pred_best_accuracy\n1,x,2,3\n'
        b=b'idx,dataset,gt_best_accuracy,pred_best_accuracy\n2,x,2,3\n'
        with self.assertRaises(ValueError): matrix([a,b],1)
    def test_invalid_fit_withholds_inference(self):
        self.assertFalse(article.fit_is_valid(True,[1.,-1.],[],[1.,2.]))
        self.assertFalse(article.fit_is_valid(True,[1.,2.],['Hessian is not positive definite'],[1.,2.]))
        self.assertTrue(article.fit_is_valid(True,[1.,2.],[],[1.,2.]))
    def test_source_tamper_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copytree(REPO/'paper_runners/article_analysis/original',root/'paper_runners/article_analysis/original')
            shutil.copy(REPO/'paper_runners/article_analysis/sources.json',root/'paper_runners/article_analysis/sources.json')
            (root/'paper_runners/article_analysis/original/selection.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                article.verify_sources(root)

    @unittest.skipUnless(__import__('os').environ.get('ARTICLE_FULL_TEST')=='1','set ARTICLE_FULL_TEST=1 for canonical full calculation')
    def test_full_calculation_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh'
            result=article.run(REPO,out)
            primary=result['primary_endpoint']
            self.assertEqual(primary['contract']['bootstrap_draws'],20000)
            self.assertEqual(primary['contract']['pooled_cells'],636)
            self.assertAlmostEqual(primary['primary_by_record']['gain'],.4956466540880506,places=12)
            robust=primary['robustness_non_collapsed']
            self.assertEqual(robust['two_way_cluster_bootstrap']['p'],.0127)
            self.assertAlmostEqual(robust['sign_flip_permutation']['p'],.000199990000499975,places=15)
            self.assertAlmostEqual(robust['location_median']['taylor'],4.51)
            self.assertAlmostEqual(robust['location_trimmed_20pct']['taylor'],4.715549738219895)
            mixed=robust['crossed_mixed_model']
            if mixed['status']!='valid': self.assertIsNone(mixed['p'])
            rows=result['multiplicity']
            self.assertEqual(len(rows),90)
            from collections import Counter
            self.assertEqual(sorted(Counter(r['family'] for r in rows).values()),[5,10,30,45])
            self.assertEqual(rows[0]['p'],.0128)
            self.assertEqual(rows[0]['p_holm'],.064)
            self.assertAlmostEqual(rows[0]['q_bh'],.0855)
            self.assertNotEqual(rows[0]['delta'],rows[0]['bootstrap_mean_delta'])
            self.assertEqual(len(result['ensemble']),7)
            self.assertEqual(len(result['within_dataset']),20)
            self.assertEqual(len(result['collapse_detection_stratified_bootstrap']),2)
            self.assertAlmostEqual(result['collapse_detection_stratified_bootstrap'][0]['auroc'],.9905660377358491)
            provenance=json.loads((out/'provenance.json').read_text())
            self.assertTrue(provenance['prediction_archive']['unchanged_after_run'])
            self.assertEqual(len(provenance['prediction_sources']),276)
            self.assertEqual([r['n'] for r in result['stratification'][::2]],[115,9,106,88,80])
            self.assertEqual(len(result['selection']['rows']),126)
            receipt=json.loads((out/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'complete')
            for name,digest in receipt['files_sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)
            with self.assertRaises(ValueError): article.run(REPO,out)

    def test_stable_selection(self):
        ns=article.functions(REPO,'selection.py',{'np':np,'stats':__import__('scipy').stats})
        result=ns['sel_metrics'](np.ones(12),np.arange(12.),np.ones(12,dtype=bool))
        self.assertEqual(result['regret@1'],11.)

if __name__=='__main__': unittest.main()
