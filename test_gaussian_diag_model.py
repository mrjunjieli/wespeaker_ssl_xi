"""CPU checks for per-feature precision, variants, and experiment isolation."""

import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_GaussianLogPrec,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_LightAdapter,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Adapter,
    SSL_BACKEND_LN_GaussianLogPrec_Diag_WeightedSum,
)
from wespeaker.models.projections import (
    ArcMarginProduct_intertopk_subcenter_uncertainty,
    ArcMarginProduct_intertopk_subcenter_gap_margin,
)


class GaussianDiagTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(716)
        self.args = dict(feat_dim=16, compression_dim=8, embed_dim=6,
                         head_nb=3, nb_layer=3)

    def model(self, **overrides):
        return SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag(
            **dict(self.args, **overrides))

    def assert_gradients(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_manual_posterior_and_feature_specific_variance(self):
        model = self.model(head_nb=1, compression_dim=2)
        with torch.no_grad():
            model.prior_mean.copy_(torch.tensor([[[1., 3.]]]))
            model.prior_logprec.copy_(torch.tensor([[[2., 4.]]]).log())
        values = torch.tensor([[[2., 5.], [6., 7.]]])
        precisions = torch.tensor([[[[1., 2.]], [[3., 6.]]]])
        mean, variance = model._posterior_pooling(values, precisions.log())
        expected_mean = torch.tensor([[[(2.+18.+2.)/6., (10.+42.+12.)/12.]]])
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(variance, torch.tensor([[[1./6., 1./12.]]]))
        self.assertNotEqual(variance[0, 0, 0], variance[0, 0, 1])

    def test_scalar_equivalence_when_precision_is_tied(self):
        scalar = SSL_BACKEND_MHFA_LN_GaussianLogPrec(**self.args)
        diag = self.model()
        with torch.no_grad():
            scalar.prior_mean.normal_()
            scalar.prior_logprec.normal_()
            diag.prior_mean.copy_(scalar.prior_mean)
            diag.prior_logprec.copy_(scalar.prior_logprec.unsqueeze(-1))
        value = torch.randn(2, 7, 8)
        scalar_logits = torch.randn(2, 7, 3)
        diag_logits = scalar_logits.unsqueeze(-1).expand(-1, -1, -1, 8)
        for expected, actual in zip(scalar._posterior_pooling(value, scalar_logits),
                                    diag._posterior_pooling(value, diag_logits)):
            torch.testing.assert_close(actual, expected)

    def test_chunking_and_recomputation_output_and_gradients(self):
        chunked = self.model(pooling_head_chunk_size=1, pooling_checkpoint=True)
        full = self.model(pooling_head_chunk_size=3, pooling_checkpoint=False)
        full.load_state_dict(chunked.state_dict(), strict=True)
        x1 = torch.randn(3, 16, 11, 3, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        result1, result2 = chunked(x1), full(x2)
        for expected, actual in zip(result1, result2):
            torch.testing.assert_close(actual, expected)
        sum(t.square().mean() for t in result1[1:]).backward()
        sum(t.square().mean() for t in result2[1:]).backward()
        self.assert_gradients(chunked)
        torch.testing.assert_close(x1.grad, x2.grad)
        for name, parameter in chunked.named_parameters():
            torch.testing.assert_close(parameter.grad,
                                       dict(full.named_parameters())[name].grad,
                                       atol=1e-6, rtol=1e-4)

    def test_all_variants_factory_forward_backward_and_round_trip(self):
        for cls in (
            SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag,
            SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer,
            SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_LightAdapter,
            SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Adapter,
            SSL_BACKEND_LN_GaussianLogPrec_Diag_WeightedSum,
        ):
            with self.subTest(model=cls.__name__):
                self.assertIs(get_speaker_model(cls.__name__), cls)
                args = dict(self.args, pooling_head_chunk_size=2)
                if 'Transformer' in cls.__name__:
                    args.update(transformer_nhead=2, transformer_dim_feedforward=16,
                                transformer_dropout=0., transformer_use_position_encoding=False)
                if 'Adapter' in cls.__name__:
                    args['adapter_dim'] = 7
                if 'WeightedSum' in cls.__name__:
                    args.pop('head_nb')
                model = cls(**args)
                value, variance, embedding = model(torch.randn(3, 16, 11, 3))
                self.assertEqual(value.shape, (3, 11, 8))
                self.assertEqual(variance.shape, (3, 6))
                self.assertEqual(embedding.shape, (3, 6))
                self.assertTrue((variance > 0).all())
                (embedding.square().mean() + variance.mean()).backward()
                self.assert_gradients(model)
                restored = cls(**args).eval()
                restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                model.eval()
                with torch.no_grad():
                    for frames in (1, 17, 101):
                        x = torch.randn(1, 16, frames, 3)
                        for expected, actual in zip(model(x), restored(x)):
                            torch.testing.assert_close(actual, expected)

    def test_autocast_extreme_precision(self):
        model = self.model()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, variance, embedding = model(torch.randn(3, 16, 13, 3))
            loss = embedding.float().square().mean() + variance.mean()
        self.assertEqual(variance.dtype, torch.float32)
        loss.backward()
        self.assert_gradients(model)
        for extreme in (-100., 100.):
            with torch.no_grad():
                model.prior_logprec.fill_(extreme)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                mean, variance = model._posterior_pooling(
                    torch.ones(2, 13, 8, dtype=torch.bfloat16),
                    torch.full((2, 13, 24), extreme, dtype=torch.bfloat16))
            self.assertEqual(variance.dtype, torch.float32)
            self.assertTrue(torch.isfinite(mean).all())
            self.assertTrue(torch.isfinite(variance).all())
            self.assertTrue((variance > 0).all())
            torch.testing.assert_close(variance, torch.full_like(
                variance, torch.exp(torch.tensor(-max(-15., min(15., extreme)))) / 14))

    def test_parameter_delta_and_scalar_checkpoint_rejection(self):
        args = dict(self.args, compression_dim=128, head_nb=64)
        scalar = SSL_BACKEND_MHFA_LN_GaussianLogPrec(**args)
        diag = SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag(**args)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(diag) - count(scalar), 1056640)
        self.assertEqual(diag.att_head.weight.shape, (8192, 128))
        self.assertEqual(diag.prior_logprec.shape, (1, 64, 128))
        with self.assertRaises(RuntimeError):
            diag.load_state_dict(scalar.state_dict(), strict=False)

    def test_projection_loss_and_frontend_gradient(self):
        for cls in (ArcMarginProduct_intertopk_subcenter_uncertainty,
                    ArcMarginProduct_intertopk_subcenter_gap_margin):
            for multiplier in (0., 1.):
                with self.subTest(projection=cls.__name__, multiplier=multiplier):
                    model = self.model(feature_grad_mult=multiplier)
                    projection = cls(6, 7, k_top=2)
                    x = torch.randn(3, 16, 7, 3, requires_grad=True)
                    _, variance, embedding = model(x)
                    labels = torch.tensor([0, 1, 2])
                    if getattr(projection, 'requires_uncertainty', False):
                        logits = projection(embedding, variance, labels)
                    else:
                        logits = projection(embedding, labels)
                    F.cross_entropy(logits, labels).backward()
                    self.assert_gradients(model)
                    self.assert_gradients(projection)
                    if multiplier == 0.:
                        self.assertEqual(torch.count_nonzero(x.grad), 0)
                    else:
                        self.assertGreater(torch.count_nonzero(x.grad), 0)

    def test_position_switch_and_invalid_chunk(self):
        for chunk in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                self.model(pooling_head_chunk_size=chunk)
        args = dict(self.args, transformer_nhead=2, transformer_dim_feedforward=16,
                    transformer_dropout=0.)
        with_pos = SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer(**args).eval()
        without_pos = SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer(
            **args, transformer_use_position_encoding=False).eval()
        without_pos.load_state_dict(with_pos.state_dict())
        x = torch.randn(2, 16, 17, 3)
        with torch.no_grad():
            for expected, actual in zip(without_pos(x)[1:], without_pos(x.flip(2))[1:]):
                torch.testing.assert_close(expected, actual)
            self.assertFalse(torch.allclose(with_pos(x)[-1], without_pos(x)[-1]))

    def test_recipe_config_variants(self):
        root = Path(__file__).resolve().parent / 'examples/voxceleb/v2_w2vbert_sv'
        spec = importlib.util.spec_from_file_location(
            'prepare_diag', root / 'local/prepare_gaussian_diag_config.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for variant in module.VARIANTS:
            for stage in module.STAGES:
                with self.subTest(variant=variant, stage=stage):
                    config = module.make_config(variant, stage, f'/tmp/diag_{variant}')
                    self.assertIn('_Diag', config['model'])
                    self.assertEqual(config['num_epochs'], 30 if stage == 'stage1' else 3)
                    self.assertEqual(config['gpus'], '[0,1]')
                    args = dict(config['model_args'], feat_dim=16)
                    args.update(nb_layer=3, compression_dim=8, embed_dim=6)
                    if 'head_nb' in args:
                        args['head_nb'] = 2
                    model = get_speaker_model(config['model'])(**args).eval()
                    if variant.startswith('transformer_'):
                        self.assertEqual(model.transformer_use_position_encoding,
                                         variant == 'transformer_pos')
                    with torch.no_grad():
                        self.assertEqual(model(torch.randn(1, 16, 7, 3))[-1].shape, (1, 6))

    def test_run_entry_points_prepare_without_cuda(self):
        root = Path(__file__).resolve().parent / 'examples/voxceleb/v2_w2vbert_sv'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', PYTHON_BIN=sys.executable)
        scripts = ('noadapter', 'transformer_pos', 'transformer_nopos',
                   'lightadapter', 'adapter', 'weightedsum')
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            for variant in scripts:
                result = subprocess.run(
                    ['bash', f'run_gaussian_diag_{variant}.sh', '--prepare_only', 'true',
                     '--exp', str(Path(temporary) / variant)], cwd=root, env=environment,
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('no training or GPU initialization', result.stdout)
            result = subprocess.run(
                ['bash', 'run_compare_gaussian_diag_position_stage1.sh',
                 '--prepare_only', 'true', '--exp', str(Path(temporary) / 'comparison'),
                 '--with_position', str(Path(temporary) / 'pos'),
                 '--without_position', str(Path(temporary) / 'nopos')],
                cwd=root, env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for bad_options in (['--gpus', '[0,1]'], ['--stage', '0']):
                result = subprocess.run(
                    ['bash', 'run_gaussian_diag.sh', '--prepare_only', 'true'] + bad_options,
                    cwd=root, env=environment, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_metric_summary_and_missing_metrics_rejection(self):
        root = Path(__file__).resolve().parent / 'examples/voxceleb/v2_w2vbert_sv'
        spec = importlib.util.spec_from_file_location(
            'summarize_diag', root / 'local/summarize_gaussian_diag.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            temporary = Path(temporary)
            for variant in ('pos', 'nopos'):
                for directory in ('scores', 'scores_uncertainty'):
                    path = temporary / variant / 'eval' / directory / 'vox1_cos_result'
                    path.parent.mkdir(parents=True)
                    path.write_text(''.join(
                        f'---- vox1_{subset}_cleaned.kaldi.score -----\n'
                        'EER = 1.234\nminDCF (p_target:0.01 c_miss:1 c_fa:1) = 0.123\n'
                        for subset in ('O', 'E', 'H')))
            rows = module.summarize(temporary / 'pos', temporary / 'nopos',
                                    temporary / 'result.tsv')
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0], ('with_position', 'cosine', 'O', 1.234, 0.123))
            path.write_text('')
            with self.assertRaises(ValueError):
                module.summarize(temporary / 'pos', temporary / 'nopos',
                                 temporary / 'missing.tsv')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
