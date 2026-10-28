"""CPU regression checks for the temporal Transformer MHFA experiment."""

import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from wespeaker.models.projections import (
    ArcMarginProduct_intertopk_subcenter_gap_margin,
    ArcMarginProduct_intertopk_subcenter_uncertainty,
)
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_GaussianLogPrec,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer,
)


class TransformerMHFATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(716)
        self.args = dict(
            feat_dim=16, compression_dim=8, embed_dim=6, head_nb=2,
            nb_layer=3, transformer_nhead=2,
            transformer_dim_feedforward=16, transformer_dropout=0.0)

    def make_model(self, **overrides):
        return SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer(
            **dict(self.args, **overrides))

    def assert_gradients(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_factory_and_variable_lengths(self):
        self.assertIs(
            get_speaker_model('SSL_BACKEND_MHFA_LN_GaussianLogPrec'),
            SSL_BACKEND_MHFA_LN_GaussianLogPrec)
        self.assertIs(
            get_speaker_model('SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer'),
            SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer)
        model = self.make_model().eval()
        with torch.no_grad():
            for frames in (1, 17, 501):
                value, variance, embedding = model(
                    torch.randn(1, 16, frames, 3))
                self.assertEqual(value.shape, (1, frames, 8))
                self.assertEqual(variance.shape, (1, 6))
                self.assertEqual(embedding.shape, (1, 6))
                for output in (value, variance, embedding):
                    self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((variance >= 0).all())

    def test_order_sensitivity(self):
        model = self.make_model().eval()
        baseline = SSL_BACKEND_MHFA_LN_GaussianLogPrec(
            **{k: self.args[k] for k in (
                'feat_dim', 'compression_dim', 'embed_dim', 'head_nb',
                'nb_layer')}).eval()
        x = torch.randn(2, 16, 17, 3)
        with torch.no_grad():
            torch.testing.assert_close(baseline(x)[-1], baseline(x.flip(2))[-1])
            self.assertFalse(torch.allclose(model(x)[-1], model(x.flip(2))[-1]))
            k, v = model._fuse_features(x)
            separate_k, separate_v = model._fuse_features(x[:1])
            torch.testing.assert_close(k[:1], separate_k)
            torch.testing.assert_close(v[:1], separate_v)

    def test_position_encoding_switch(self):
        with_position = self.make_model().eval()
        without_position = self.make_model(
            transformer_use_position_encoding=False).eval()
        without_position.load_state_dict(with_position.state_dict(), strict=True)
        self.assertTrue(with_position.transformer_use_position_encoding)
        x = torch.randn(2, 16, 17, 3)
        with torch.no_grad():
            forward = without_position(x)
            reverse = without_position(x.flip(2))
            torch.testing.assert_close(forward[0], reverse[0].flip(1))
            for expected, actual in zip(forward[1:], reverse[1:]):
                torch.testing.assert_close(expected, actual)
            self.assertFalse(torch.allclose(with_position(x)[-1], forward[-1]))
        without_position.train()
        _, variance, embedding = without_position(x)
        (embedding.square().mean() + variance.mean()).backward()
        self.assert_gradients(without_position)

    def test_loss_paths_and_frontend_gradient(self):
        for projection_cls in (
                ArcMarginProduct_intertopk_subcenter_uncertainty,
                ArcMarginProduct_intertopk_subcenter_gap_margin):
            for grad_mult in (0.0, 1.0):
                with self.subTest(projection=projection_cls, grad_mult=grad_mult):
                    model = self.make_model(feature_grad_mult=grad_mult)
                    projection = projection_cls(6, 7, k_top=2)
                    x = torch.randn(3, 16, 11, 3, requires_grad=True)
                    labels = torch.tensor([0, 1, 2])
                    for _ in range(2):
                        model.zero_grad(set_to_none=True)
                        projection.zero_grad(set_to_none=True)
                        x.grad = None
                        _, variance, embedding = model(x)
                        if getattr(projection, 'requires_uncertainty', False):
                            logits = projection(embedding, variance, labels)
                        else:
                            logits = projection(embedding, labels)
                        loss = F.cross_entropy(logits, labels)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assert_gradients(model)
                        self.assert_gradients(projection)
                        self.assertTrue(torch.isfinite(x.grad).all())
                        if grad_mult == 0.0:
                            self.assertEqual(torch.count_nonzero(x.grad), 0)
                        else:
                            self.assertGreater(torch.count_nonzero(x.grad), 0)

    def test_autocast_and_extreme_precision(self):
        model = self.make_model()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, variance, embedding = model(torch.randn(3, 16, 13, 3))
            loss = embedding.float().square().mean() + variance.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assert_gradients(model)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            for logprec in (-100.0, 100.0):
                mean, var = model._posterior_pooling(
                    torch.ones(2, 13, 8, dtype=torch.bfloat16),
                    torch.full((2, 13, 2), logprec, dtype=torch.bfloat16))
                self.assertEqual(var.dtype, torch.float32)
                self.assertTrue(torch.isfinite(mean).all())
                self.assertTrue(torch.isfinite(var).all())
                self.assertTrue((var > 0).all())

    def test_checkpoint_round_trip(self):
        model = self.make_model().eval()
        restored = self.make_model().eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        x = torch.randn(2, 16, 11, 3)
        with torch.no_grad():
            for expected, actual in zip(model(x), restored(x)):
                torch.testing.assert_close(expected, actual)

    def test_invalid_settings_and_odd_dimension(self):
        for options in (
                dict(transformer_num_layers=0), dict(transformer_nhead=3),
                dict(transformer_dim_feedforward=0),
                dict(transformer_dropout=1.0)):
            with self.assertRaises(ValueError):
                self.make_model(**options)
        model = self.make_model(compression_dim=9, transformer_nhead=3).eval()
        with torch.no_grad():
            self.assertEqual(model(torch.randn(1, 16, 7, 3))[-1].shape, (1, 6))

    def test_position_ablation_recipe_parity(self):
        conf = Path(__file__).resolve().parent / (
            'examples/voxceleb/v2_w2vbert_sv/conf')
        with_position = yaml.safe_load((conf / (
            'mhfa_gaussian_logprec_noadapter_transformer_stage1.yaml'
        )).read_text())
        without_position = yaml.safe_load((conf / (
            'mhfa_gaussian_logprec_noadapter_transformer_nopos_stage1.yaml'
        )).read_text())
        self.assertIs(without_position['model_args'].pop(
            'transformer_use_position_encoding'), False)
        with_position.pop('exp_dir')
        without_position.pop('exp_dir')
        self.assertEqual(with_position, without_position)

    def test_recipe_parity(self):
        conf = Path(__file__).resolve().parent / (
            'examples/voxceleb/v2_w2vbert_sv/conf')
        for stage, num_avg in (
                ('stage1', 4), ('stage2_joint', 1), ('stage3_lmft', 1)):
            baseline = yaml.safe_load((conf / (
                f'mhfa_gaussian_logprec_noadapter_{stage}.yaml')).read_text())
            new = yaml.safe_load((conf / (
                f'mhfa_gaussian_logprec_noadapter_transformer_{stage}.yaml'
            )).read_text())
            self.assertEqual(new['num_avg'], num_avg)
            for field in ('exp_dir', 'gpus', 'model', 'num_avg'):
                baseline.pop(field)
                new.pop(field)
            for field in list(new['model_args']):
                if field.startswith('transformer_'):
                    new['model_args'].pop(field)
            self.assertEqual(new, baseline)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
