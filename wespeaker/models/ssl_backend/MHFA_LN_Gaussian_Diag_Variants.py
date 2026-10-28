"""Diagonal Gaussian variants retaining the original feature pathways."""

from .MHFA_LN_Gaussian_Diag import DiagonalGaussianMixin
from .MHFA_LN_Gaussian_Transformer import (
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer)
from .MHFA_LN_Gaussian_LightAdapter import (
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_LightAdapter)
from .MHFA_LN_Gaussian_Adapter import (
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter)
from .MHFA_LN_Gaussian_WeightedSum import (
    SSL_BACKEND_LN_GaussianLogPrec_WeightedSum)

__all__ = [
    'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer',
    'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_LightAdapter',
    'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Adapter',
    'SSL_BACKEND_LN_GaussianLogPrec_Diag_WeightedSum',
]


class SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer(
        DiagonalGaussianMixin, SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer):
    """Temporal Transformer followed by diagonal Gaussian MHFA."""


class SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_LightAdapter(
        DiagonalGaussianMixin, SSL_BACKEND_MHFA_LN_GaussianLogPrec_LightAdapter):
    """Per-layer light adapters followed by diagonal Gaussian MHFA."""

    def _pool_inputs(self, x):
        return self._fuse_features(self._apply_adapters(x))


class SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Adapter(
        DiagonalGaussianMixin, SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter):
    """Per-layer adapters followed by diagonal Gaussian MHFA."""

    def _pool_inputs(self, x):
        return self._fuse_features(self._apply_adapters(x))


class SSL_BACKEND_LN_GaussianLogPrec_Diag_WeightedSum(
        DiagonalGaussianMixin, SSL_BACKEND_LN_GaussianLogPrec_WeightedSum):
    """Single-path, single-head weighted sum with per-feature precision."""

    def _pool_inputs(self, x):
        value = self._fuse_features(x)
        return value, value
