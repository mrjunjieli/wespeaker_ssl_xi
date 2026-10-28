"""Materialize separate diagonal-Gaussian configs from the reference recipes."""

import argparse
from pathlib import Path

import yaml

RECIPE = Path(__file__).resolve().parents[1]
VARIANTS = {
    'noadapter': ('mhfa_gaussian_logprec_noadapter',
                  'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag'),
    'transformer_pos': ('mhfa_gaussian_logprec_noadapter_transformer',
                        'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer'),
    'transformer_nopos': ('mhfa_gaussian_logprec_noadapter_transformer',
                          'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Transformer'),
    'lightadapter': ('mhfa_gaussian_logprec',
                     'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_LightAdapter'),
    'adapter': ('mhfa_gaussian_logprec',
                'SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag_Adapter'),
    'weightedsum': ('weightedsum_gaussian_logprec',
                    'SSL_BACKEND_LN_GaussianLogPrec_Diag_WeightedSum'),
}
STAGES = ('stage1', 'stage2_joint', 'stage3_lmft')


def make_config(variant, stage, exp_prefix, chunk_size=8, pooling_checkpoint=True):
    source, model = VARIANTS[variant]
    config = yaml.safe_load((RECIPE / 'conf' / f'{source}_{stage}.yaml').read_text())
    config['model'] = model
    config['model_init'] = None
    config['exp_dir'] = f'{exp_prefix}_{stage}'
    config['gpus'] = '[0,1]'
    config['num_avg'] = 4 if stage == 'stage1' else 1
    args = config['model_args']
    args['pooling_head_chunk_size'] = chunk_size
    args['pooling_checkpoint'] = pooling_checkpoint
    if variant.startswith('transformer_'):
        args['transformer_use_position_encoding'] = variant == 'transformer_pos'
    if variant in ('adapter', 'lightadapter'):
        args['adapter_dim'] = 256 if variant == 'adapter' else 128
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--exp-prefix', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pooling-head-chunk-size', type=int, default=8)
    options = parser.parse_args()
    if options.pooling_head_chunk_size < 1:
        parser.error('pooling-head-chunk-size must be positive')
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        config = make_config(options.variant, stage, options.exp_prefix,
                             options.pooling_head_chunk_size)
        path = options.output_dir / f'{stage}.yaml'
        text = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise SystemExit(f'Refusing to overwrite a different config: {path}')
        path.write_text(text)
        print(f'{path}: {config["model"]}, epochs={config["num_epochs"]}')


if __name__ == '__main__':
    main()
