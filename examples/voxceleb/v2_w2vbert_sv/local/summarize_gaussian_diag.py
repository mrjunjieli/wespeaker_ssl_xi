"""Summarize Stage1 position-encoding ablations for diagonal Gaussian models."""

import argparse
import csv
import re
from pathlib import Path

RESULT_PATTERN = re.compile(
    r'----\s+(\S+)\s+-----\s*EER\s*=\s*([0-9.]+)\s*'
    r'minDCF[^\n]*=\s*([0-9.]+)')


def summarize(with_position, without_position, output):
    rows, metrics = [], {}
    for variant, root in (('with_position', Path(with_position)),
                          ('without_position', Path(without_position))):
        for scoring, directory in (('cosine', 'scores'),
                                   ('uncertainty', 'scores_uncertainty')):
            path = root / 'eval' / directory / 'vox1_cos_result'
            results = {name: (float(eer), float(dcf))
                       for name, eer, dcf in RESULT_PATTERN.findall(path.read_text())}
            for subset in ('O', 'E', 'H'):
                trial = f'vox1_{subset}_cleaned.kaldi.score'
                if trial not in results:
                    raise ValueError(f'Missing metrics for {trial}: {path}')
                eer, dcf = results[trial]
                rows.append((variant, scoring, subset, eer, dcf))
                metrics[variant, scoring, subset] = (eer, dcf)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(('variant', 'scoring', 'trial', 'EER_percent', 'minDCF'))
        writer.writerows(rows)
    print(output.read_text())
    print('Delta = without_position - with_position; negative favors no positions.')
    for scoring in ('cosine', 'uncertainty'):
        for subset in ('O', 'E', 'H'):
            baseline = metrics['with_position', scoring, subset]
            candidate = metrics['without_position', scoring, subset]
            print(f'{scoring} Vox1-{subset}: EER delta={candidate[0]-baseline[0]:+.3f} pp, '
                  f'minDCF delta={candidate[1]-baseline[1]:+.3f}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-position', type=Path, required=True)
    parser.add_argument('--without-position', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    summarize(options.with_position, options.without_position, options.output)


if __name__ == '__main__':
    main()
