# SPDX-License-Identifier: Apache-2.0
"""Compare original baseline and final kernel on identical inputs and fp64 truth."""
import importlib.util
import json
from pathlib import Path
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kind = sys.argv[1]
release_root = Path(__file__).resolve().parent
here = release_root / kind
bench = load(here / 'benchmark.py', f'{kind}_benchmark')
baseline = load(here / 'baseline.py', f'{kind}_baseline')
candidate = load(release_root / 'build/torch-cuda' / f'_{kind}.py', f'{kind}_candidate')
scope = {}
exec(json.loads((here / 'definition.json').read_text())['reference'], scope)
reference = scope['run']
device = torch.device('cuda')
report = {'component': kind, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
          'method': 'Identical inputs for both implementations, original judge unchanged, fp64 rebuilt truth.',
          'global_seed_plus_batch': 20261026, 'relative_tolerance': bench.REL_TOL,
          'absolute_cap_factor': bench.CAP, 'reference_error_floor_factor': bench.FLOOR,
          'individual_state_check_is_additional': True, 'rows': []}
for batch in (16, 32, 64, 128, 256, 512):
    torch.manual_seed(20261026 + batch)
    first = bench.structured_inputs(batch, device, seed=batch)
    ref1 = reference(*[x.clone() for x in first])
    second = (*ref1, *bench.update_buffer(batch, device, torch.Generator(device=device).manual_seed(batch + 1)), first[-1])
    random = bench.random_inputs(batch, device, seed=batch + 2)
    for family, inputs, ref in (('structured', first, ref1), ('second_window', second, None), ('random', random, None)):
        if ref is None:
            ref = reference(*[x.clone() for x in inputs])
        exact = bench.exact_state(inputs)
        ref_error = (bench.decode(*ref) - exact).abs().mean((-2, -1))
        magnitude = exact.abs().mean((-2, -1))
        tight = (ref_error >= bench.FLOOR * magnitude) & (ref_error > 0)
        row = {'batch_size': batch, 'family': family}
        for name, impl in (('baseline', baseline), ('current', candidate)):
            output = impl.run(*[x.clone() for x in inputs])
            torch.cuda.synchronize()
            passed, ratio = bench.judge(output, ref, exact)
            error = (bench.decode(*output) - exact).abs().mean((-2, -1))
            strict = torch.where(tight, error <= (1 + bench.REL_TOL) * ref_error, error <= bench.CAP * magnitude)
            row[name] = {
                'original_judge_passed': passed,
                'worst_head_error_ratio': float(ratio),
                'max_individual_head_error_ratio': float((error[tight] / ref_error[tight]).max()),
                'individual_states_checked': int(strict.numel()),
                'individual_states_passing_same_bound': int(strict.sum()),
                'max_state_relative_mae': float((error[magnitude > 0] / magnitude[magnitude > 0]).max()),
            }
            assert passed, (kind, batch, family, name, ratio)
        report['rows'].append(row)
        (here / 'precision_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(kind, batch, 'baseline / current:', [(r['family'], r['baseline']['worst_head_error_ratio'], r['current']['worst_head_error_ratio']) for r in report['rows'][-3:]], flush=True)
report['status'] = 'original_benchmark_pass'
report['summary'] = {}
for name in ('baseline', 'current'):
    values = [row[name] for row in report['rows']]
    report['summary'][name] = {
        'original_cases_passed': sum(v['original_judge_passed'] for v in values),
        'worst_original_benchmark_error_ratio': max(v['worst_head_error_ratio'] for v in values),
        'individual_states_checked': sum(v['individual_states_checked'] for v in values),
        'additional_individual_state_failures': sum(v['individual_states_checked'] - v['individual_states_passing_same_bound'] for v in values),
        'max_individual_head_error_ratio': max(v['max_individual_head_error_ratio'] for v in values),
    }
(here / 'precision_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(kind, 'complete', flush=True)
