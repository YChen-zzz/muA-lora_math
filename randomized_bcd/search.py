"""Random-permutation BCD and uniformly randomized coordinate descent."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcd_sft.search import REPO, result, run_name, write_json


class BudgetReached(Exception):
    pass


def search(config, method, seed, runner=None):
    if method not in ('rp_bcd', 'r_cd'):
        raise ValueError('method must be rp_bcd or r_cd')
    output = REPO / config['output_dir'] / method / f'seed-{seed}'
    if not output.resolve().is_relative_to((REPO / 'checkpoints/randomized-bcd').resolve()):
        raise ValueError('Use an output_dir under checkpoints/randomized-bcd/')
    snapshot = {**config, 'method':method, 'search_seed':seed}
    path = output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('Use a new output_dir when changing the experiment')
    write_json(path, snapshot)
    rng = random.Random(seed)  # Independent of the training seed (42).
    params = dict(config['defaults'])
    names = list(config['order'])
    history, observed, events, orders = [], {}, [], []
    current_loss = None

    def report(reason):
        best = min(history, key=lambda row:row['loss'])
        answer = {'method':method, 'search_seed':seed, 'completed':len(history),
                  'budget':config['iterations'], 'metric':config['metric'],
                  'best_params':best['params'], 'best_loss':best['loss'],
                  'best_run_dir':best['run_dir'], 'accepted_params':dict(params),
                  'accepted_loss':current_loss, 'coordinate_visits':len(events),
                  'stop_reason':reason}
        write_json(output / 'best.json', answer)
        return answer

    def evaluate(candidate):
        key = run_name(candidate)
        if key not in observed:
            if len(history) >= config['iterations']:
                raise BudgetReached
            observed[key] = result(config, candidate, output, history, runner)
            report('running')
        return observed[key]

    def sweep(name, round_number=None):
        nonlocal params, current_loss
        base = dict(params)
        old_loss = current_loss
        count = len(history)
        best_value, best_loss = params[name], current_loss
        complete = True
        for value in config['grids'][name]:
            candidate = {**base, name:value}
            try:
                loss = evaluate(candidate)
            except BudgetReached:
                complete = False
                break
            if loss < best_loss:
                best_value, best_loss = value, loss
        accepted = complete and old_loss - best_loss > config['min_improvement']
        if accepted:
            params[name], current_loss = best_value, best_loss
        events.append({'visit':len(events)+1, 'round':round_number, 'coordinate':name,
                       'base_params':base, 'base_loss':old_loss,
                       'new_trials':len(history)-count, 'grid_complete':complete,
                       'accepted':accepted, 'accepted_params':dict(params),
                       'accepted_loss':current_loss, 'best_candidate_loss':best_loss})
        write_json(output / 'coordinate_history.json', events)
        report('running')
        return accepted, complete

    current_loss = evaluate(params)
    reason = 'budget_exhausted'
    if method == 'rp_bcd':
        while len(history) < config['iterations'] and len(orders) < config.get('max_rounds', float('inf')):
            order = rng.sample(names, len(names))
            orders.append(order)
            write_json(output / 'permutations.json', orders)
            changed = False
            for name in order:
                accepted, complete = sweep(name, len(orders))
                changed |= accepted
                if not complete:
                    break
            if not complete:
                break
            if not changed:
                reason = 'coordinate_stationary'
                break
    else:
        checked = set()
        while len(history) < config['iterations']:
            name = rng.choice(names)  # Uniform sampling with replacement.
            accepted, complete = sweep(name)
            if not complete:
                break
            if accepted:
                checked = {name}  # This coordinate's line was fully scanned at the new base.
            else:
                checked.add(name)
            if checked == set(names):
                reason = 'coordinate_stationary'
                break
    answer = report(reason)
    print(json.dumps(answer), flush=True)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('randomized_bcd/config.json'))
    parser.add_argument('--method', choices=['rp_bcd','r_cd'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    options = parser.parse_args()
    search(json.loads(options.config.read_text()), options.method, options.seed)
