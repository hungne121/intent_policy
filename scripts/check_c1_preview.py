"""Check the requested C1 timing from paired preview logs, not dataset safety."""
import argparse
import json
import math
from pathlib import Path


def check(folder):
    result = {}
    for mode in ('Observed', 'Intent'):
        s = json.loads((folder / f'C1_{mode}_summary.json').read_text())
        human, robot = s['human_events'], s['events']
        trace = json.loads((folder / f'C1_{mode}_trace.json').read_text())
        errors = []
        if not s['success']:
            errors.append('task did not complete both trays')
        for i in range(2):
            h = {e['event']: e['t'] for e in human if e['round'] == i}
            if any(k not in h for k in ('reach_cloth', 'lift_cloth', 'withdraw', 'wait_cup_return')):
                errors.append(f'round {i}: missing human event'); continue
            next_reach = next((e['t'] for e in human if e['round'] == i + 1 and e['event'] == 'reach_cloth'), float('inf'))
            events = [e for e in robot if h['reach_cloth'] <= e['t'] < next_reach]
            returns = [e for e in events if e['event'] == 'return_started']
            if len(returns) != 1:
                errors.append(f'round {i}: expected one return event'); continue
            ret = returns[0]
            if mode == 'Observed':
                targets = [e for e in events if e['event'] == 'observed_cloth_target']
                if not targets or targets[0]['t'] < h['lift_cloth']:
                    errors.append(f'round {i}: observed approached before cloth lift')
                if ret['t'] < h['wait_cup_return'] or ret['source'] != 'hand_at_rest':
                    errors.append(f'round {i}: observed returned before the hand rested')
            else:
                forecast = [e for e in events if e['event'] == 'forecast_follow_started']
                if len(forecast) != 1 or forecast[0]['t'] >= h['lift_cloth']:
                    errors.append(f'round {i}: no continuous forecast follow before cloth lift')
                else:
                    start = forecast[0]['t']
                    follow = [f for f in trace if start <= f['t'] < h['lift_cloth'] and f['robot_phase'] == 'forecast_approach']
                    if not follow or not any(math.dist(f['tcp'], follow[0]['tcp']) >= .002 for f in follow):
                        errors.append(f'round {i}: no measured robot motion while following')
                    if any(f['selected'] is not None for f in follow):
                        errors.append(f'round {i}: selected a cloth before observing the lift')
                if not (h['withdraw'] <= ret['t'] < h['wait_cup_return']) or ret['source'] != 'forecast_withdraw':
                    errors.append(f'round {i}: forecast return did not start during withdrawal')
        result[mode] = errors
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder', type=Path)
    report = check(parser.parse_args().folder)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(int(any(report.values())))
