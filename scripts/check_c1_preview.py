"""Check the requested C1 timing from paired preview logs, not dataset safety."""
import argparse
import json
from pathlib import Path


def check(folder):
    result = {}
    for mode in ('Observed', 'Intent'):
        s = json.loads((folder / f'C1_{mode}_summary.json').read_text())
        human, robot = s['human_events'], s['events']
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
                forecast = [e for e in events if e['event'] == 'forecast_target']
                closes = [e for e in events if e['event'] == 'close']
                if len(forecast) != 1 or not closes or forecast[0]['cup'] != closes[0]['cup']:
                    errors.append(f'round {i}: forecast target missing or changed')
                if not forecast or forecast[0]['t'] >= h['lift_cloth']:
                    errors.append(f'round {i}: no forecast approach before cloth lift')
                if not (h['withdraw'] <= ret['t'] < h['wait_cup_return']) or ret['source'] != 'forecast_withdraw':
                    errors.append(f'round {i}: forecast return did not start during withdrawal')
        result[mode] = errors
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder', type=Path)
    report = check(parser.parse_args().folder)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(int(any(report.values())))
