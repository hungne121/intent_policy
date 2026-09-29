"""View a Phase-1 scenario scene (GUI), optionally with the scripted expert acting, or export
one headless screenshot of a camera after a few ticks."""
import argparse
from pathlib import Path
import time

from intent_policy.benchmark.runner import ExpertAgent
from intent_policy.sim.restricted_action import RestrictedActionConfig, RestrictedActionMapper
from intent_policy.scenarios.scenario_registry import PHASE1_SCENARIOS, make_scenario


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=PHASE1_SCENARIOS, default='instructor_object_to_target')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expert', action='store_true', help='let the scripted expert perform the task')
    p.add_argument('--screenshot', type=Path, help='save one image and exit (headless)')
    p.add_argument('--camera', choices=['scene', 'wrist', 'top', 'front'], default='front')
    p.add_argument('--ticks', type=int, default=40, help='ticks simulated before the screenshot')
    args = p.parse_args()
    sc = make_scenario(args.scenario, gui=args.screenshot is None)
    try:
        sc.reset(args.seed)
        mapper = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
        agent = ExpertAgent()
        agent.reset(sc, mapper)

        def tick():
            command = mapper.map(agent.act(None).action_id).joint_target if args.expert else sc.env.hold()
            return sc.step(command)

        if args.screenshot:
            from PIL import Image
            for _ in range(args.ticks):
                if tick():
                    break
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sc.env.render(args.camera, 960, 720)).save(args.screenshot)
            print(f'saved {args.screenshot}')
            return
        done = False
        while sc.env.viewer.is_running():
            start = time.monotonic()
            if not done:
                done = tick()
                if done:
                    print(f'episode finished: success={sc.success} failure={sc.failure}', flush=True)
            else:
                sc.env.viewer.sync()        # keep showing the final state
            time.sleep(max(0, sc.env.dt - (time.monotonic() - start)))
    finally:
        sc.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped by user.', flush=True)
        raise SystemExit(130)
