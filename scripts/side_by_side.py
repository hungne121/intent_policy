"""Side-by-side evaluation videos of two models on the same episodes (same seed): left `--left`, right `--right`, each
labelled with the model, the case and the result; the shorter one holds its last frame. From the videos recorded
while the evaluations ran (scripts/evaluate.py --videos), no re-simulation.
  ./run.sh -m scripts.side_by_side --left outputs/eval/t2deep/<run>/A_ft --right outputs/eval/t2deep/<run>/D_ft \\
      --names "A (late)" "D-tok (early + ManiCast)" --out outputs/videos/<date>/A_vs_D
"""
import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path


def videos(run: Path) -> dict[int, Path]:
    return {int(re.search(r'seed(\d+)', p.name).group(1)): p for p in run.glob('videos/*/*.mp4')}


def info(run: Path, seed: int) -> tuple[str, str]:
    f = next(run.glob(f'episodes/*/*/*seed{seed}.json.gz'))
    r = json.load(gzip.open(f))
    res = 'SUCCESS' if r['success'] else f"FAIL {r['failure']}"
    return r['human_trajectory_variant'].get('case', ''), f"{res}  t={r['duration_s']:.1f}s"


def duration(p: Path) -> float:
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(p)],
                                capture_output=True, text=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left', required=True)
    ap.add_argument('--right', required=True)
    ap.add_argument('--names', nargs=2, default=['left', 'right'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    left, right, out = Path(args.left), Path(args.right), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lv, rv = videos(left), videos(right)
    for seed in sorted(set(lv) & set(rv)):
        case, lres = info(left, seed)
        _, rres = info(right, seed)
        T = max(duration(lv[seed]), duration(rv[seed]))
        lab = lambda name, res: (f"drawtext=text='{name} | {case} | {res}':x=10:y=26:fontsize=18:fontcolor=white:"
                                 "box=1:boxcolor=black@0.7:boxborderw=6")
        f = (f"[0:v]tpad=stop_mode=clone:stop_duration={T},trim=duration={T},{lab(args.names[0], lres)}[a];"
             f"[1:v]tpad=stop_mode=clone:stop_duration={T},trim=duration={T},{lab(args.names[1], rres)}[b];"
             "[a][b]hstack=inputs=2[v]")
        dst = out / f'{case}_seed{seed}.mp4'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(lv[seed]), '-i', str(rv[seed]), '-filter_complex', f,
                        '-map', '[v]', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', str(dst)], check=True)
        print(f'{dst}  ({args.names[0]}: {lres} | {args.names[1]}: {rres})', flush=True)


if __name__ == '__main__':
    main()
