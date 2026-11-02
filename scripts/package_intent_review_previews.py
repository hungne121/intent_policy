"""Build a local gallery, contact sheets, and a verified manifest for the 50 previews."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import yaml

from intent_policy.sim.base_env import ROOT

LABELS = {'BASE': 'Workflow', 'C1': 'Đúng', 'C2': 'Mơ hồ', 'C3': 'Dự báo sai', 'C4': 'Đổi ý'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    folder = ROOT / args.folder
    cfg = yaml.safe_load((folder / 'config.yaml').read_text())
    cases, records, rows, cards = list(LABELS), {}, [], []
    font = ImageFont.truetype(cfg['video']['font'], 15)
    sheet_dir = folder / 'contact_sheets'
    sheet_dir.mkdir(exist_ok=True)
    for code in ('I1', 'I2', 'I3', 'C1', 'C2', 'C3', 'C4', 'W1', 'W2', 'W3'):
        row = [code]
        tabs, videos = [], []
        sheet = Image.new('RGB', (5 * 384, 4 * 210 + 32), '#181c22')
        draw = ImageDraw.Draw(sheet)
        any_frame = False
        for col, case in enumerate(cases):
            key = f'{code}_{case}'
            p = folder / f'{key}.mp4'
            task = cfg['tasks'][key]
            draw.text((col * 384 + 6, 6), f'{code} | {LABELS[case]}', font=font, fill='white')
            stat_file = folder / f'{key}.stats.json'
            if not p.exists() or not stat_file.exists() or stat_file.stat().st_mtime_ns < p.stat().st_mtime_ns:
                if not args.partial:
                    raise RuntimeError(f'Missing completed clip: {p}')
                row.append('Đang dựng')
                continue
            stream = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=width,height:format=duration', '-of', 'json', str(p)], text=True))
            assert stream['streams'][0]['width'] == 960 and stream['streams'][0]['height'] == 464, key
            stats = json.loads((folder / f'{key}.stats.json').read_text())
            if case in ('C3', 'C4'):
                assert stats.get('action_before_reveal'), f'{key}: robot did not move before correction'
            records[key] = dict(file=p.name, duration_s=float(stream['format']['duration']),
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest(), stats=stats,
                                title=task['title'], role=task['role'], decision=task.get('decision'),
                                review_times=task['review_times'])
            frame_dir = folder / 'review_frames' / key
            frame_dir.mkdir(parents=True, exist_ok=True)
            for wanted in task['review_times']:
                at = round(float(wanted) * cfg['fps']) / cfg['fps']
                target = frame_dir / f'{at:06.2f}.jpg'
                if not target.exists():
                    subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', str(at), '-i', str(p),
                                    '-frames:v', '1', '-y', str(target)], check=True)
            row.append(f'[{LABELS[case]}]({p.resolve()})')
            first_at = round(float(task['review_times'][0]) * cfg['fps']) / cfg['fps']
            poster = f'review_frames/{key}/{first_at:06.2f}.jpg'
            tabs.append(f'<button data-case="{case}" data-poster="{poster}">{LABELS[case]}</button>')
            videos.append(f'<option value="{p.name}">{LABELS[case]}</option>')
            frames = sorted((folder / 'review_frames' / key).glob('*.jpg'))
            for j, frame in enumerate(frames[:4]):
                with Image.open(frame) as src:
                    src.thumbnail((384, 186))
                    sheet.paste(src, (col * 384, 32 + j * 210))
                draw.text((col * 384 + 6, 32 + j * 210 + 187), f't={float(frame.stem):.2f}s', font=font, fill='#b9c5d6')
                any_frame = True
        if any_frame:
            sheet.save(sheet_dir / f'{code}.jpg', quality=90)
        rows.append('| ' + ' | '.join(row) + ' |')
        base = cfg['tasks'][f'{code}_BASE']
        name = base['title'].split(' | ')[0]
        first_at = round(float(base['review_times'][0]) * cfg['fps']) / cfg['fps']
        poster = f'review_frames/{code}_BASE/{first_at:06.2f}.jpg'
        cards.append(f'<article id="{code}"><h2>{html.escape(name)}</h2>'
                     f'<div class="tabs">{"".join(tabs)}</div>'
                     f'<video controls playsinline preload="none" poster="{poster}" src="{code}_BASE.mp4"></video>'
                     f'<p class="selected">Workflow</p><a href="contact_sheets/{code}.jpg">Xem các khung kiểm tra</a></article>')
    if not args.partial:
        assert len(records) == 50
        assert len({r['sha256'] for r in records.values()}) == 50, 'Duplicate video files'
    manifest = dict(preview_only=True, physics=False, manicast_inference=False, policy_rollout=False,
                    expected_clips=50, completed_clips=len(records), clips=records)
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (folder / 'index.md').write_text(
        '# 10 task × 5 preview\n\n'
        'Mỗi task: workflow đầy đủ + C1 dự báo đúng + C2 mơ hồ + C3 dự báo sai + C4 người đổi ý. '
        'Các chuyển động/dự báo/phản ứng được viết tay để duyệt; chưa chạy ManiCast, policy hoặc physics.\n\n'
        '| Task | Workflow | C1 đúng | C2 mơ hồ | C3 sai | C4 đổi ý |\n'
        '|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n')
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50 preview HRI</title><style>
body{margin:0;background:#0f141b;color:#e9eef5;font:16px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}
h1{font-size:28px}p{color:#b8c7d9;line-height:1.6}nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}
a{color:#83bcff}nav a,button{background:#202c3b;border:1px solid #34475e;border-radius:8px;padding:9px 14px;color:#dce9fb;text-decoration:none;cursor:pointer}
article{background:#17202b;padding:20px;margin:24px 0;border:1px solid #2e3e51;border-radius:14px}h2{font-size:21px;margin:0 0 16px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}button.active{background:#245b91;border-color:#6ba6e4}
video{width:100%;background:black;border-radius:8px}.selected{margin:6px 0}
</style><main><h1>10 task · 50 preview</h1>
<p>Mỗi task có workflow đầy đủ và bốn trường hợp. C3: dự báo minh họa sai, người giữ mục tiêu. C4: người đổi ý sau khi robot đã hành động.
Các clip là storyboard để duyệt nội dung; chưa chạy ManiCast, policy hoặc physics. Chọn một trường hợp dưới tên task để xem.</p><nav>'''
    page += ''.join(f'<a href="#{c}">{c}</a>' for c in ('I1','I2','I3','C1','C2','C3','C4','W1','W2','W3'))
    page += '</nav>' + '\n'.join(cards)
    page += '''</main><script>
document.querySelectorAll('article').forEach(card=>{
const video=card.querySelector('video'); const buttons=card.querySelectorAll('button');
buttons[0]?.classList.add('active');buttons.forEach(button=>button.onclick=()=>{
video.pause();video.src=card.id+'_'+button.dataset.case+'.mp4';video.poster=button.dataset.poster;video.load();
buttons.forEach(b=>b.classList.toggle('active',b===button));card.querySelector('.selected').textContent=button.textContent;
});});
</script></html>'''
    (folder / 'index.html').write_text(page)
    print(json.dumps(dict(completed=len(records), expected=50,
                         max_ik_mm=max((r['stats']['ik_max_m'] * 1000 for r in records.values()), default=0),
                         max_reach_mm=max((r['stats']['reach_max_m'] * 1000 for r in records.values()), default=0)), ensure_ascii=False))


if __name__ == '__main__':
    main()
