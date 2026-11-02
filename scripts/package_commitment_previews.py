"""Assemble ten chaptered review videos and an offline gallery."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from intent_policy.sim.base_env import ROOT


def wrapped(draw, text, font, width):
    result, line = [], ''
    for w in text.split():
        trial = (line + ' ' + w).strip()
        if draw.textlength(trial, font=font) > width and line:
            result.append(line)
            line = w
        else:
            line = trial
    return result + ([line] if line else [])


def card(path, cfg, group, spec, task, index):
    W = cfg['video']['width'] * len(cfg['show'])
    H = cfg['video']['height'] + cfg['video']['caption_h']
    im = Image.new('RGB', (W, H), '#101a27')
    d = ImageDraw.Draw(im)
    fp = cfg['video']['font']
    f1, f2, f3 = [ImageFont.truetype(fp, size) for size in (28, 22, 17)]
    color = {'Instructor': '#80b8ff', 'Collaborator': '#83d4a3', 'Cowork': '#f1c77b'}[spec['role']]
    d.rectangle((0, 0, 12, H), fill=color)
    d.text((36, 28), f'{group}  ·  {spec["role"]}  ·  Lượt {index + 1}/{len(spec["keys"])}', font=f2, fill=color)
    y = 80
    for line in wrapped(d, spec['name'], f1, W - 72):
        d.text((36, y), line, font=f1, fill='white')
        y += 37
    y += 16
    d.text((36, y), task['chapter'], font=f2, fill='#ffd98d')
    y += 48
    for line in wrapped(d, spec['question'], f3, W - 90):
        d.text((36, y), line, font=f3, fill='#d1ddeb')
        y += 26
    d.text((36, H - 100), 'Bắt đầu từ tay trống; quá trình lấy đồ/chuẩn bị được hiển thị.', font=f3, fill='#d1ddeb')
    d.text((36, H - 72), 'Mỗi lượt là một episode riêng, đặt lại cùng trạng thái đầu.', font=f3, fill='#d1ddeb')
    d.text((36, H - 40), 'Preview kịch bản — chưa phải kết quả ACT, ManiCast hoặc physics.', font=f3, fill='#91a6bc')
    fps = cfg['fps']
    with av.open(str(path), 'w', options={'movflags': 'faststart'}) as out:
        stream = out.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = W, H, 'yuv420p'
        stream.options = dict(crf='22', preset='veryfast')
        frame = np.asarray(im)
        for _ in range(fps * 3):
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                out.mux(packet)
        for packet in stream.encode(): out.mux(packet)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'stream=width,height:format=duration', '-of', 'json', str(path)], text=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    folder = ROOT / args.folder
    cfg_path = ROOT / 'configs/scenarios/commitment_review_10.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    shutil.copy2(cfg_path, folder / 'config.yaml')
    (folder / 'cards').mkdir(exist_ok=True)
    (folder / 'contact_sheets').mkdir(exist_ok=True)
    manifest = dict(scope='kinematic_storyboards', forecasts_displayed=False, learned_policy_rollout=False,
                    physics=False, main_videos={}, expected_videos=10, expected_roles=cfg['review_protocol']['roles'])
    articles, md = [], ['# 10 task — quá trình chuẩn bị và mức cam kết', '', 'Video dùng để duyệt nội dung; chưa phải rollout policy.', '']
    font = ImageFont.truetype(cfg['video']['font'], 15)
    for group, spec in cfg['review_groups'].items():
        keys = spec['keys']
        incomplete = False
        for key in keys:
            sf, vf = folder / 'segments' / f'{key}.stats.json', folder / 'segments' / f'{key}.mp4'
            if not sf.exists() or not vf.exists() or sf.stat().st_mtime_ns < vf.stat().st_mtime_ns:
                incomplete = True
            elif not json.loads(sf.read_text()).get('state_probes'):
                incomplete = True
        if incomplete:
            if args.partial: continue
            raise RuntimeError(f'Incomplete group {group}')
        concat, chapters, offset = [], [], 0.0
        rows = max(5, max(len(list((folder / 'segments/review_frames' / key).glob('*.jpg'))) for key in keys))
        sheet = Image.new('RGB', (480 * len(keys), 32 + rows * 270), '#14202c')
        draw = ImageDraw.Draw(sheet)
        for i, key in enumerate(keys):
            t = cfg['tasks'][key]
            segment = folder / 'segments' / f'{key}.mp4'
            stats = json.loads((folder / 'segments' / f'{key}.stats.json').read_text())
            assert stats['ik_max_m'] < .001, (key, stats['ik_max_m'])
            assert stats['reach_max_m'] < .003, (key, stats['reach_max_m'])
            assert stats['state_probes'][0]['held_by_robot'] == [], key
            intro = folder / 'cards' / f'{key}.mp4'
            card(intro, cfg, group, spec, t, i)
            concat.extend([intro, segment])
            duration = float(probe(segment)['format']['duration'])
            chapters.append(dict(key=key, title=t['chapter'], start_s=offset,
                                 episode_start_s=offset + 3, duration_s=duration, stats=stats))
            offset += 3 + duration
            draw.text((480 * i + 6, 6), t['chapter'], font=font, fill='white')
            frames = sorted((folder / 'segments/review_frames' / key).glob('*.jpg'))
            for row, frame in enumerate(frames):
                with Image.open(frame) as im:
                    im.thumbnail((480, 244))
                    sheet.paste(im, (i * 480, 32 + row * 270))
                draw.text((i * 480 + 6, 32 + row * 270 + 246), f't = {float(frame.stem):.2f} s', font=font, fill='#b9cbe0')
        sheet.save(folder / 'contact_sheets' / f'{group}.jpg', quality=92)
        listing = folder / 'cards' / f'{group}.txt'
        # All paths are generated local names without apostrophes.
        listing.write_text(''.join(f"file '{p.resolve()}'\n" for p in concat))
        result = folder / f'{group}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
                        '-c', 'copy', '-movflags', '+faststart', str(result)], check=True)
        media = probe(result)
        assert (media['streams'][0]['width'], media['streams'][0]['height']) == (960, 488), group
        check = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(result), '-f', 'null', '-'], capture_output=True, text=True)
        assert check.returncode == 0 and not check.stderr.strip(), (group, check.stderr)
        poster_key = keys[-1] if group in ('C3', 'C4') else keys[0]
        frames = sorted((folder / 'segments/review_frames' / poster_key).glob('*.jpg'))
        poster = str(frames[min(2, len(frames) - 1)].relative_to(folder))
        manifest['main_videos'][group] = dict(file=result.name, role=spec['role'], title=spec['name'],
            chapters=chapters, duration_s=float(media['format']['duration']), decoded=True,
            sha256=hashlib.sha256(result.read_bytes()).hexdigest())
        buttons = ''.join(f'<button data-time="{c["start_s"]:.4f}">{html.escape(c["title"])}</button>' for c in chapters)
        articles.append(f'<article id="{group}"><div class="role">{spec["role"]} · {group}</div><h2>{html.escape(spec["name"])}</h2>'
                        f'<p>{html.escape(spec["question"])}</p><div class="chapters">{buttons}</div>'
                        f'<video controls preload="metadata" playsinline poster="{poster}" src="{group}.mp4"></video>'
                        f'<p class="links"><a href="{group}.mp4">Mở riêng video</a> · <a href="contact_sheets/{group}.jpg">Xem các khung chính</a></p></article>')
        md += [f'- [{group} — {spec["name"]}]({result.resolve()}) · {len(chapters)} lượt']
    manifest['completed_videos'] = len(manifest['main_videos'])
    if not args.partial:
        assert manifest['completed_videos'] == 10
        assert len({v['sha256'] for v in manifest['main_videos'].values()}) == 10
        # Assert actual object commitment, not just an end-effector displacement.
        rec = manifest['main_videos']
        def state(group, chapter, label):
            return next(p for p in rec[group]['chapters'][chapter]['stats']['state_probes'] if p['label'] == label)
        assert state('C3', 1, 'before_grasp')['held_by_robot'] == []
        assert state('C3', 2, 'late_cue_state')['held_by_robot'] == ['BOWL0']
        assert state('C4', 1, 'before_open')['objects']['DRAWER0']['pos'][1] == .13
        assert abs(state('C4', 2, 'late_cue_state')['objects']['DRAWER0']['pos'][1] + .01) < .002
        for c in range(3):
            for obj in ('DRAWER0', 'DRAWER1'):
                assert abs(state('C4', c, 'final')['objects'][obj]['pos'][1] - .13) < .002
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (folder / 'index.md').write_text('\n'.join(md) + '\n')
    nav = ''.join(f'<a href="#{g}">{g}</a>' for g in cfg['review_groups'])
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>10 task — chuẩn bị và đổi ý sớm/muộn</title><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#eef4fc;font:16px system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:28px}
h1{font-size:29px;line-height:1.3}p{line-height:1.55;color:#bacbdf}nav,.chapters{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
a{color:#8ec8ff}nav a,button{border:1px solid #435c77;background:#25374b;color:#edf5ff;border-radius:8px;padding:9px 13px;text-decoration:none;cursor:pointer}
button.active{background:#2b689c;border-color:#92caff}article{padding:20px;margin:24px 0;background:#182533;border:1px solid #344b63;border-radius:13px;scroll-margin-top:12px}
h2{font-size:22px;margin:7px 0}.role{font-size:13px;color:#88c9e8;text-transform:uppercase;letter-spacing:.07em}video{width:100%;display:block;background:black;border-radius:8px}
.notice{padding:14px 18px;border-left:3px solid #78badb;background:#1c3042}.links{font-size:14px;margin-bottom:0}
</style><main><h1>10 task · xem từ lúc robot bắt đầu chuẩn bị</h1><div class="notice"><p>3 Instructor · 4 Collaborator · 3 Cowork. Mỗi task có một video; các nút bên dưới cho phép tới từng lượt.</p>
<p>Trải khăn và ngăn kéo: cùng trạng thái đầu, so giữ đích / đổi trước gắp–mở / đổi sau cam kết. Mỗi lượt được đặt lại scene và có bảng mở đầu.</p>
<p>Video dùng để duyệt nội dung. Robot chạy kịch bản; chưa phải kết quả policy, dự báo ManiCast hay kiểm chứng va chạm. Thời gian video không chứng minh lợi thế tốc độ.</p></div><nav>'''
    page += nav + '</nav>' + ''.join(articles)
    page += '''</main><script>document.querySelectorAll('article').forEach(card=>{const v=card.querySelector('video'), bs=card.querySelectorAll('button');
bs.forEach(b=>b.onclick=()=>{document.querySelectorAll('video').forEach(x=>{if(x!==v)x.pause()});
const seek=()=>{v.currentTime=Number(b.dataset.time);v.play().catch(()=>{})};if(v.readyState>=1)seek();else{v.addEventListener('loadedmetadata',seek,{once:true});v.load()}
bs.forEach(x=>x.classList.toggle('active',x===b))});});</script></html>'''
    (folder / 'index.html').write_text(page)
    print(json.dumps(dict(videos=manifest['completed_videos'], total_seconds=sum(v['duration_s'] for v in manifest['main_videos'].values())), ensure_ascii=False))


if __name__ == '__main__': main()
