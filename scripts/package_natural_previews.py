"""Package selected review cases with actual forecasting diagnostics and a variable-case gallery."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont
import yaml

from intent_policy.sim.base_env import ROOT


def label(key):
    suffixes = [('ERROR_REAL', 'C3 · dự báo thật'), ('ERROR_CONTROLLED', 'C3 · can thiệp'),
                ('TIMING_CONTROLLED', 'Forecast can thiệp'), ('NEXT_ITEM_SWITCH', 'Đổi món lấy'),
                ('WAIT_RESUME', 'Chờ rồi tiếp tục'), ('ALTERNATIVE', 'Có món thay thế'),
                ('SWITCH_HELD', 'C4 · đã nhấc bát A'), ('SWITCH_OPEN', 'C4 · đã mở A'),
                ('CANCEL', 'C4 · hủy nhận'), ('AMBIG_A', 'Mơ hồ → A'), ('AMBIG_B', 'Mơ hồ → B'),
                ('GRIP_A', 'Chỉnh nắm · nâng sớm'), ('GRIP_B', 'Chỉnh nắm · nâng muộn'), ('CLEAR', 'Thuận')]
    return next((v for k, v in suffixes if key.endswith(k)), key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--decode', action='store_true')
    args = ap.parse_args()
    p = ROOT / args.folder
    cfg = yaml.safe_load((p / 'config.yaml').read_text())
    expected = len(cfg['tasks'])
    records, cards, md = {}, [], []
    font = ImageFont.truetype(cfg['video']['font'], 14)
    (p / 'contact_sheets').mkdir(exist_ok=True)
    for group, spec in cfg['review_groups'].items():
        tabs = []
        first = None
        sheet = Image.new('RGB', (384 * len(spec['keys']), 32 + 4 * 210), '#18202b')
        draw = ImageDraw.Draw(sheet)
        md += [f'## {group} — {spec["name"]}', '']
        for col, key in enumerate(spec['keys']):
            f, sf = p / f'{key}.mp4', p / f'{key}.stats.json'
            task = cfg['tasks'][key]
            draw.text((col * 384 + 6, 6), label(key), font=font, fill='white')
            if not f.exists() or not sf.exists() or sf.stat().st_mtime_ns < f.stat().st_mtime_ns:
                if not args.partial:
                    raise RuntimeError(f'Incomplete clip: {key}')
                continue
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                'stream=width,height:format=duration', '-of', 'json', str(f)], text=True))
            stream = probe['streams'][0]
            assert stream['width'] == 960 and stream['height'] == 488, key
            if args.decode:
                check = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(f), '-f', 'null', '-'],
                                       capture_output=True, text=True)
                assert check.returncode == 0, (key, check.stderr)
            stats = json.loads(sf.read_text())
            if task['case_family'] in ('natural_forecast_error', 'controlled_forecast_error', 'local_goal_change', 'local_next_item_change'):
                assert stats.get('action_before_reveal'), (key, stats)
            audit = json.loads((ROOT / 'outputs/natural_case_preview/motion_audit' / f'{key}.json').read_text())
            description = 'ManiCast đóng băng chạy thật trên chuyển động dựng; robot đi theo storyboard, chưa chạy ACT hoặc physics.'
            if task.get('forecast_intervention'):
                description = 'Chấm đỏ là quỹ đạo can thiệp để minh họa bài kiểm có kiểm soát. Không tính là lỗi tự nhiên; chưa đưa vào ACT.'
            if task.get('ambiguity_pair'):
                branch = task['ambiguity_pair']['branch_t']
                description = f'Hai continuation có cùng khớp tay và forecast đến {branch:g} s. Nhãn A/B chỉ dành cho người xem.'
            witness = audit.get('natural_error_witness')
            if witness:
                description = (f'Khung {witness["t_s"]:.2f} s, horizon {witness["horizon_s"]:g} s: '
                               f'cổ tay phải lệch {witness["error_m"] * 100:.1f} cm so với tương lai ghi lại. '
                               'Đây là mẫu được chọn để chẩn đoán, không phải tỷ lệ lỗi tổng thể. Robot vẫn là minh họa.')
            review = sorted((p / 'review_frames' / key).glob('*.jpg'))
            for row, frame in enumerate(review[:4]):
                with Image.open(frame) as im:
                    im.thumbnail((384, 188))
                    sheet.paste(im, (col * 384, 32 + row * 210))
                draw.text((col * 384 + 5, 32 + row * 210 + 190), f't={float(frame.stem):.2f}s', font=font, fill='#b8c7d9')
            at = round(task['review_times'][0] * cfg['fps']) / cfg['fps']
            poster = f'review_frames/{key}/{at:06.2f}.jpg'
            tabs.append(f'<button data-video="{key}.mp4" data-poster="{poster}" '
                        f'data-note="{html.escape(description, quote=True)}">{html.escape(label(key))}</button>')
            if first is None:
                first = (key, poster, description)
            md += [f'- [{label(key)}]({f.resolve()}) — {description}']
            records[key] = dict(file=f.name, title=task['title'], case_family=task['case_family'],
                                duration_s=float(probe['format']['duration']), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                stats=stats, forecast_audit=audit, decoded=bool(args.decode))
        sheet.save(p / 'contact_sheets' / f'{group}.jpg', quality=90)
        md += ['']
        if first:
            key, poster, note = first
            cards.append(f'<article id="{group}"><h2>{html.escape(group + " · " + spec["name"])}</h2>'
                         f'<div class="tabs">{"".join(tabs)}</div><p class="note">{html.escape(note)}</p>'
                         f'<video controls playsinline preload="none" poster="{poster}" src="{key}.mp4"></video>'
                         f'<p><a class="clip-link" href="{key}.mp4">Mở riêng video</a> · '
                         f'<a href="contact_sheets/{group}.jpg">Khung kiểm nội dung</a></p></article>')
    if not args.partial:
        assert len(records) == expected == 26
        assert len({r['sha256'] for r in records.values()}) == expected
    manifest = dict(expected_clips=expected, completed_clips=len(records),
                    robot_controller='hand-authored kinematic storyboard', learned_policy_rollout=False, physics=False,
                    forecaster='frozen ManiCast checkpoint on scripted joint motion',
                    goal_logs_and_annotations_are_evaluation_only=True, records=records)
    (p / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (p / 'index.md').write_text('# Sáu task — preview theo case tự nhiên\n\n'
                               'Dự báo chạy bằng checkpoint; robot vẫn là storyboard. C3 thật và can thiệp được tách riêng.\n\n'
                               + '\n'.join(md) + '\n')
    nav = ''.join(f'<a href="#{g}">{g}</a>' for g in cfg['review_groups'])
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preview theo case tự nhiên</title><style>
body{margin:0;background:#0f141b;color:#eef3fa;font:16px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}
h1{font-size:28px}p{line-height:1.6;color:#b9c8db}nav,.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
button,nav a{background:#202d3d;color:#e2edfc;border:1px solid #3a4e68;border-radius:8px;padding:9px 13px;cursor:pointer;text-decoration:none}
button.active{background:#255f96;border-color:#72afea}article{margin:24px 0;padding:20px;background:#17212d;border:1px solid #304257;border-radius:14px}
h2{font-size:21px;margin:0}.note{font-size:14px;min-height:44px}video{width:100%;border-radius:8px;background:#000}a{color:#91c6ff}
.notice{padding:14px 18px;background:#1c2d3e;border-left:3px solid #6fbee9;border-radius:4px}
</style><main><h1>6 task · 26 preview theo trường hợp phù hợp</h1>
<div class="notice"><p>Chấm xanh: ManiCast thật, 8 mốc cổ tay trong 1 giây. Chấm đỏ: forecast bị can thiệp có kiểm soát.
Robot vẫn đi theo storyboard; chưa chạy policy ACT hoặc mô phỏng tiếp xúc/vật lý. Video dùng để duyệt kịch bản.</p>
<p>Cặp A/B có chung đoạn đầu để kiểm sự mơ hồ. Đổi khay sau khi đã nhấc bát và đổi ngăn sau khi đã mở tạo chi phí sửa cụ thể.
Các nhãn/log/overlay chỉ dành cho người xem và đánh giá, không phải đầu vào camera của policy.</p></div><nav>'''
    page += nav + '</nav>' + ''.join(cards)
    page += '''</main><script>document.querySelectorAll('article').forEach(card=>{
const video=card.querySelector('video'),buttons=card.querySelectorAll('button');buttons[0]?.classList.add('active');
buttons.forEach(b=>b.onclick=()=>{video.pause();video.src=b.dataset.video;video.poster=b.dataset.poster;video.load();
buttons.forEach(x=>x.classList.toggle('active',x===b));card.querySelector('.note').textContent=b.dataset.note;
card.querySelector('.clip-link').href=b.dataset.video;});});</script></html>'''
    (p / 'index.html').write_text(page)
    if not args.partial:
        target = p / 'motion_audit'
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / 'outputs/natural_case_preview/motion_audit', target)
    print(json.dumps(dict(complete=len(records), expected=expected,
                         max_ik_mm=max((r['stats']['ik_max_m'] * 1000 for r in records.values()), default=0),
                         max_reach_mm=max((r['stats']['reach_max_m'] * 1000 for r in records.values()), default=0)), ensure_ascii=False))


if __name__ == '__main__':
    main()
