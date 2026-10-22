"""Contact sheet: frames at given times (s) of one video, with timestamps. usage: sheet.py video out.jpg t1 t2 ..."""
import sys, av
from PIL import Image, ImageDraw
src, out, ts = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]
c = av.open(src); s = c.streams.video[0]
print('codec', s.codec_context.name, 'fps', s.average_rate, 'frames', s.frames, s.width, s.height)
frames, want = {}, sorted(ts)
for i, fr in enumerate(c.decode(s)):
    t = i / float(s.average_rate)
    for w in want:
        if w not in frames and t >= w - 1e-6:
            frames[w] = fr.to_image().resize((320, 240))
    if len(frames) == len(want): break
cols = 4; rows = (len(want) + cols - 1) // cols
sheet = Image.new('RGB', (320 * cols, 240 * rows))
for j, w in enumerate(want):
    im = frames.get(w)
    if im is None: continue
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 70, 16], fill=(0, 0, 0)); d.text((3, 2), f't={w:.1f}s', fill=(255, 255, 0))
    sheet.paste(im, ((j % cols) * 320, (j // cols) * 240))
sheet.save(out, quality=85)
