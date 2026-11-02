"""Loopback-only artifact server with byte ranges for seeking local MP4s."""
from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

from intent_policy.sim.base_env import ROOT


class PreviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.range_bytes = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get('Range')
        if not requested or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
        if not match or not any(match.groups()):
            self.send_error(HTTPStatus.BAD_REQUEST, 'Unsupported byte range')
            return None
        handle = path.open('rb')
        size = path.stat().st_size
        first, last = match.groups()
        if first:
            start, end = int(first), min(int(last), size - 1) if last else size - 1
        else:
            start, end = max(0, size - int(last)), size - 1
        if start > end or start >= size:
            handle.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Last-Modified', self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        handle.seek(start)
        self.range_bytes = end - start + 1
        return handle

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.range_bytes is None:
            return super().copyfile(source, outputfile)
        remaining = self.range_bytes
        while remaining:
            data = source.read(min(65536, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--port', type=int, default=8768)
    args = ap.parse_args()
    directory = (ROOT / args.directory).resolve()
    directory.relative_to((ROOT / 'outputs').resolve())
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(PreviewHandler, directory=str(directory)))
    print(f'Serving {directory} at http://127.0.0.1:{args.port}/index.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
