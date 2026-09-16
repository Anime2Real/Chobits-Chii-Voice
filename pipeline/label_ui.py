#!/usr/bin/env python3
"""人工标注界面: 浏览器中听片段、快捷键打标签, 实时落盘, 可断点续标.

数据: annotations/clips.json (由 prepare_labeling.py 生成)
标签: annotations/labels.json  {文件名: 标签}, 每次按键立即保存

快捷键:
  1 = 小叽(干净)   2 = 不是小叽   3 = 混有其他人声   4 = 小叽但音质差
  U = 不确定      空格 = 重播     ←/→ = 上一条/下一条

用法: .venv/bin/python pipeline/label_ui.py [--port 8017]
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAB_DIR = "annotations"
# 片段音频位置: candidates/review 由 batch.py 产出, clips 由 prepare_labeling.py 产出,
# wavs 为 finalize.py 产物 (兼容历史清单里 src=wavs 的条目)
AUDIO_DIRS = {"wavs": os.path.join("dataset", "wavs"),
              "candidates": os.path.join("build", "candidates"),
              "review": os.path.join("build", "review"),
              "clips": os.path.join("build", "labeling", "clips")}
LABELS_PATH = os.path.join(LAB_DIR, "labels.json")

INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>小叽语音标注</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "PingFang SC", sans-serif;
         background: #1b1d23; color: #e6e6e6; display: flex; height: 100vh; }
  #main { flex: 1; padding: 28px 36px; display: flex; flex-direction: column; gap: 14px; }
  #progress { font-size: 14px; color: #9aa0aa; }
  #bar { height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
  #bar > div { height: 100%; background: #4caf50; width: 0; transition: width .2s; }
  #meta { font-size: 14px; color: #9aa0aa; }
  #meta b { color: #e6e6e6; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 12px;
           margin-right: 6px; background: #444; }
  .gA { background: #2d5a2d; } .gB { background: #7a5c1e; } .gC { background: #555; }
  #text { font-size: 26px; min-height: 40px; line-height: 1.4; }
  audio { width: 100%; }
  #btns { display: flex; gap: 10px; flex-wrap: wrap; }
  button { font-size: 16px; padding: 12px 18px; border: 0; border-radius: 8px;
           cursor: pointer; color: #fff; }
  button small { opacity: .75; margin-left: 6px; }
  .b1 { background: #2e7d32; } .b2 { background: #c62828; } .b3 { background: #ef6c00; }
  .b4 { background: #6a4c93; } .b5 { background: #546e7a; }
  #cur { font-size: 15px; min-height: 22px; font-weight: 600; }
  #keys { font-size: 13px; color: #777; margin-top: auto; }
  #side { width: 300px; border-left: 1px solid #333; display: flex; flex-direction: column; }
  #filter { padding: 8px 10px; font-size: 13px; color: #9aa0aa; border-bottom: 1px solid #333; }
  #list { flex: 1; overflow-y: auto; font-size: 12px; }
  .item { padding: 5px 10px; cursor: pointer; border-left: 4px solid transparent;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .item:hover { background: #2a2d35; }
  .item.active { background: #31352d; border-left-color: #4caf50; }
  .l-chi { border-left-color: #2e7d32; } .l-not_chi { border-left-color: #c62828; }
  .l-mixed { border-left-color: #ef6c00; } .l-bad { border-left-color: #6a4c93; }
  .l-unsure { border-left-color: #546e7a; }
</style></head><body>
<div id="main">
  <div id="progress">加载中…</div>
  <div id="bar"><div></div></div>
  <div id="meta"></div>
  <div id="text"></div>
  <audio id="player" controls autoplay></audio>
  <div id="btns">
    <button class="b1" onclick="setLabel('chi')">1 小叽<small>干净</small></button>
    <button class="b2" onclick="setLabel('not_chi')">2 不是<small>其他人/非语音</small></button>
    <button class="b3" onclick="setLabel('mixed')">3 混合<small>小叽+其他人</small></button>
    <button class="b4" onclick="setLabel('bad')">4 音质差<small>是小叽但不可用</small></button>
    <button class="b5" onclick="setLabel('unsure')">U 不确定</button>
  </div>
  <div id="cur"></div>
  <div id="keys">空格 重播 · ← 上一条 · → 下一条 · 标签自动保存, 随时可关闭续标</div>
</div>
<div id="side">
  <div id="filter"><label><input type="checkbox" id="onlyUn" onchange="renderList()"> 只看未标注</label>
    <span id="stats" style="float:right"></span></div>
  <div id="list"></div>
</div>
<script>
let clips = [], labels = {}, idx = 0;
const player = document.getElementById('player');
const LABEL_NAMES = {chi:'小叽', not_chi:'不是', mixed:'混合', bad:'音质差', unsure:'不确定'};

async function init() {
  const [m, l] = await Promise.all([
    fetch('/manifest').then(r => r.json()), fetch('/labels').then(r => r.json())]);
  clips = m; labels = l;
  idx = clips.findIndex(c => !labels[c.file]);
  if (idx < 0) idx = 0;
  show(); renderList();
}

function show() {
  const c = clips[idx];
  player.src = '/audio/' + c.src + '/' + encodeURIComponent(c.file) + '.wav';
  player.play().catch(() => {});
  document.getElementById('text').textContent = c.text || '(无转写文本)';
  const score = c.prob !== undefined ? `prob=${c.prob}` : `sim=${c.sim} margin=${c.margin}`;
  document.getElementById('meta').innerHTML =
    `<span class="badge g${c.group}">${c.group}组</span><b>${c.file}</b> · ` +
    `${c.dur.toFixed(1)}s · ${score}`;
  document.getElementById('cur').textContent =
    labels[c.file] ? '当前标签: ' + LABEL_NAMES[labels[c.file]] : '';
  const done = clips.filter(c => labels[c.file]).length;
  document.getElementById('progress').textContent =
    `进度 ${done} / ${clips.length} (第 ${idx + 1} 条)`;
  document.querySelector('#bar > div').style.width = (100 * done / clips.length) + '%';
  document.querySelectorAll('.item').forEach((el, i) =>
    el.classList.toggle('active', i === idx));
  const act = document.querySelector('.item.active');
  if (act) act.scrollIntoView({block: 'nearest'});
}

function setLabel(v) {
  const c = clips[idx];
  labels[c.file] = v;
  fetch('/label', {method: 'POST', body: JSON.stringify({file: c.file, label: v})});
  show(); renderList();
  if (idx < clips.length - 1) { idx++; show(); renderList(); }
}

function move(d) {
  idx = Math.max(0, Math.min(clips.length - 1, idx + d));
  show();
}

function renderList() {
  const onlyUn = document.getElementById('onlyUn').checked;
  const list = document.getElementById('list');
  list.innerHTML = '';
  const cnt = {chi: 0, not_chi: 0, mixed: 0, bad: 0, unsure: 0};
  clips.forEach((c, i) => {
    const l = labels[c.file];
    if (l) cnt[l]++;
    if (onlyUn && l) return;
    const div = document.createElement('div');
    div.className = 'item' + (l ? ' l-' + l : '') + (i === idx ? ' active' : '');
    div.textContent = (l ? LABEL_NAMES[l] + ' | ' : '') + c.file + ' ' + (c.text || '');
    div.onclick = () => { idx = i; show(); };
    list.appendChild(div);
  });
  document.getElementById('stats').textContent =
    `叽${cnt.chi} 否${cnt.not_chi} 混${cnt.mixed} 差${cnt.bad} 疑${cnt.unsure}`;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox') return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); player.currentTime = 0; player.play(); }
  else if (k === '1') setLabel('chi');
  else if (k === '2') setLabel('not_chi');
  else if (k === '3') setLabel('mixed');
  else if (k === '4') setLabel('bad');
  else if (k === 'u' || k === 'U') setLabel('unsure');
  else if (k === 'ArrowLeft') move(-1);
  else if (k === 'ArrowRight') move(1);
});
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/manifest":
            with open(os.path.join(LAB_DIR, "clips.json"), "rb") as f:
                self._send(200, f.read())
        elif path == "/labels":
            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, "rb") as f:
                    self._send(200, f.read())
            else:
                self._send(200, "{}")
        elif path.startswith("/audio/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[2] in AUDIO_DIRS:
                name = os.path.basename(parts[3])
                fp = os.path.join(AUDIO_DIRS[parts[2]], name)
                if os.path.isfile(fp):
                    with open(fp, "rb") as f:
                        self._send(200, f.read(), "audio/wav")
                    return
            self._send(404, "{}")
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path.split("?")[0] != "/label":
            self._send(404, "{}")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        labels = {}
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, encoding="utf-8") as f:
                labels = json.load(f)
        labels[body["file"]] = body["label"]
        tmp = LABELS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=1)
        os.replace(tmp, LABELS_PATH)
        self._send(200, "{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(LAB_DIR, "clips.json")):
        raise SystemExit("先运行: .venv/bin/python pipeline/prepare_labeling.py")
    url = f"http://localhost:{args.port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"标注界面: {url}  (Ctrl+C 退出, 标签已实时保存)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
