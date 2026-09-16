#!/usr/bin/env python3
"""数据集一致性校验 (纯标准库, 可直接运行, 也可 import 核心函数).

用法: python tools/validate.py [仓库根目录]    (默认: 本文件所在的仓库根)

对照 git 跟踪的 csv/json 做结构断言, 不依赖 dataset/wavs/ 音频本体 (音频被 gitignore):
  1. metadata.csv (file|text) 与 metadata_full.csv 的文件名集合、行数完全一致
  2. annotations/labels.json 覆盖 metadata 全部文件名, 标签值合法 (chi/not_chi/mixed/bad/unsure),
     并统计输出各标签数量
  3. metadata_full.csv 的 source 字段合法 (以 metadata_schema.SOURCES 为准)
  4. start/end 数值合法 (0 <= start < end), 文件名符合内容寻址格式 (复用 metadata_schema 命名规则)
  5. transcripts.csv 覆盖 ep01-ep24 共 24 话 (跳过 8.5/16.5/24.5 总集篇), 列结构合法,
     in_dataset 标记与 metadata 片段按区间重叠可互查 (与 build_transcript_index.py 同口径)
  6. 实测片段数/总时长/标签数/transcripts 行数 与 README 声明比对: 不一致仅 [WARN],
     不判失败 (数据会增长, README 可能滞后; 结构性断言才失败退出非零)

退出码: 有 [FAIL] 项 -> 1; 仅 [WARN] 或全 [OK] -> 0.
报告为人类可读分节输出, 每条带 [OK]/[WARN]/[FAIL] 前缀.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE_DIR = os.path.join(_REPO_ROOT, "pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import metadata_schema as ms  # noqa: E402

VALID_LABELS = ("chi", "not_chi", "mixed", "bad", "unsure")
TRANSCRIPT_FIELDS = ["ep", "start", "end", "dur", "text", "chi_prob", "in_dataset"]

EP_DIR_RE = re.compile(r"^ep\d{2}(?:_\d+)?$")
CLIP_NAME_RE = re.compile(r"^(ep\d{2}(?:_\d+)?)_(\d+\.\d{2})s$")
# TV 全 24 话 (跳过 8.5/16.5/24.5 总集篇): 当前数据的实测覆盖 (ep01-ep24)
EXPECTED_EPS = [f"ep{i:02d}" for i in range(1, 25)]
DURATION_TOL_MIN = 0.1  # README "约 21.4 分钟" 类声明的容差

_DECL_PATTERNS = {
    "clips": re.compile(r"片段数量\s*\|\s*(\d+)"),
    "duration_min": re.compile(r"总时长\s*\|\s*约\s*([0-9]+(?:\.[0-9]+)?)\s*分钟"),
    "labels": re.compile(r"(\d+)\s*条人工标签"),
    "transcripts": re.compile(r"全集台词索引\s*\((\d+)\s*句\)"),
}


class Reporter:
    """分节收集校验结果, 统计失败/警告数; stream 为 None 时静默 (供测试)."""

    def __init__(self, stream=None):
        self.stream = stream
        self.fails = 0
        self.warns = 0

    def section(self, title):
        if self.stream:
            print(f"\n== {title} ==", file=self.stream)

    def ok(self, msg):
        if self.stream:
            print(f"[OK]   {msg}", file=self.stream)

    def warn(self, msg):
        self.warns += 1
        if self.stream:
            print(f"[WARN] {msg}", file=self.stream)

    def fail(self, msg):
        self.fails += 1
        if self.stream:
            print(f"[FAIL] {msg}", file=self.stream)

    @property
    def passed(self):
        return self.fails == 0


def ep_label_of(ep_dir):
    """'ep01'->'01', 'ep08_5'->'08.5' (ep_dir_name 的逆操作, 供规范性格式校验)."""
    return ep_dir[2:].replace("_", ".")


def check_clip_name(name):
    """内容寻址片段名规范校验: 返回问题描述列表 (空 = 合法).

    与 metadata_schema.clip_name 同一格式约定: {ep}_{起始秒:08.2f}s, ep 为 ep_dir_name 形式.
    """
    problems = []
    m = CLIP_NAME_RE.match(name)
    if not m:
        return [f"不符合内容寻址格式 {{ep}}_{{起始秒:08.2f}}s: {name!r}"]
    ep_dir, sec = m.group(1), m.group(2)
    if ep_label_of(ep_dir).isdigit() and ms.ep_dir_name(ep_label_of(ep_dir)) != ep_dir:
        problems.append(f"ep 部分非规范 (ep_dir_name 应得 {ms.ep_dir_name(ep_label_of(ep_dir))!r}): {name!r}")
    if f"{float(sec):08.2f}" != sec:
        problems.append(f"起始秒部分非 %08.2f 规范: {name!r}")
    return problems


def check_metadata_pair(rep, pairs, full_rows):
    """校验项 1: metadata.csv 与 metadata_full.csv 文件名集合/行数完全一致."""
    pipe_names = [name for name, _ in pairs]
    full_names = [r["file"] for r in full_rows]
    dup_pipe = [n for n, c in Counter(pipe_names).items() if c > 1]
    dup_full = [n for n, c in Counter(full_names).items() if c > 1]
    if dup_pipe:
        rep.fail(f"metadata.csv 存在重复文件名 {len(dup_pipe)} 个: {dup_pipe[:3]}")
    if dup_full:
        rep.fail(f"metadata_full.csv 存在重复文件名 {len(dup_full)} 个: {dup_full[:3]}")
    if not dup_pipe and not dup_full:
        rep.ok(f"两表均无重复文件名 (各 {len(pipe_names)} 行)")
    only_pipe = sorted(set(pipe_names) - set(full_names))
    only_full = sorted(set(full_names) - set(pipe_names))
    if only_pipe or only_full:
        rep.fail(f"文件名集合不一致: 仅 metadata.csv 有 {len(only_pipe)} 个 {only_pipe[:3]},"
                 f" 仅 metadata_full.csv 有 {len(only_full)} 个 {only_full[:3]}")
    else:
        rep.ok(f"文件名集合完全一致 ({len(set(pipe_names))} 个)")
    if len(pairs) != len(full_rows):
        rep.fail(f"行数不等: metadata.csv {len(pairs)} 行 vs metadata_full.csv {len(full_rows)} 行")
    else:
        rep.ok(f"行数一致 ({len(pairs)} 行)")
    return dup_pipe or dup_full or only_pipe or only_full


def check_labels(rep, labels, meta_files):
    """校验项 2: labels.json 覆盖全部 metadata 文件名, 标签值合法并统计."""
    bad_value = sorted({v for v in labels.values() if v not in VALID_LABELS})
    if bad_value:
        rep.fail(f"存在非法标签值 {bad_value} (合法值: {list(VALID_LABELS)})")
    else:
        rep.ok(f"{len(labels)} 条标签值全部合法 {list(VALID_LABELS)}")
    missing = sorted(meta_files - set(labels))
    if missing:
        rep.fail(f"labels.json 未覆盖 metadata 全部文件名, 缺 {len(missing)} 个: {missing[:3]}")
    else:
        rep.ok(f"覆盖 metadata 全部 {len(meta_files)} 个文件名")
    bad_key = [k for k in labels if check_clip_name(k)]
    if bad_key:
        rep.fail(f"labels.json 存在 {len(bad_key)} 个非法键名: {bad_key[:3]}")
    counts = Counter(labels.values())
    rep.ok("标签统计: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts, key=lambda k: (-counts[k], k))))
    return bad_value or missing or bad_key


def check_full_fields(rep, full_rows):
    """校验项 3+4: source 合法, 0<=start<end, 文件名符合内容寻址格式且与 ep 列一致."""
    bad_source = sorted({r["source"] for r in full_rows if r["source"] not in ms.SOURCES})
    if bad_source:
        rep.fail(f"存在非法 source {bad_source} (合法值: {list(ms.SOURCES)})")
    else:
        src_counts = Counter(r["source"] for r in full_rows)
        rep.ok("source 全部合法: " + ", ".join(f"{k}={src_counts[k]}" for k in sorted(src_counts)))
    bad_time = [(r["file"], r["start"], r["end"]) for r in full_rows
                if not (r["start"] >= 0.0 and r["end"] > r["start"])]
    if bad_time:
        rep.fail(f"start/end 数值非法 {len(bad_time)} 行 (须 0 <= start < end): {bad_time[:3]}")
    else:
        rep.ok(f"start/end 数值全部合法 (0 <= start < end, {len(full_rows)} 行)")
    bad_name = [(r["file"], p) for r in full_rows for p in check_clip_name(r["file"])]
    prefix_bad = [r["file"] for r in full_rows
                  if not check_clip_name(r["file"]) and not r["file"].startswith(r["ep"] + "_")]
    if bad_name or prefix_bad:
        examples = [b[0] for b in bad_name[:2]] + prefix_bad[:2]
        rep.fail(f"文件名不符合内容寻址格式 {len(bad_name)} 个, 与 ep 列不一致 {len(prefix_bad)} 个:"
                 f" {examples}")
    else:
        rep.ok(f"文件名全部符合内容寻址格式且与 ep 列一致 ({len(full_rows)} 行)")
    return bad_source or bad_time or bad_name or prefix_bad


def _transcript_row_problems(r):
    """单条 transcripts 行字段问题列表 (空 = 合法)."""
    problems = []
    ep = r["ep"]
    if not EP_DIR_RE.match(ep) or ms.ep_dir_name(ep_label_of(ep)) != ep:
        problems.append(f"ep 非法: {ep!r}")
    try:
        s, e, d = float(r["start"]), float(r["end"]), float(r["dur"])
        if s < 0.0 or e <= s:
            problems.append(f"start/end 非法: {r['start']!r},{r['end']!r}")
        elif abs(d - (e - s)) > 0.011:
            problems.append(f"dur 与 end-start 不符: {r['dur']!r} vs {round(e - s, 2)!r}")
    except ValueError:
        problems.append(f"start/end/dur 非数值: {r['start']!r},{r['end']!r},{r['dur']!r}")
    if not r["text"].strip():
        problems.append("text 为空")
    if r["chi_prob"] != "":
        try:
            p = float(r["chi_prob"])
            if not 0.0 <= p <= 1.0:
                problems.append(f"chi_prob 越界: {r['chi_prob']!r}")
        except ValueError:
            problems.append(f"chi_prob 非数值: {r['chi_prob']!r}")
    if r["in_dataset"] not in ("0", "1"):
        problems.append(f"in_dataset 非法: {r['in_dataset']!r}")
    return problems


def check_transcripts(rep, t_rows, full_rows):
    """校验项 5: 行数、覆盖 24 话、列结构、in_dataset 与 metadata 按区间重叠互查."""
    rep.ok(f"共 {len(t_rows)} 行")
    bad = [(r.get("_lineno", "?"), p) for r in t_rows for p in _transcript_row_problems(r)]
    if bad:
        rep.fail(f"{len(bad)} 个字段问题, 前 3 例: {bad[:3]}")
    else:
        rep.ok("各行 ep/start/end/dur/text/chi_prob/in_dataset 字段全部合法")
    eps = sorted({r["ep"] for r in t_rows})
    if eps == EXPECTED_EPS:
        rep.ok("集数覆盖 = ep01-ep24 共 24 话 (无 8.5/16.5/24.5 总集篇)")
    else:
        rep.fail(f"集数覆盖不符: 实测 {len(eps)} 个 {eps}, 应为 ep01-ep24 (跳过总集篇)")
    meta_eps = {r["ep"] for r in full_rows}
    stray = sorted(meta_eps - set(eps))
    if stray:
        rep.fail(f"metadata_full 的集数在 transcripts 中缺失: {stray}")
    else:
        rep.ok(f"metadata_full 的 {len(meta_eps)} 个集数全部出现在 transcripts")
    # 互查口径与 build_transcript_index.py 一致: 区间严格重叠 (s < xe and e > xs)
    def _f(r, keys):
        try:
            return float(r[keys[0]]), float(r[keys[1]])
        except (KeyError, TypeError, ValueError):
            return None
    in1 = {}
    for r in t_rows:
        if r["in_dataset"] == "1":
            se = _f(r, ("start", "end"))
            if se:
                in1.setdefault(r["ep"], []).append(se)
    clips = {}
    for r in full_rows:
        se = _f(r, ("start", "end"))
        if se:
            clips.setdefault(r["ep"], []).append((r["file"], se[0], se[1]))
    clip_no_row = [f for ep, cs in clips.items() for f, s, e in cs
                   if not any(s < xe and e > xs for xs, xe in in1.get(ep, []))]
    row_no_clip = [(ep, xs, xe) for ep, rows in in1.items() for xs, xe in rows
                   if not any(s < xe and e > xs for _, s, e in clips.get(ep, []))]
    if clip_no_row:
        rep.fail(f"{len(clip_no_row)} 个 metadata 片段在 in_dataset=1 的台词行中无重叠: {clip_no_row[:3]}")
    else:
        rep.ok(f"全部 {sum(len(v) for v in clips.values())} 个片段均与某条 in_dataset=1 台词行重叠")
    if row_no_clip:
        rep.fail(f"{len(row_no_clip)} 条 in_dataset=1 的台词行不与任何片段重叠: {row_no_clip[:3]}")
    else:
        rep.ok(f"全部 {sum(len(v) for v in in1.values())} 条 in_dataset=1 台词行均与某个片段重叠")
    return bool(bad or eps != EXPECTED_EPS or stray or clip_no_row or row_no_clip)


def extract_readme_declarations(text):
    """从 README 提取声明值: {'clips': int|None, 'duration_min': float|None, 'labels': int|None, 'transcripts': int|None}."""
    decl = {}
    for key, pat in _DECL_PATTERNS.items():
        m = pat.search(text)
        decl[key] = None if m is None else (float(m.group(1)) if key == "duration_min" else int(m.group(1)))
    return decl


def check_readme(rep, decl, measured):
    """校验项 6: 实测值与 README 声明比对, 不一致仅警告 (数据增长, README 可能滞后)."""
    for key, label in (("clips", "片段数量"), ("duration_min", "总时长 (分钟)"),
                       ("labels", "人工标签总数"), ("transcripts", "transcripts 行数")):
        actual = measured.get(key)
        stated = decl.get(key)
        if actual is None:
            rep.warn(f"{label}: 无法从数据实测, 跳过比对")
        elif stated is None:
            rep.warn(f"{label}: README 未声明 (实测 {actual}), 建议补充声明")
        elif key == "duration_min":
            if abs(actual - stated) <= DURATION_TOL_MIN:
                rep.ok(f"{label}: 实测 {actual:.2f} ≈ README 声明 {stated} (容差 ±{DURATION_TOL_MIN})")
            else:
                rep.warn(f"{label}: 实测 {actual:.2f} 与 README 声明 {stated} 不符 (容差 ±{DURATION_TOL_MIN}),"
                         " README 可能滞后, 请更新")
        elif actual == stated:
            rep.ok(f"{label}: 实测 {actual} == README 声明 {stated}")
        else:
            rep.warn(f"{label}: 实测 {actual} != README 声明 {stated}, README 可能滞后, 请更新")


def _load_json_object(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"应为 JSON object, 实为 {type(data).__name__}")
    return data


def _load_transcripts(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("空文件")
        if [h.strip() for h in header] != TRANSCRIPT_FIELDS:
            raise ValueError(f"表头 {header} 应为 {TRANSCRIPT_FIELDS}")
        rows = []
        for lineno, vals in enumerate(reader, start=2):
            if not vals or all(v == "" for v in vals):
                continue
            row = dict(zip(TRANSCRIPT_FIELDS, vals))
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def validate_repo(root=_REPO_ROOT, stream=None):
    """对仓库根目录运行全部校验, 返回 Reporter (rep.passed 即通过)."""
    root = os.path.abspath(root)
    rep = Reporter(stream)
    dataset = os.path.join(root, "dataset")
    meta_path = os.path.join(dataset, "metadata.csv")
    full_path = os.path.join(dataset, "metadata_full.csv")
    labels_path = os.path.join(root, "annotations", "labels.json")
    transcripts_path = os.path.join(dataset, "transcripts.csv")
    readme_path = os.path.join(root, "README.md")

    pairs, full_rows = None, None
    rep.section("1. metadata.csv <-> metadata_full.csv")
    try:
        pairs = ms.read_metadata(meta_path)
    except (OSError, ms.MetadataRowError) as e:
        rep.fail(f"无法读取 {os.path.relpath(meta_path, root)}: {e}")
    try:
        full_rows = ms.read_metadata_full(full_path)
    except (OSError, ms.MetadataRowError) as e:
        rep.fail(f"无法读取 {os.path.relpath(full_path, root)}: {e}")
    if pairs is not None and full_rows is not None:
        check_metadata_pair(rep, pairs, full_rows)

    meta_files = {name for name, _ in pairs} if pairs else set()
    meta_rows = full_rows or []

    rep.section("2. annotations/labels.json")
    labels = None
    try:
        labels = _load_json_object(labels_path)
    except (OSError, ValueError) as e:
        rep.fail(f"无法读取 {os.path.relpath(labels_path, root)}: {e}")
    if labels is not None:
        check_labels(rep, labels, meta_files)

    rep.section("3. metadata_full.csv 字段 (source / 数值 / 命名)")
    if full_rows:
        check_full_fields(rep, full_rows)

    rep.section("4. transcripts.csv")
    t_rows = None
    try:
        t_rows = _load_transcripts(transcripts_path)
    except (OSError, ValueError) as e:
        rep.fail(f"无法读取 {os.path.relpath(transcripts_path, root)}: {e}")
    if t_rows is not None and full_rows:
        check_transcripts(rep, t_rows, full_rows)

    rep.section("5. README 声明核对 (不符仅警告)")
    measured = {}
    if full_rows:
        measured["clips"] = len(full_rows)
        measured["duration_min"] = round(sum(r["end"] - r["start"] for r in full_rows) / 60.0, 2)
    if labels is not None:
        measured["labels"] = len(labels)
    if t_rows is not None:
        measured["transcripts"] = len(t_rows)
    try:
        with open(readme_path, encoding="utf-8") as f:
            decl = extract_readme_declarations(f.read())
    except OSError as e:
        rep.warn(f"无法读取 README: {e}")
        decl = {k: None for k in _DECL_PATTERNS}
    check_readme(rep, decl, measured)

    rep.section("汇总")
    rep.ok(f"实测: 片段 {measured.get('clips', '?')} 段,"
           f" 总时长 {measured.get('duration_min', '?')} 分钟,"
           f" 标签 {measured.get('labels', '?')} 条,"
           f" transcripts {measured.get('transcripts', '?')} 行")
    if rep.stream:
        if rep.passed:
            print(f"\n结果: 通过 ({rep.warns} 个警告)", file=rep.stream)
        else:
            print(f"\n结果: 失败 ({rep.fails} 个失败, {rep.warns} 个警告)", file=rep.stream)
    return rep


def main(argv=None):
    parser = argparse.ArgumentParser(description="数据集一致性校验 (纯标准库)")
    parser.add_argument("root", nargs="?", default=_REPO_ROOT,
                        help="仓库根目录 (默认: 本脚本所在的仓库根)")
    args = parser.parse_args(argv)
    rep = validate_repo(args.root, stream=sys.stdout)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
