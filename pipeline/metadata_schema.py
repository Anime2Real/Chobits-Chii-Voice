#!/usr/bin/env python3
"""metadata.csv / metadata_full.csv 的格式契约 (唯一事实源).

纯标准库实现, 不依赖 torch/demucs 等重依赖, 流水线各模块与 tests 均可安全 import.

metadata_full.csv 为逗号分隔表, 表头固定为 FIELDS:
  file   内容寻址片段名 (不含扩展名): {ep}_{起始秒:08.2f}s, 如 ep05_00667.42s
  ep     集数目录名 (与 build/ 下目录一致): ep01 / ep08_5
  start/end  片段相对该集音轨的起止秒 (两位小数)
  prob   分类概率或相似度; 人工确认行为空字符串
  source human=人工标注确认  auto=高置信自动收录
         candidate=batch 候选  review=待人工复核 (后两者为 build/ 中间产物)
  text   转写文本

metadata.csv 为 TTS 标注格式, 每行 `file|text` (GPT-SoVITS / VITS 常用).

所有写出均走"临时文件 + os.replace"原子切换, 避免半截文件毁掉已发布数据.
"""
import csv
import io
import os
import tempfile
import warnings

FIELDS = ["file", "ep", "start", "end", "prob", "source", "text"]

SOURCE_HUMAN = "human"
SOURCE_AUTO = "auto"
SOURCE_CANDIDATE = "candidate"
SOURCE_REVIEW = "review"
SOURCES = (SOURCE_HUMAN, SOURCE_AUTO, SOURCE_CANDIDATE, SOURCE_REVIEW)


class MetadataRowError(ValueError):
    """metadata_full.csv 行字段缺失或数值非法."""


def ep_dir_name(label):
    """集数标签 -> 目录名: '1' -> ep01, '8.5' -> ep08_5."""
    if "." in label:
        head, tail = label.split(".")
        return f"ep{int(head):02d}_{tail}"
    return f"ep{int(label):02d}"


def clip_name(ep, start):
    """内容寻址片段名: ('ep05', 667.42) -> 'ep05_00667.42s' (ep 须为 ep_dir_name 形式)."""
    return f"{ep}_{start:08.2f}s"


def make_row(file, ep, start, end, prob="", source="", text=""):
    """构造一行 dict, 统一数值规整: start/end 两位小数, prob 三位小数或空串."""
    row = {"file": str(file), "ep": str(ep),
           "start": round(float(start), 2), "end": round(float(end), 2),
           "prob": "" if prob in ("", None) else round(float(prob), 3),
           "source": str(source), "text": str(text)}
    if row["source"] not in SOURCES:
        raise MetadataRowError(f"未知 source: {row['source']!r} (应为 {SOURCES})")
    return row


def parse_row(raw):
    """校验并规整原始行 dict; 异常时抛 MetadataRowError."""
    if raw is None:
        raise MetadataRowError("空行")
    row = {}
    for k in FIELDS:
        v = raw.get(k)
        row[k] = "" if v is None else str(v).strip()
    if not row["file"] or not row["ep"]:
        raise MetadataRowError(f"file/ep 为空: {raw!r}")
    try:
        row["start"] = float(row["start"])
        row["end"] = float(row["end"])
    except ValueError:
        raise MetadataRowError(f"start/end 非数值: {raw!r}") from None
    if row["end"] <= row["start"]:
        raise MetadataRowError(f"end<=start: {raw!r}")
    if row["prob"] != "":
        try:
            row["prob"] = float(row["prob"])
        except ValueError:
            raise MetadataRowError(f"prob 非数值: {raw!r}") from None
    if row["source"] not in SOURCES:
        raise MetadataRowError(f"未知 source: {row['source']!r}")
    return row


def read_metadata_full(path, strict=True):
    """读 metadata_full.csv, 返回按文件序的 row dict 列表.

    strict=True 时首行异常即抛 MetadataRowError; 否则跳过异常行并 warnings.warn.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for lineno, raw in enumerate(csv.DictReader(f), start=2):
            try:
                rows.append(parse_row(raw))
            except MetadataRowError as e:
                if strict:
                    raise MetadataRowError(f"{path}:{lineno}: {e}") from None
                warnings.warn(f"{path}:{lineno}: 跳过异常行: {e}")
    return rows


def write_metadata_full(path, rows):
    """原子写出 metadata_full.csv (CRLF 行尾, 与已发布文件一致)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in FIELDS})
    atomic_write_text(path, buf.getvalue())


def read_metadata(path):
    """读 metadata.csv (`file|text` 每行), 返回 [(file, text), ...]; 容忍 CRLF/LF."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if "|" not in line:
                raise MetadataRowError(f"{path}:{lineno}: 缺少 '|' 分隔: {line!r}")
            name, text = line.split("|", 1)
            if not name:
                raise MetadataRowError(f"{path}:{lineno}: 文件名为空")
            pairs.append((name, text))
    return pairs


def write_metadata(path, pairs):
    """原子写出 metadata.csv (LF 行尾; file 不得含 '|'/换行, 否则无法无损读回)."""
    lines = []
    for name, text in pairs:
        name, text = str(name), str(text)
        if "|" in name or "\n" in name or "\r" in name:
            raise MetadataRowError(f"文件名含非法字符: {name!r}")
        lines.append(f"{name}|{text}")
    atomic_write_text(path, "".join(line + "\n" for line in lines))


def atomic_write_text(path, text, encoding="utf-8"):
    """写文本到临时文件再 os.replace 原子切换; 行尾不做任何转换."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
