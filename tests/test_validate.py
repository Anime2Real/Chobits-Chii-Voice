#!/usr/bin/env python3
"""tools/validate.py 的单元测试 (纯标准库, 不依赖第三方包).

用 tempfile 构造小型假数据集 (3 段片段 / 24 话各 1 行 transcripts),
只测核心校验逻辑, 不跑真实数据集全量 (validate.py 默认跑真实数据).

运行: python -m unittest discover -s tests -v   (仓库根目录)
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import validate  # noqa: E402

# 3 段假片段: (file, ep, start, end, text)
CLIPS = [
    ("ep01_00001.00s", "ep01", 1.0, 3.0, "チー"),
    ("ep02_00010.00s", "ep02", 10.0, 11.5, "チチ"),
    ("ep03_00020.00s", "ep03", 20.0, 21.0, "ちい"),
]

README_TEMPLATE = """\
# Fake Dataset

| 项目 | 数值 |
| --- | --- |
| 片段数量 | {clips} |
| 总时长 | 约 {duration} 分钟 |

两轮人工标注（共 {labels} 条）
└── transcripts.csv           # 全集台词索引 ({transcripts} 句)
└── labels.json               # {labels} 条人工标签
"""


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\r\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\r\n")


def _readme_for(clips, labels_count, transcripts_count):
    total_min = round(sum(c[3] - c[2] for c in clips) / 60.0, 1)
    return README_TEMPLATE.format(clips=len(clips), duration=total_min,
                                  labels=labels_count, transcripts=transcripts_count)


def _make_dataset(root, clips=CLIPS, labels=None, readme=None):
    """构造最小可通过的假数据集, 返回仓库根路径.

    metadata.csv 与 metadata_full.csv 均由 clips 派生, 保证两表天然一致;
    要造不一致, 直接改写写出的文件 (见各测试).
    """
    dataset = os.path.join(root, "dataset")
    os.makedirs(dataset)
    os.makedirs(os.path.join(root, "annotations"))

    # metadata.csv: file|text (TTS 标注格式)
    with open(os.path.join(dataset, "metadata.csv"), "w", encoding="utf-8") as f:
        for name, _, _, _, text in clips:
            f.write(f"{name}|{text}\n")

    # metadata_full.csv: file,ep,start,end,prob,source,text
    _write_csv(os.path.join(dataset, "metadata_full.csv"),
               validate.ms.FIELDS,
               [(name, ep, start, end, "", "human", text)
                for name, ep, start, end, text in clips])

    # labels.json: 覆盖全部片段 + 1 条候选标签 (模拟真实结构)
    if labels is None:
        labels = {name: "chi" for name, _, _, _, _ in clips}
        labels["ep01_00099.00s"] = "not_chi"  # 未入选的候选, 允许存在
    with open(os.path.join(root, "annotations", "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    # transcripts.csv: 24 话各 1 行, 有片段的话 in_dataset=1
    rows = []
    for i in range(1, 25):
        ep = f"ep{i:02d}"
        clip = next((c for c in clips if c[1] == ep), None)
        if clip:
            _, _, start, end, _ = clip
            rows.append((ep, start - 0.5, end + 0.5, end - start + 1.0, "セリフ", "0.9", 1))
        else:
            rows.append((ep, 0.0, 1.0, 1.0, "セリフ", "", 0))
    _write_csv(os.path.join(dataset, "transcripts.csv"),
               validate.TRANSCRIPT_FIELDS, rows)

    if readme is None:
        readme = _readme_for(clips, labels_count=len(labels), transcripts_count=len(rows))
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    return root


class ValidateRepoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = _make_dataset(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_dataset_passes(self):
        rep = validate.validate_repo(self.root)
        self.assertTrue(rep.passed, f"假数据集应通过, 实际 {rep.fails} 失败 {rep.warns} 警告")
        self.assertEqual(rep.fails, 0)
        self.assertEqual(rep.warns, 0)

    def test_valid_dataset_report_prints_sections(self):
        out = io.StringIO()
        rep = validate.validate_repo(self.root, stream=out)
        self.assertTrue(rep.passed)
        report = out.getvalue()
        self.assertIn("[OK]", report)
        self.assertIn("ep01-ep24", report)  # 集数覆盖

    def test_filename_set_mismatch_fails(self):
        with open(os.path.join(self.root, "dataset", "metadata.csv"), "a", encoding="utf-8") as f:
            f.write("ep09_00123.45s|ゴースト\n")  # 只存在于 metadata.csv
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_missing_label_fails(self):
        labels = {name: "chi" for name, _, _, _, _ in CLIPS[1:]}  # 缺第 1 段
        with open(os.path.join(self.root, "annotations", "labels.json"), "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False)
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_invalid_label_value_fails(self):
        path = os.path.join(self.root, "annotations", "labels.json")
        with open(path, encoding="utf-8") as f:
            labels = json.load(f)
        labels[CLIPS[0][0]] = "robot"  # 非法标签值
        with open(path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False)
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_bad_source_and_time_fail(self):
        _write_csv(os.path.join(self.root, "dataset", "metadata_full.csv"),
                   validate.ms.FIELDS,
                   [("ep01_00001.00s", "ep01", 3.0, 3.0, "", "human", "x"),   # end<=start
                    ("ep02_00010.00s", "ep02", -1.0, 0.5, "", "human", "x"),  # start<0
                    ("ep03_00020.00s", "ep03", 20.0, 21.0, "", "alien", "x")])  # 非法 source
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_clip_name_format_fails(self):
        # 文件名集合两表仍一致 (均由 clips 派生); 违规命名会连带触发命名/ep 列一致性/
        # labels 键名/台词覆盖等多个结构断言, 此处只断言整体不通过
        clips = [("ep01_clip001", "ep01", 1.0, 3.0, "x"),              # 非内容寻址命名
                 ("ep02_00010.00s", "ep01", 10.0, 11.5, "x"),          # 与 ep 列不符
                 CLIPS[2]]
        tmp2 = tempfile.mkdtemp()
        try:
            root = _make_dataset(tmp2, clips=clips)
            rep = validate.validate_repo(root)
            self.assertFalse(rep.passed)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_transcripts_wrong_eps_fail(self):
        rows = [("ep01", 0.0, 1.0, 1.0, "x", "", 0) for _ in range(24)]  # 缺 ep02-ep24
        _write_csv(os.path.join(self.root, "dataset", "transcripts.csv"),
                   validate.TRANSCRIPT_FIELDS, rows)
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_transcripts_clip_without_in_dataset_row_fails(self):
        rows = []
        for i in range(1, 25):
            ep = f"ep{i:02d}"
            clip = next((c for c in CLIPS if c[1] == ep), None)
            in_ds = 0 if ep == "ep01" else (1 if clip else 0)  # ep01 片段所在行误标 0
            if clip:
                _, _, s, e, _ = clip
                rows.append((ep, s - 0.5, e + 0.5, e - s + 1.0, "x", "", in_ds))
            else:
                rows.append((ep, 0.0, 1.0, 1.0, "x", "", 0))
        _write_csv(os.path.join(self.root, "dataset", "transcripts.csv"),
                   validate.TRANSCRIPT_FIELDS, rows)
        rep = validate.validate_repo(self.root)
        self.assertFalse(rep.passed)

    def test_readme_mismatch_warns_only(self):
        with open(os.path.join(self.root, "README.md"), "w", encoding="utf-8") as f:
            f.write(_readme_for(CLIPS, labels_count=12345, transcripts_count=777)
                    .replace(f"| 片段数量 | {len(CLIPS)} |", "| 片段数量 | 999 |"))
        rep = validate.validate_repo(self.root)
        self.assertTrue(rep.passed)      # README 声明不符仅警告, 不失败
        self.assertEqual(rep.fails, 0)
        self.assertGreaterEqual(rep.warns, 3)  # clips/labels/transcripts 三项不符

    def test_readme_missing_declaration_warns_only(self):
        with open(os.path.join(self.root, "README.md"), "w", encoding="utf-8") as f:
            f.write("# 无任何声明数字的 README\n")
        rep = validate.validate_repo(self.root)
        self.assertTrue(rep.passed)
        self.assertGreaterEqual(rep.warns, 4)


class ExtractReadmeDeclarationsTest(unittest.TestCase):
    def test_extracts_all(self):
        text = ("| 片段数量 | 487 |\n| 总时长 | 约 21.4 分钟 |\n"
                "labels.json  # 1150 条人工标签\n全集台词索引 (8520 句)")
        decl = validate.extract_readme_declarations(text)
        self.assertEqual(decl, {"clips": 487, "duration_min": 21.4,
                                "labels": 1150, "transcripts": 8520})

    def test_missing_patterns_are_none(self):
        decl = validate.extract_readme_declarations("没有声明")
        self.assertEqual(decl, {"clips": None, "duration_min": None,
                                "labels": None, "transcripts": None})


class CheckClipNameTest(unittest.TestCase):
    def test_canonical_names_pass(self):
        for name in ("ep01_00001.00s", "ep24_99999.99s", "ep08_5_00003.00s"):
            self.assertEqual(validate.check_clip_name(name), [])

    def test_non_canonical_names_fail(self):
        self.assertTrue(validate.check_clip_name("ep01_clip001"))
        self.assertTrue(validate.check_clip_name("ep1_00001.00s"))        # ep 未补零
        self.assertTrue(validate.check_clip_name("ep01_1.00s"))           # 秒未补零
        self.assertTrue(validate.check_clip_name("ep01_00001.00s.wav"))   # 含扩展名
        self.assertTrue(validate.check_clip_name(""))


if __name__ == "__main__":
    unittest.main()
