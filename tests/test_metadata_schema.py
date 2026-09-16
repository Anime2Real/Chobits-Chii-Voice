#!/usr/bin/env python3
"""metadata_schema 的单元测试 (纯标准库, 不依赖第三方包).

运行: python -m unittest discover -s tests -v   (仓库根目录)
"""
import csv
import io
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import metadata_schema as ms  # noqa: E402


class TestNaming(unittest.TestCase):
    def test_ep_dir_name(self):
        self.assertEqual(ms.ep_dir_name("1"), "ep01")
        self.assertEqual(ms.ep_dir_name("24"), "ep24")
        self.assertEqual(ms.ep_dir_name("8.5"), "ep08_5")
        self.assertEqual(ms.ep_dir_name("16.5"), "ep16_5")

    def test_clip_name(self):
        # 与 README 及已发布 dataset/metadata.csv 的命名一致 (内容寻址)
        self.assertEqual(ms.clip_name("ep05", 667.42), "ep05_00667.42s")
        self.assertEqual(ms.clip_name("ep01", 65.88), "ep01_00065.88s")
        self.assertEqual(ms.clip_name("ep08_5", 3.0), "ep08_5_00003.00s")


class TestMakeRow(unittest.TestCase):
    def test_normalization(self):
        r = ms.make_row(file="ep01_00065.88s", ep="ep01", start=65.876, end=67.9,
                        prob=0.60001, source=ms.SOURCE_CANDIDATE, text="チー")
        self.assertEqual(r["start"], 65.88)
        self.assertEqual(r["end"], 67.9)
        self.assertEqual(r["prob"], 0.6)

    def test_empty_prob_for_human(self):
        r = ms.make_row(file="f", ep="ep01", start=1, end=2, prob="",
                        source=ms.SOURCE_HUMAN, text="x")
        self.assertEqual(r["prob"], "")

    def test_unknown_source_rejected(self):
        with self.assertRaises(ms.MetadataRowError):
            ms.make_row(file="f", ep="ep01", start=1, end=2, source="bogus", text="x")


class TestMetadataFullRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "metadata_full.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_finalize_style_rows(self):
        # finalize.py 产出的两类行: 人工确认 (prob 空) 与自动收录 (prob 数值)
        rows = [
            ms.make_row(file="ep01_01042.52s", ep="ep01", start=1042.52, end=1043.52,
                        prob="", source=ms.SOURCE_HUMAN, text="チー"),
            ms.make_row(file="ep02_00667.42s", ep="ep02", start=667.42, end=669.88,
                        prob=0.973, source=ms.SOURCE_AUTO, text="チーン, 「テスト」"),
        ]
        ms.write_metadata_full(self.path, rows)
        back = ms.read_metadata_full(self.path)
        self.assertEqual(back, rows)
        # prepare_labeling.py 消费所需字段: file 与可 float 化的 prob
        self.assertEqual(back[0]["file"], "ep01_01042.52s")
        self.assertEqual(back[0]["prob"], "")
        self.assertEqual(float(back[1]["prob"]), 0.973)

    def test_batch_candidate_rows(self):
        # batch.py 中间产物: source=candidate/review, prob=相似度
        rows = [ms.make_row(file="ep05_00667.42s", ep="ep05", start=667.42, end=669.0,
                            prob=0.512345, source=ms.SOURCE_REVIEW, text="ちい?")]
        ms.write_metadata_full(self.path, rows)
        back = ms.read_metadata_full(self.path)
        self.assertEqual(back[0]["source"], "review")
        self.assertAlmostEqual(float(back[0]["prob"]), 0.512)

    def test_header_written(self):
        ms.write_metadata_full(self.path, [])
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.readline().strip(), ",".join(ms.FIELDS))

    def test_strict_raises_on_bad_row(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(ms.FIELDS)
            w.writerow(["ep01_00001.00s", "ep01", "abc", "2.0", "", "human", "x"])
        with self.assertRaises(ms.MetadataRowError):
            ms.read_metadata_full(self.path, strict=True)

    def test_skip_mode_tolerates_bad_rows(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(ms.FIELDS)
            w.writerow(["ep01_00001.00s", "ep01", "1.0", "2.0", "0.5", "human", "ok"])
            w.writerow(["", "", "", "", "", "", ""])           # 空行
            w.writerow(["ep01_00003.00s", "ep01", "5.0", "4.0", "", "human", "end<=start"])
            w.writerow(["ep01_00004.00s", "ep01", "xx", "9.0", "", "human", "bad start"])
            w.writerow(["ep01_00005.00s", "ep01", "9.0", "10.0", "zz", "human", "bad prob"])
            w.writerow(["ep01_00006.00s", "ep01", "11.0", "12.0", "0.1", "alien", "bad source"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = ms.read_metadata_full(self.path, strict=False)
        self.assertEqual([r["file"] for r in rows], ["ep01_00001.00s"])
        self.assertEqual(len(caught), 5)


class TestMetadataPairsRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "metadata.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        pairs = [("ep01_01042.52s", "チー"), ("ep02_00166.40s", "a|b")]  # 文本可含 '|'
        ms.write_metadata(self.path, pairs)
        self.assertEqual(ms.read_metadata(self.path), pairs)

    def test_crlf_input_tolerated(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write("ep01_00001.00s|チー\r\nep01_00002.00s|チチ!\r\n")
        self.assertEqual(ms.read_metadata(self.path),
                         [("ep01_00001.00s", "チー"), ("ep01_00002.00s", "チチ!")])

    def test_bad_line_rejected(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("no-delimiter-here\n")
        with self.assertRaises(ms.MetadataRowError):
            ms.read_metadata(self.path)

    def test_pipe_in_filename_rejected(self):
        with self.assertRaises(ms.MetadataRowError):
            ms.write_metadata(self.path, [("a|b", "x")])


class TestAtomicWrite(unittest.TestCase):
    def test_replace_and_no_leftover(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("old")
            ms.atomic_write_text(path, "new")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "new")
            leftovers = [n for n in os.listdir(d) if n.startswith(".tmp-")]
            self.assertEqual(leftovers, [])

    def test_newline_not_translated(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            ms.atomic_write_text(path, "a\nb\r\nc")
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"a\nb\r\nc")


if __name__ == "__main__":
    unittest.main()
