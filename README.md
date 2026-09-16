---
license: cc-by-nc-sa-4.0
language:
- ja
- zh
tags:
- audio
- speech
- tts
- voice-cloning
- anime
task_categories:
- text-to-speech
pretty_name: Chobits-Chii-Voice
---

# Chobits-Chii-Voice

[![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4.svg)](https://madewithlove.org.in)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Chobits--Chii--Voice-yellow)](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice)
[![GitHub](https://img.shields.io/badge/GitHub-Chobits--Chii--Voice-181717?logo=github)](https://github.com/Anime2Real/Chobits-Chii-Voice)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Language: Japanese](https://img.shields.io/badge/Language-Japanese-green.svg)]()
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)]()

> 💖 如果这个项目对你有帮助，欢迎在 [GitHub](https://github.com/Anime2Real/Chobits-Chii-Voice) 点个 Star、在 [Hugging Face](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice) 点个 Like —— 你的支持能让更多人发现小叽！

《人形电脑天使心》(Chobits) 中 **小叽 (Chi / ちぃ)** 角色的语音数据集，可用于语音合成 (TTS)、声音克隆等任务的训练与微调。

本数据集从《Chobits》TV 动画全 24 话（总集篇除外）的音轨中提取角色**小叽**（CV: 田中理惠）的语音片段，经人声分离、语音识别切分、说话人嵌入分类，并经过**两轮人工标注（共 1150 条）+ 最终全量人工复听**筛选而成。

> ⚠️ 注意：原始动画音频的版权归其权利方所有。本数据集仅供学习与研究使用，请勿用于商业用途。

## 数据统计

| 项目 | 数值 |
| --- | --- |
| 片段数量 | 487 |
| 总时长 | 约 21.4 分钟 |
| 片段时长 | 中位 2.24s，平均 2.63s（0.98s – 9.05s） |
| 采样率 | 22050 Hz，单声道，16-bit PCM |
| 语言 | 日语 |
| 覆盖范围 | TV 全 24 话（跳过总集篇 8.5/16.5/24.5） |

## 数据集结构

```
dataset/
├── wavs/                     # 音频片段, 内容寻址命名: {集数}_{起始秒}s.wav
│   ├── ep01_01042.52s.wav
│   └── ...
├── metadata.csv              # TTS 标注: 文件名|文本
├── metadata_full.csv         # 完整元信息: 文件名,集数,起止时间,分类概率,来源,文本
└── transcripts.csv           # 全集台词索引 (8520 句): 集数,起止时间,文本,小叽概率,是否入选
```

> 📦 **音频文件不随本仓库分发**：`dataset/wavs/` 被 git 忽略，请从 [Hugging Face 数据集页](https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice) 下载 `wavs/` 放到 `dataset/` 下；`metadata.csv`、`metadata_full.csv`、`transcripts.csv` 已随仓库跟踪，可直接使用。

- **标注格式**：`metadata.csv` 每行为 `文件名|文本`（GPT-SoVITS / VITS 等框架常用格式）
- **命名规则**：`ep05_00667.42s.wav` 表示第 5 话、起始时间 667.42 秒，跨处理轮次稳定，便于溯源到原始音轨
- **台词检索**：`transcripts.csv` 覆盖 24 话全部转写句子（OP/ED 已剔除），`grep 'ちい' dataset/transcripts.csv` 即可定位"哪一集第几秒说过某句话"；`chi_prob` 列为小叽分类概率（粗略区分说话人），`in_dataset` 标记该句是否有片段入选

## 使用方法

```python
import csv
import soundfile as sf

with open("dataset/metadata.csv", encoding="utf-8") as f:
    for line in f:
        name, text = line.strip().split("|", 1)
        audio, sr = sf.read(f"dataset/wavs/{name}.wav")  # sr = 22050
        # ...
```

## 数据处理流程

全部代码见 `pipeline/` 目录，流程如下：

1. **音轨提取**：从视频提取音轨（`batch.py`，ffmpeg）
2. **人声分离**：Demucs (htdemucs) 分离人声与 BGM/音效
3. **语音识别**：Whisper (mlx-whisper, large-v3-turbo) 转写日语台词并按句切分，利用 B 站官方 OP/ED 跳过时间戳剔除片头片尾；长句用词级时间戳二次切分（1–10s）
4. **静音过滤**：RMS/峰值阈值剔除静音与近静音片段
5. **说话人分类**：ECAPA-TDNN (SpeechBrain, VoxCeleb) 提取说话人嵌入，用人工标注训练逻辑回归分类器（`train_classifier.py`）判别是否为小叽
6. **人工标注**：两轮交互式标注（`label_ui.py` 浏览器工具，快捷键打标），共 1150 条，覆盖分类器边界区与低分区抽查；最终导出前对全部收录片段人工复听一遍
7. **导出**：能量谷修剪切边、统一转 22050Hz（`finalize.py`）

数据一致性校验：`python tools/validate.py`（CI 已接入）会对 `dataset/` 与 `annotations/labels.json` 做结构断言（两表文件集合/行数一致、标签覆盖与合法性、字段与内容寻址命名、transcripts 24 话覆盖与 `in_dataset` 互查，失败即非零退出），并将实测片段数/总时长/标签数与 README 声明比对（不符仅警告）。

### 纯度说明

- 最终 487 段**全部经过人工听辨确认**为小叽单人语音，不含其他角色、混合人声或明显噪声段
- 分类器仅用于挑选候选片段（精确率为历史轮次 5 折交叉验证实测约 91%；当前 `train_classifier.py` 以精确率 ≥0.95 为目标选阈值，运行时会打印实测精确率/召回率）；最终纯度由两轮人工标注 + 全量复听保证，不依赖分类器兜底

## 仓库结构

```
Chobits-Chii-Voice/
├── README.md               # 本文件 (数据集卡片)
├── requirements.txt        # pipeline 依赖锁定 (Python 3.12, 仅 macOS Apple Silicon)
├── tests/                  # metadata_schema / validate 单元测试 (纯标准库 unittest)
├── tools/                  # 数据一致性校验 validate.py (纯标准库, CI 接入)
├── dataset/                # 最终数据集 (见上)
├── annotations/            # 人工标注资产
│   ├── labels.json             # 1150 条人工标签 {文件名: chi/not_chi/mixed/bad/unsure}
│   ├── clips.json              # 第 2 轮标注清单 (片段元信息)
│   ├── human_labels.npz        # 标注片段的说话人嵌入
│   ├── labeled_embeddings.npz  # 分类器正/负样本池
│   ├── chi_lr.pkl              # 训练好的逻辑回归分类器
│   ├── chi_reference_v5.npy    # 小叽参考向量 (确认样本均值)
│   └── legacy/                 # 历史参考向量与中间统计
├── pipeline/               # 处理流水线代码
│   ├── batch.py                # 抽轨 / 人声分离 / 转写 / 候选切分 (幂等, 产物在 build/, 不写 dataset/)
│   ├── common.py               # chunk 生成与嵌入缓存
│   ├── audio_utils.py          # 静音判定 / 词级拆分 / 能量谷切分与修剪
│   ├── prepare_labeling.py     # 按分类器概率挑选待标注片段 (清单入 annotations/, wav 入 build/)
│   ├── label_ui.py             # 浏览器标注工具 (快捷键, 实时落盘)
│   ├── embed_human_labels.py   # 标注片段嵌入
│   ├── train_classifier.py     # 训练分类器 + 交叉验证选阈值
│   ├── finalize.py             # 最终导出 (dataset/ 的唯一写入入口)
│   ├── build_transcript_index.py  # 生成全集台词索引到 build/ (finalize 发布到 dataset/)
│   └── legacy/                 # 历史迭代脚本 (round2-round8 等, 仅供考古)
├── Chobits_Movie.dvc       # 原始视频 (DVC, 存储于腾讯云 COS)
└── build/                  # 中间产物缓存 (不入库, 可由 pipeline 重新生成)
```

### 复现/增量处理

> ⚠️ 流水线**仅支持 macOS (Apple Silicon)**：Whisper 转写依赖 MLX（仅提供 arm64 wheel），
> Demucs 默认用 `mps` 设备。ffmpeg 静态二进制由 `imageio-ffmpeg` 提供，也可自备 ffmpeg（加入 `PATH` 即可）。

```bash
# 0. 环境: Python 3.12 + 依赖锁定 (见 requirements.txt)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. 原始视频 (需配置 COS 凭据, 见 .dvc/config.local; COS 要求 virtual-host 寻址, 通过 .dvc/aws_config 指定)
AWS_CONFIG_FILE=$PWD/.dvc/aws_config dvc pull

# 2. 抽轨 + 人声分离 + 转写 + 候选切分 (幂等, 增量处理新剧集)
#    产物全部写 build/ 暂存区 (含 build/metadata_full.csv 候选总表、build/epXX/kept.json),
#    不触碰 dataset/; demucs 设备可用 CHII_VOICE_DEVICE 覆盖 (默认 mps)
.venv/bin/python pipeline/batch.py

# 3. 人工标注循环 (annotations/labels.json 与 clips.json 已随仓库跟踪,
#    已标注片段自动跳过, 可随时中断续标)
.venv/bin/python pipeline/prepare_labeling.py   # 生成待标注清单 (B/C 组 wav 在 build/labeling/clips/)
.venv/bin/python pipeline/label_ui.py           # 浏览器标注, 标签实时落盘 annotations/labels.json
.venv/bin/python pipeline/embed_human_labels.py # 嵌入新标注
.venv/bin/python pipeline/train_classifier.py   # 重训分类器

# 4. 全集台词索引 -> build/transcripts.csv
.venv/bin/python pipeline/build_transcript_index.py

# 5. 最终导出: 读 annotations/labels.json + build/ 中间产物,
#    原子写出 dataset/wavs/ + metadata.csv + metadata_full.csv (+ transcripts.csv)
#    dataset/ 下正式文件只由本步产出
.venv/bin/python pipeline/finalize.py
```

## 许可协议

本数据集以 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)（署名-非商业性使用-相同方式共享）协议发布：

- **署名 (BY)**：使用时须注明数据集来源。
- **非商业性使用 (NC)**：不得将本数据集用于商业目的。
- **相同方式共享 (SA)**：基于本数据集的衍生作品须以相同协议发布。

## 免责声明

- 本数据集仅用于学术研究与个人学习，不构成对原作品版权的任何主张。
- 使用本数据集训练的模型，其生成内容不得用于侵犯原作品及相关声优权益的用途。
- 若权利方提出要求，本数据集将被下架。

## 引用

如果您在研究中使用了本数据集，请引用：

```bibtex
@misc{chobits-chii-voice,
  author = {chenxin199305},
  title = {Chobits-Chii-Voice: Voice Dataset of Chi from Chobits},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/chenxin199305/Chobits-Chii-Voice}}
}
```
