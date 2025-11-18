#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 MusicXML / MXL 乐谱复杂度评估脚本（修正版）

功能：
- 扫描单个文件或一个目录下的所有 .mxl / .musicxml / .xml
- 解析乐谱结构，提取若干“复杂度”相关特征
- 计算一个综合复杂度分数 complexity_score
- 将所有结果写入 CSV，方便后续筛选训练数据

用法示例：
    python assess_musicxml_complexity.py path/to/your/xml_or_folder -o scores.csv

依赖：
- 只用到 Python 标准库（xml, zipfile, csv 等），不需要额外安装第三方包。
"""

import sys
import math
import csv
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET


def strip_namespaces(elem):
    """
    去掉 XML namespace，方便用简单的 tag 名称（比如 'note'）来查找。
    """
    for el in elem.iter():
        if isinstance(el.tag, str) and '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]


def load_xml_root(path: Path):
    """
    读取 .musicxml / .xml / .mxl 文件，返回去掉 namespace 之后的 XML 根节点。

    修复点：
    - 对 .mxl，优先解析 META-INF/container.xml，找到真正的乐谱文件 full-path（如 score.xml）
    - 如果没有 container.xml，再退化为在 zip 中挑最可能是乐谱的 xml 文件
    """
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path, "r") as zf:
            xml_bytes = None

            # 1) 尝试 MusicXML 容器规范：META-INF/container.xml
            if "META-INF/container.xml" in zf.namelist():
                try:
                    cont = ET.fromstring(zf.read("META-INF/container.xml"))
                    strip_namespaces(cont)
                    rootfile = cont.find(".//rootfile")
                    if rootfile is not None and rootfile.get("full-path"):
                        target = rootfile.get("full-path")
                        if target in zf.namelist():
                            xml_bytes = zf.read(target)
                except Exception:
                    # 如果 container 解析失败，后面再用 fallback
                    xml_bytes = None

            # 2) fallback：从 zip 里挑一个最像乐谱的 xml 文件
            if xml_bytes is None:
                # 优先挑：不在 META-INF 里的 xml/musicxml
                candidates = [
                    name for name in zf.namelist()
                    if name.lower().endswith((".xml", ".musicxml"))
                    and not name.upper().startswith("META-INF/")
                ]
                if not candidates:
                    # 实在没有就全局找 xml/musicxml
                    candidates = [
                        name for name in zf.namelist()
                        if name.lower().endswith((".xml", ".musicxml"))
                    ]
                if not candidates:
                    raise ValueError("No .xml or .musicxml file found inside MXL archive.")

                # 简单策略：先读第一个；如有需要可以再加“score-partwise”等判断
                xml_bytes = zf.read(candidates[0])

            root = ET.fromstring(xml_bytes)

    else:
        tree = ET.parse(path)
        root = tree.getroot()

    strip_namespaces(root)
    return root


@dataclass
class ScoreFeatures:
    filename: str
    total_measures: int = 0
    total_notes: int = 0
    total_rests: int = 0
    unique_durations: int = 0
    short_note_ratio: float = 0.0
    chord_ratio: float = 0.0
    avg_voices_per_staff: float = 0.0
    dynamics_count: int = 0
    articulations_count: int = 0
    ornaments_count: int = 0
    slur_count: int = 0
    tie_count: int = 0
    accidental_count: int = 0
    grace_note_count: int = 0
    key_change_count: int = 0
    time_change_count: int = 0
    has_lyrics: int = 0
    lyric_count: int = 0
    staff_count: int = 0
    part_count: int = 0
    complexity_score: float = 0.0

    def compute_complexity_score(self) -> float:
        """
        尝试让“复杂度”更多地来自结构/密度，而不是单纯长度。
        建议在调用脚本时，先用 --min-notes 做一次长度过滤。
        """
        if self.total_notes <= 0 or self.total_measures <= 0:
            return 0.0

        # ---- 一些派生指标 ----
        # 每小节音符密度
        avg_notes_per_measure = self.total_notes / self.total_measures

        # 基础复杂度（尽量与长度弱相关）
        base = 0.0

        # 节奏丰富度：不同时值 + 短时值比例
        base += 1.2 * self.unique_durations          # 种类多一点加分
        base += 2.0 * self.short_note_ratio          # 短时值多→节奏更碎

        # 和声/多声部
        base += 2.0 * self.chord_ratio               # 和弦比例
        base += 1.0 * self.avg_voices_per_staff      # 每个谱表有多少个 voice

        # 记号丰富度：力度 + articulations + ornaments
        mark_sum = self.dynamics_count + self.articulations_count + self.ornaments_count
        base += 0.7 * math.log(mark_sum + 1, 2)

        # 连音线/延音线
        base += 0.5 * math.log(self.slur_count + self.tie_count + 1, 2)

        # 升降号
        base += 0.4 * math.log(self.accidental_count + 1, 2)

        # 调号 / 拍号变化（通常和长度相关，但数量本身也说明结构复杂）
        base += 0.3 * (self.key_change_count + self.time_change_count)

        # 歌词
        if self.has_lyrics:
            base += 1.0

        # 密度项：每小节平均多少音符（对数，避免超大）
        density_term = math.log(avg_notes_per_measure + 1, 2)

        # 长度项：有一定作用，但做饱和，避免“越长越离谱”
        len_term = math.log(self.total_notes + 1, 2)
        len_term = min(len_term, 10.0)  # 设个上限，比如 10

        # 最终分数：密度和结构占主导，长度是轻微加分
        score = base + 0.8 * density_term + 0.3 * len_term

        return round(score, 3)



def analyze_musicxml_file(path: Path) -> ScoreFeatures:
    """
    对单个 MusicXML/MXL 文件进行特征提取与复杂度评估。
    """
    root = load_xml_root(path)
    feats = ScoreFeatures(filename=str(path))

    # part / staff 相关
    parts = root.findall("part")
    feats.part_count = len(parts)

    # 小节数：使用 measure 的 number 属性去重
    measure_numbers = set()
    for measure in root.findall(".//measure"):
        num = measure.get("number")
        if num is not None:
            measure_numbers.add(num)
    if measure_numbers:
        feats.total_measures = len(measure_numbers)
    else:
        feats.total_measures = len(root.findall(".//measure"))

    # 音符相关特征
    duration_types = set()
    short_types = {"eighth", "16th", "32nd", "64th", "128th"}
    short_note_count = 0
    chord_note_count = 0
    staff_voices = defaultdict(set)

    for note in root.findall(".//note"):
        is_rest = note.find("rest") is not None
        if is_rest:
            feats.total_rests += 1
        else:
            feats.total_notes += 1

        # 时值类型
        type_el = note.find("type")
        if type_el is not None and type_el.text:
            t = type_el.text.strip()
            duration_types.add(t)
            if t in short_types and not is_rest:
                short_note_count += 1

        # 和弦（带 <chord/> 的音默认为和弦中的一个）
        if note.find("chord") is not None and not is_rest:
            chord_note_count += 1

        # 装饰音（倚音等）
        if note.find("grace") is not None:
            feats.grace_note_count += 1

        # 延音线（tie）
        ties = note.findall("tie")
        feats.tie_count += len(ties)

        # 升降号
        if note.find("accidental") is not None:
            feats.accidental_count += 1

        # 歌词
        lyrics = note.findall("lyric")
        if lyrics:
            feats.lyric_count += len(lyrics)

        # staff / voice 统计（用于估算多声部情况）
        staff_el = note.find("staff")
        staff_id = staff_el.text.strip() if staff_el is not None and staff_el.text else "1"
        voice_el = note.find("voice")
        voice_id = voice_el.text.strip() if voice_el is not None and voice_el.text else "1"
        staff_voices[staff_id].add(voice_id)

    feats.unique_durations = len(duration_types)
    if feats.total_notes > 0:
        feats.short_note_ratio = short_note_count / feats.total_notes
        feats.chord_ratio = chord_note_count / feats.total_notes
    else:
        feats.short_note_ratio = 0.0
        feats.chord_ratio = 0.0

    feats.staff_count = len(staff_voices) if staff_voices else 0
    if feats.staff_count > 0:
        feats.avg_voices_per_staff = (
            sum(len(v) for v in staff_voices.values()) / feats.staff_count
        )
    else:
        feats.avg_voices_per_staff = 0.0

    feats.has_lyrics = 1 if feats.lyric_count > 0 else 0

    # 表情 / 力度 / 装饰等记号
    feats.dynamics_count = len(root.findall(".//dynamics"))
    feats.articulations_count = len(root.findall(".//articulations"))
    feats.ornaments_count = len(root.findall(".//ornaments"))
    feats.slur_count = len(root.findall(".//slur"))

    # 调号与拍号变化
    last_key = None
    last_time = None
    key_changes = 0
    time_changes = 0

    for part in parts:
        for measure in part.findall("measure"):
            attributes = measure.find("attributes")
            if attributes is None:
                continue

            # 调号
            key_el = attributes.find("key")
            if key_el is not None:
                fifths = key_el.findtext("fifths", default="0")
                mode = key_el.findtext("mode", default="")
                key_repr = f"{fifths}|{mode}"
                if key_repr != last_key:
                    if last_key is not None:
                        key_changes += 1
                    last_key = key_repr

            # 拍号
            time_el = attributes.find("time")
            if time_el is not None:
                beats = time_el.findtext("beats", default="")
                beat_type = time_el.findtext("beat-type", default="")
                time_repr = f"{beats}/{beat_type}"
                if time_repr != last_time:
                    if last_time is not None:
                        time_changes += 1
                    last_time = time_repr

    feats.key_change_count = key_changes
    feats.time_change_count = time_changes

    # 计算综合复杂度分
    feats.complexity_score = feats.compute_complexity_score()

    return feats


def find_musicxml_files(input_path: Path):
    """
    在给定路径下查找所有 MusicXML / MXL 文件。
    """
    if input_path.is_file():
        if input_path.suffix.lower() in (".mxl", ".musicxml", ".xml"):
            return [input_path]
        return []

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in (".mxl", ".musicxml", ".xml"):
            files.append(p)
    return files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MusicXML/MXL files and estimate score complexity."
    )
    parser.add_argument(
        "input_path",
        help="Path to a .mxl/.musicxml/.xml file or a directory containing such files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="complexity_scores.csv",
        help="Output CSV filename (default: complexity_scores.csv)",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=1,
        help="Minimum number of notes required to keep a piece (default: 1).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = find_musicxml_files(input_path)
    if not files:
        print("No .mxl/.musicxml/.xml files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s). Analyzing...")

    feature_list = []
    for f in files:
        try:
            feats = analyze_musicxml_file(f)
            if feats.total_notes < args.min_notes:
                # 过滤掉音符太少的（比如只有几拍示例）
                print(f"[INFO] Skip {f} because total_notes={feats.total_notes}", file=sys.stderr)
                continue
            feature_list.append(feats)
        except Exception as e:
            print(f"[WARN] Failed to process {f}: {e}", file=sys.stderr)

    if not feature_list:
        print("No valid scores after filtering.", file=sys.stderr)
        sys.exit(1)

    # 按复杂度从高到低排序
    feature_list.sort(key=lambda x: x.complexity_score, reverse=True)

    fieldnames = list(asdict(feature_list[0]).keys())
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for feats in feature_list:
            writer.writerow(asdict(feats))

    print(f"Wrote {len(feature_list)} rows to {args.output}")
    print("Top 5 (most complex) files:")
    for feats in feature_list[:5]:
        print(f"{feats.complexity_score:8.3f}  {feats.filename}")


if __name__ == "__main__":
    main()
