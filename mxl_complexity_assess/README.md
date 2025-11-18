
# 简单的 MusicXML / MXL 乐谱复杂度评估脚本

* 功能：
  - 扫描单个文件或一个目录下的所有 .mxl / .musicxml / .xml
  - 解析乐谱结构，提取若干“复杂度”相关特征
  - 计算一个综合复杂度分数 complexity_score
  - 将所有结果写入 CSV，方便后续筛选训练数据

* 用法示例：
    python assess_musicxml_complexity.py path/to/your/xml_or_folder -o scores.csv


## `ScoreFeatures` 相关字段

* 结构长度相关

  * `total_measures`：小节总数（用 `<measure number>` 去重）
  * `total_notes`：音符数（不含休止）
  * `total_rests`：休止符数

* 节奏 & 密度

  * `unique_durations`：不同时值种类数（从 `<note><type>` 收集）
  * `short_note_ratio`：短时值音符比例（八分及更细）
  * `chord_ratio`：和弦音比例（有 `<chord/>` 的 note）

* 声部 / 织体

  * `avg_voices_per_staff`：每个 staff 平均有多少个 voice（看 `<staff>` + `<voice>`）
  * `staff_count`、`part_count`

* 记号丰富度

  * `dynamics_count`：力度 `<dynamics>` 次数
  * `articulations_count`：表情 `<articulations>`
  * `ornaments_count`：装饰音 `<ornaments>`
  * `slur_count`：连音线 `<slur>`
  * `tie_count`：延音线 `<tie>`（在 `<note>` 里面）
  * `accidental_count`：临时升降号 `<accidental>`
  * `grace_note_count`：倚音 `<grace>`

* 调号 / 拍号变化

  * `key_change_count`：调号变化次数
  * `time_change_count`：拍号变化次数

* 歌词

  * `has_lyrics`（0/1）、`lyric_count`

## 评价逻辑
* 在 `compute_complexity_score()` 里，用对数缩放，避免“长度”影响，强调：

  * 节奏多样（时值种类、多短音）
  * 和声 / 多声部
  * 记号多（力度、表情、装饰、连/延音线、升降号）
  * 调号/拍号变化
  * 每小节音符密度