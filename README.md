# 机器作曲 - 遗传算法

使用遗传算法（Genetic Algorithm）自动生成音乐旋律的Python实现。

## 📋 项目概述

本项目实现了一个基于遗传算法的音乐自动创作系统，能够：
- 随机生成初始旋律种群
- 通过遗传操作（交叉、变异）进行进化
- 应用音乐变换（移调、倒影、逆行）
- 根据音乐学理论评估旋律质量
- 导出ABC记谱法和MIDI文件

## 🎵 音乐参数

- **音域**：F3 到 G5（完整半音阶）
- **节拍**：4/4拍
- **长度**：4小节（16拍）
- **时值**：八分音符、四分音符、二分音符

## 🧬 遗传算法特性

### 适应度函数
基于三个音乐学标准（总分100分）：
1. **音阶符合度**（40分）：偏好C大调音阶
2. **旋律流畅度**（30分）：惩罚过大音程跳跃
3. **节奏多样性**（30分）：奖励时值变化

### 遗传操作
- **交叉**：单点交叉结合两个父代
- **变异**：随机改变音高和时值
- **音乐变换**：
  - 移调（Transpose）
  - 倒影（Inversion）
  - 逆行（Retrograde）

### 选择策略
- 锦标赛选择（tournament size = 3）
- 精英保留（保留最优2个个体）

## 🚀 快速开始

### 运行程序
```bash
python main.py
```

### 预期输出
程序将：
1. 生成20个随机初始旋律
2. 进化100代
3. 输出Top 3最优旋律
4. 生成ABC记谱法文件（`melody_1.abc`, `melody_2.abc`, `melody_3.abc`）
5. 尝试生成MIDI文件（需要安装midiutil）

### 可选：安装MIDI支持
```bash
pip install midiutil
```

## 📊 实验结果

典型的进化过程：
- **初始适应度**：约82分
- **最终适应度**：约140分
- **提升幅度**：69.7%
- **收敛代数**：约70代

## 🎼 播放生成的旋律

### 方法1：ABC在线播放器
1. 访问 https://abcjs.net/abcjs-editor.html
2. 将生成的ABC文件内容复制到编辑器
3. 点击播放按钮聆听

### 方法2：MIDI播放
如果安装了midiutil，会生成MIDI文件：
- `melody_1.mid`
- `melody_2.mid`
- `melody_3.mid`

使用任何MIDI播放器打开（如Windows Media Player、VLC等）

## 📁 项目结构

```
Machine-Composition/
├── src/                        # 源代码目录
│   ├── __init__.py            # 包初始化文件
│   ├── constants.py           # 音乐常量和参数
│   ├── note.py                # Note类：音符表示
│   ├── melody.py              # Melody类：旋律表示
│   ├── fitness.py             # 适应度评估
│   ├── genetic_operators.py   # 遗传操作（交叉、变异、选择）
│   ├── musical_transforms.py  # 音乐变换（移调、倒影、逆行）
│   ├── population.py          # 种群生成
│   ├── genetic_algorithm.py   # 遗传算法主类
│   └── exporter.py            # 导出功能（ABC、MIDI）
├── main.py                    # 主程序入口
├── melody_1.abc               # 生成的旋律1（ABC记谱法）
├── melody_2.abc               # 生成的旋律2
├── melody_3.abc               # 生成的旋律3
├── EXPERIMENT_REPORT.md       # 详细实验报告
└── README.md                  # 本文件
```

## 🛠️ 自定义参数

可以在`main.py`的主函数中调整以下参数：

```python
ga = GeneticAlgorithm(
    population_size=20,    # 种群大小
    mutation_rate=0.15,    # 变异率
    crossover_rate=0.7     # 交叉率
)

best_melodies = ga.evolve(generations=100)  # 进化代数
```

或修改`src/constants.py`中的音乐参数：

```python
TARGET_BEATS = 16           # 小节数 × 每小节拍数
TIME_SIGNATURE = "4/4"      # 拍号
KEY_SIGNATURE = "Cmaj"      # 调号
DURATIONS = [0.5, 1.0, 2.0] # 可用时值
```

## 📖 原始需求

### 机器作曲·遗传算法

1. 采用下述方法产生初始种群：
   - 随机产生：给定乐音体系 $S=\{F_3,\sharp F_3,\dots,B_3,C_4,\sharp C_4,\dots,B_4,C_5,\sharp C_5,\dots,\sharp F_5,G_5\}$
   - 随机选取 $S$ 中的音级，配以不同的时值，产生 $10 \sim 20$ 段 $4/4$ 拍、$4$ 小节的"旋律"
   - 音符的最短时值为八分音符

2. 在任何一个软件平台上实现遗传算法
   - 遗传操作应包括交叉(crossover)、变异(mutation)
   - 对旋律进行的移调、倒影、逆行变换等

3. 探索建立适应度函数(fitness function)
   - 用以指导旋律进化的方向

4. 对初始种群进行遗传迭代
   - 看是否能够得到较好的音乐片段

## ✅ 需求完成情况

- ✅ 初始种群生成（20个旋律）
- ✅ 遗传算法实现
  - ✅ 交叉操作
  - ✅ 变异操作
  - ✅ 移调变换
  - ✅ 倒影变换
  - ✅ 逆行变换
- ✅ 适应度函数（基于音乐学理论）
- ✅ 遗传迭代（100代进化）
- ✅ 生成高质量旋律
- ✅ ABC记谱法导出
- ✅ MIDI文件导出（可选）

## 📚 项目文档

- **[README.md](README.md)** - 项目介绍和快速开始（本文档）
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 详细架构设计说明
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** - 实验报告和结果分析

## 📈 实验报告

详细的实验结果和分析请参阅 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)

内容包括：
- 实验设计详情
- 进化过程分析
- 生成旋律评价
- 讨论与改进方向

## 🔬 技术细节

### 模块架构
项目采用模块化设计，各模块职责清晰：

- **constants.py**：全局常量和音乐参数
- **note.py**：音符类，包含ABC转换
- **melody.py**：旋律类，管理音符序列
- **fitness.py**：适应度评估器，基于音乐学理论
- **genetic_operators.py**：遗传操作（选择、交叉、变异）
- **musical_transforms.py**：音乐变换（移调、倒影、逆行）
- **population.py**：随机种群生成
- **genetic_algorithm.py**：主算法协调器
- **exporter.py**：多格式导出

### 核心算法流程
1. **初始化**：随机生成满足约束的旋律（`PopulationGenerator`）
2. **评估**：计算每个旋律的适应度（`FitnessEvaluator`）
3. **选择**：锦标赛选择父代（`GeneticOperators.select_parent`）
4. **交叉**：单点交叉生成子代（`GeneticOperators.crossover`）
5. **变异**：随机改变音高/时值（`GeneticOperators.mutate`）
6. **变换**：10%概率应用音乐变换（`MusicalTransforms`）
7. **替换**：精英保留 + 新种群
8. **迭代**：重复2-7步直至收敛

## 🎓 参考资料

- ABC记谱法标准：http://abcnotation.com/
- 遗传算法理论
- 音乐信息检索（MIR）
- 算法作曲（Algorithmic Composition）

## 📝 许可

本项目为学术研究项目，仅供学习和研究使用。

---

Composing music using genetic algorithms. PKU Music And Math assignment.
