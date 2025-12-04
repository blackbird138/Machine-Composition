# 项目架构说明

## 概览

本项目采用模块化设计，将遗传算法音乐生成系统拆分为多个独立模块，每个模块负责特定功能。这种设计提高了代码的可维护性、可测试性和可扩展性。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│                      (程序入口点)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              GeneticAlgorithm (核心协调器)                   │
│         src/genetic_algorithm.py                            │
└──┬────────┬──────────┬─────────────┬─────────┬──────────────┘
   │        │          │             │         │
   │        │          │             │         │
   ▼        ▼          ▼             ▼         ▼
┌─────┐ ┌──────┐  ┌───────┐   ┌──────────┐ ┌─────────┐
│Pop- │ │Fit-  │  │Genetic│   │Musical   │ │Melody   │
│ula- │ │ness  │  │Opera- │   │Trans-    │ │Exporter │
│tion │ │Eva-  │  │tors   │   │forms     │ │         │
│     │ │lua-  │  │       │   │          │ │         │
│Gen. │ │tor   │  │       │   │          │ │         │
└─────┘ └──────┘  └───────┘   └──────────┘ └─────────┘
   │        │          │             │            │
   └────────┴──────────┴─────────────┴────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Melody & Note      │
            │  (数据模型)          │
            └─────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Constants          │
            │  (配置参数)          │
            └─────────────────────┘
```

## 模块说明

### 1. 数据模型层

#### `src/constants.py`
- **职责**：定义全局常量和配置参数
- **内容**：
  - 音高映射（MIDI值）
  - 可用时值
  - 音阶定义
  - 音乐参数（拍号、调号等）
- **依赖**：无
- **被依赖**：所有其他模块

#### `src/note.py`
- **职责**：表示单个音符
- **功能**：
  - 存储音高和时值
  - 转换为ABC记谱法
- **依赖**：constants.py
- **被依赖**：melody.py, population.py, genetic_operators.py

#### `src/melody.py`
- **职责**：表示音符序列（旋律）
- **功能**：
  - 管理音符列表
  - 计算总时长
  - 转换为ABC字符串
  - 存储适应度分数
- **依赖**：无（接受Note对象）
- **被依赖**：所有操作模块

### 2. 功能模块层

#### `src/fitness.py`
- **职责**：评估旋律质量
- **功能**：
  - 音阶符合度评估（40分）
  - 旋律流畅度评估（30分）
  - 节奏多样性评估（30分）
- **核心类**：`FitnessEvaluator`
- **依赖**：constants.py
- **被依赖**：genetic_algorithm.py

#### `src/genetic_operators.py`
- **职责**：实现遗传算法的基本操作
- **功能**：
  - 锦标赛选择（tournament selection）
  - 单点交叉（single-point crossover）
  - 音高和时值变异（mutation）
- **核心类**：`GeneticOperators`
- **依赖**：note.py, melody.py, constants.py
- **被依赖**：genetic_algorithm.py

#### `src/musical_transforms.py`
- **职责**：实现音乐理论变换
- **功能**：
  - 移调（transpose）：平移半音
  - 倒影（inversion）：镜像翻转
  - 逆行（retrograde）：时间反转
- **核心类**：`MusicalTransforms`
- **依赖**：constants.py
- **被依赖**：genetic_algorithm.py

#### `src/population.py`
- **职责**：生成初始种群
- **功能**：
  - 创建随机旋律
  - 确保满足约束条件（16拍）
- **核心类**：`PopulationGenerator`
- **依赖**：note.py, melody.py, constants.py
- **被依赖**：genetic_algorithm.py

#### `src/exporter.py`
- **职责**：导出旋律到文件
- **功能**：
  - ABC记谱法导出
  - MIDI文件导出（可选）
- **核心类**：`MelodyExporter`
- **依赖**：constants.py, midiutil（可选）
- **被依赖**：main.py

### 3. 控制层

#### `src/genetic_algorithm.py`
- **职责**：协调整个遗传算法流程
- **功能**：
  - 初始化所有组件
  - 管理种群
  - 执行进化循环
  - 实现精英保留策略
  - 统计和报告
- **核心类**：`GeneticAlgorithm`
- **依赖**：所有功能模块
- **被依赖**：main.py

#### `main.py`
- **职责**：程序入口点
- **功能**：
  - 配置参数
  - 启动遗传算法
  - 显示结果
  - 导出文件
- **依赖**：genetic_algorithm.py, exporter.py
- **被依赖**：无

## 数据流

### 初始化阶段
```
main.py 
  → GeneticAlgorithm.__init__() 
    → PopulationGenerator.generate() 
      → Note, Melody (创建实例)
```

### 进化循环
```
GeneticAlgorithm.evolve()
  ┌─→ FitnessEvaluator.evaluate() (评估)
  │   
  ├─→ GeneticOperators.select_parent() (选择)
  │   
  ├─→ GeneticOperators.crossover() (交叉)
  │   
  ├─→ GeneticOperators.mutate() (变异)
  │   
  ├─→ MusicalTransforms.* (可选变换)
  │   
  └─→ 回到评估 (下一代)
```

### 导出阶段
```
main.py 
  → MelodyExporter.to_abc_file() 
    → Melody.to_abc() 
      → Note.to_abc()
```

## 设计原则

### 1. 单一职责原则 (SRP)
每个模块只负责一个明确的功能：
- `fitness.py` 只做评估
- `genetic_operators.py` 只做遗传操作
- `musical_transforms.py` 只做音乐变换

### 2. 开闭原则 (OCP)
易于扩展，无需修改现有代码：
- 添加新的适应度标准：扩展 `FitnessEvaluator`
- 添加新的变换：扩展 `MusicalTransforms`
- 更改音乐参数：修改 `constants.py`

### 3. 依赖倒置原则 (DIP)
高层模块不依赖低层实现细节：
- `GeneticAlgorithm` 通过明确的接口使用各模块
- 数据模型（Note, Melody）独立于操作逻辑

### 4. 接口隔离原则 (ISP)
各类提供专注的公共接口：
- `Note.to_abc()` - 转换接口
- `Melody.add_note()` - 操作接口
- `FitnessEvaluator.evaluate()` - 评估接口

## 使用建议

### 开发新功能
1. 确定功能属于哪一层
2. 在对应模块中添加方法
3. 更新依赖关系
4. 在main.py中集成

### 调整参数
- **音乐参数**：修改 `src/constants.py`
- **算法参数**：修改 `main.py` 中的初始化
- **评估权重**：修改 `src/fitness.py` 中的分数分配

### 添加新的遗传操作
1. 在 `src/genetic_operators.py` 添加方法
2. 在 `src/genetic_algorithm.py` 中调用
3. 更新文档

### 添加新的音乐变换
1. 在 `src/musical_transforms.py` 添加静态方法
2. 在 `_apply_random_transform()` 中添加选项
3. 更新文档