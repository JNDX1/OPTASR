# OPTASR 自动化分析流程

## 项目简介

**OPTASR** (Optimized Ancestral Sequence Reconstruction) 是一个生物信息学综合分析流程，集成了8个核心功能模块，用于从原始序列数据到祖先序列重建的全流程自动化分析。

### 主要功能

1. **序列处理和翻译** - 将DNA序列翻译为蛋白质序列
2. **去除重复序列** - 识别并处理重复的序列和登录号
3. **序列排序** - 对序列进行标准化排序
4. **信号肽分析** - 预测并移除信号肽
5. **多序列比对** - 使用ClustalW进行序列比对
6. **进化树构建** - 使用MEGA-CC构建最大似然树
7. **祖先序列重建** - 使用ANCESCON重建祖先序列
8. **祖先序列汇总** - 整理和过滤重建的祖先序列

## 目录结构

```
OPTASR/
├── integrated_pipeline.py     # 主脚本
├── Sequence/                  # 序列相关目录
│   ├── Handing/               # 处理中的序列
│   ├── paixu/                 # 排序后的序列
│   └── Signal/                # 信号肽分析结果
├── MAS/                       # 多序列比对结果
├── Tree/                      # 进化树相关
│   ├── MSA/                   # 多序列比对文件
│   ├── modelsele/             # 模型选择结果
│   └── MLtree/                # 最大似然树文件
├── ANCESCONS/                 # 祖先序列重建结果
│   └── Ancestralsequence/     # 祖先序列文件
├── README.md                  # 项目说明
├── requirements.txt           # 依赖项
└── LICENSE                    # 许可证
```

## 安装与配置

### 前提条件

- Python 3.6 或更高版本
- 以下外部工具：
  - SignalP 6.0
  - ClustalW
  - MEGA-CC
  - ANCESCON

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/yourusername/OPTASR.git
   cd OPTASR
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置工具路径**
   编辑 `integrated_pipeline.py` 文件中的 `Config` 类，设置以下路径：
   - `SIGNALP_PATH` - SignalP 6.0 可执行文件路径
   - `MEGA_DIR` - MEGA-CC 目录
   - `ANCESCON_PATH` - ANCESCON 可执行文件路径

## 使用方法

### 准备输入数据

1. 将原始FASTA格式的序列文件放入 `Sequence/Handing/` 目录，命名为 `laccase.fasta`

### 运行流程

```bash
python integrated_pipeline.py
```

### 命令行参数

- `--skip-completed` - 跳过已完成的模块（如果输出文件已存在）
- `--verbose` - 启用详细输出

示例：
```bash
python integrated_pipeline.py --skip-completed --verbose
```

## 输出文件

- **序列处理**：`Sequence/Handing/laccase_translated.fasta`
- **去重结果**：`Sequence/Handing/unique_sequences.fasta`
- **排序结果**：`Sequence/paixu/laccase_sorted.fasta`
- **信号肽分析**：`Sequence/Signal/laccase_nosignalp.fasta`
- **多序列比对**：`MAS/output.aln`
- **进化树**：`Tree/MLtree/` 目录下的 `.nwk` 文件
- **祖先序列**：`ANCESCONS/Ancestralsequence/Anc_sequence.txt`

## 运行历史

每次运行的详细信息会记录在 `run_history.json` 文件中，包括：
- 运行时间戳
- 生成的文件路径
- 目录结构信息

## 故障排除

### 常见问题

1. **工具路径错误**
   - 确保 `Config` 类中的工具路径设置正确

2. **内存不足**
   - 对于大型数据集，可能需要增加系统内存或减少输入序列数量

3. **权限问题**
   - 确保所有工具可执行文件有执行权限

4. **文件不存在错误**
   - 确保输入文件 `laccase.fasta` 存在于正确位置

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 贡献

欢迎提交问题和拉取请求！

## 联系方式

- 项目维护者：[Your Name]
- 邮箱：[your.email@example.com]
