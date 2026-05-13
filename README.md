# Pythia Inference Optimization

一个基于 `EleutherAI/pythia-70m` 的长上下文推理优化实验项目，实现了通过 `StreamingLLM`，`H2O`，`SnapKV`对 KV Cache 进行压缩，并在固定数据集上，基于`TTFT`、`TPOT`、`Throughput` 和 `PPL`4项指标，评估不同策略对推理性能和困惑度的影响。


### 1. 模型加载与推理

- 默认模型为 `EleutherAI/pythia-70m`
- 设备配置：
  - `cuda` 可用时使用 `float16`
  - 否则退化到 `cpu + float32`
- 启用 `use_cache=True`
- 强制使用 `attn_implementation="eager"`，便于显式获取 attention 权重

主入口见 [main.py](main.py)。

### 2. KV Cache 压缩策略

在 [core/compressor.py](core/compressor.py) 中实现了以下压缩策略：

- `StreamingLLMCompressor`
  - 保留前部 `sink tokens` 与最近 `window tokens`
- `SnapKVCompressor`
  - 基于中间 token 的 key 范数打分，并结合 `avg_pool1d` 平滑后选择 Top-K
- `H2OCompressor`
  - 累积 attention 分数，保留高分历史 token、sink token 与最近窗口

`main.py` 中，将默认 dense 路径作为 baseline，与上述 3 种算法进行测试与对比。

当前 benchmark 已引入 **控制变量法**：除 `Dense Baseline` 以外，其余压缩算法统一使用固定 `compression_ratio = 0.5`。也就是说，在每个给定的上下文长度下，三种压缩算法都以“保留约一半 token”为目标进行比较。

在这一设置下：

- `StreamingLLM` 根据当前 `context_length` 动态换算保留容量，并通过调整 `window_size + sink_size` 达到约 `0.5` 的保留比例
- `H2O` 根据当前 `context_length` 动态设置 `max_capacity`
- `SnapKV` 在 benchmark 中同样按当前长度动态换算目标容量，使其与其他压缩算法在同一压缩率下比较

- Baseline（Dense）
- StreamingLLM（Ratio=0.5）
- H2O（Ratio=0.5）
- SnapKV（Ratio=0.5）


### 3. Attention Patch 机制

[core/patcher.py](core/patcher.py) 会遍历模型中的 `GPTNeoXAttention` 模块，并：

- 保存原始 `forward`
- 注入 `compressor`
- 指定 `output_attentions=True`
- 在每层 attention 输出 `past_key_values` 后执行压缩


### 4. Benchmark 指标

[scripts/benchmark.py](scripts/benchmark.py) 中的 `Evaluator` 实现了以下指标：

- `TTFT (Time To First Token)`
- `TPOT (Time Per Output Token)`
- `Throughput (tokens/s)`
- `PPL (Perplexity)`

其中：

- `TTFT` 通过一次完整 prefill 后，取首个生成 token 的耗时计算
- `TPOT` 通过逐 token 解码，统计平均单步时延
- `PPL` 通过自回归缓存路径，逐步计算交叉熵

## 数据集与测试流程

### 主流程实际使用的数据

`main.py` 当前会评测两个数据源：

- `PG-19 (Fiction)`
  - 通过 `load_dataset("emozilla/pg19", streaming=True)` 流式读取
- `Wikitext (Encyclopedia)`
  - 通过 `load_dataset("wikitext", "wikitext-2-raw-v1")` 加载并拼接全文

默认上下文长度：

- `1024`
- `2048`
- `3072`
- `4096`

默认生成长度：

- `50` tokens

默认压缩率：

- `0.5`

默认有效重复次数：

- `3`

### 数据弃置机制

为避免输入 token 数变化后，触发新的显存分配与额外启动成本，导致首次测量所得时间值偏大的问题，`main.py`中引入弃置首条数据的逻辑：

- 每当切换到新的 `context_length` 后，每个算法都会预先执行 `1` 次额外调用
- 这条结果不会参与统计，只用于预热当前输入长度下的执行路径
- 均值和标准差统计数据，来源于后续 `3` 次有效结果

因此，单个算法在单个上下文长度下的实际执行次数为：

- `1` 次丢弃轮次
- `3` 次有效轮次

按照当前默认配置，每个数据集共执行：

- `4` 组上下文长度
- 每组 `4` 种算法（`1` 个 baseline + `3` 个压缩算法）
- 合计 `16` 组实验配置


## 项目结构

```text
pythia_inference_opt/
├── core/
│   ├── compressor.py      # 各类 KV Cache 压缩策略
│   └── patcher.py         # 将压缩逻辑注入 GPTNeoXAttention
├── data/
│   └── loader.py          # 独立的数据加载实验脚本
├── scripts/
│   └── benchmark.py       # TTFT / TPOT / Throughput / PPL 评估器
├── main.py                # 主 benchmark 入口
├── requirements.txt       # 依赖列表
└── README.md
```

## 环境依赖

项目依赖见 [requirements.txt](requirements.txt)，核心包括：

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `flash_attn`


## 安装

```bash
pip install -r requirements.txt
```


## 运行方式

执行：

```bash
python main.py
```

运行流程如下：

1. 加载 tokenizer 与模型
2. 若检测到 CUDA，则先做一次预热
3. 分别加载 PG-19 与 Wikitext 测试文本
4. 在不同上下文长度上，对各压缩算法按固定 `compression_ratio = 0.5` 展开测试
5. 每次切换输入 token 数后，先执行 1 次丢弃轮次
6. 保留 3 次有效测量并输出汇总表


## 输出结果示意

程序会按算法输出如下指标：

```text
TTFT=...s | TPOT=...s/token | Throughput=... tk/s | PPL=...
```

最后会按数据集与上下文长度打印汇总表，便于横向比较不同压缩策略。


## 项目报告

基于 [results/results.md](results/results.md) 中记录的实验结果，可以观察到：KV Cache 压缩在短上下文下收益有限，但在长上下文下能够显著提升解码速度。以 `compression_ratio = 0.5` 为例，`1024` tokens 时，baseline 吞吐已经较高，压缩方法整体没有明显优势；而当上下文增长到 `2048`、`3072`、`4096` tokens 后，压缩带来的收益迅速变得明显。

从吞吐率看，`StreamingLLM` 是当前实现中加速最明显的方案。在 `PG-19` 上，baseline 从 `1024` 到 `4096` tokens 的吞吐率由 `175.31 tk/s` 下降到 `10.51 tk/s`，而 `StreamingLLM` 在 `4096` tokens 时仍达到 `140.73 tk/s`，相对 baseline 约提升 `13.4x`；在 `Wikitext` 上，同样从 baseline 的 `10.97 tk/s` 提升到 `123.69 tk/s`，约为 `11.3x`。`SnapKV` 的加速幅度略低于 `StreamingLLM`，但整体仍显著优于 baseline；`H2O` 也能带来稳定加速，不过在这组参数下综合表现弱于前两者。

从精度指标 `PPL` 看，三种压缩方法都会造成一定质量损失。其中 `StreamingLLM` 通常速度最快，但 `PPL` 上升也较明显；`SnapKV` 在多数设置下的 `PPL` 低于 `StreamingLLM` 和 `H2O`，表现出更好的速度-质量折中。例如在 `PG-19, 3072 tokens` 上，baseline 的 `PPL` 为 `32.0812`，`StreamingLLM` 为 `91.1208`，`H2O` 为 `96.7746`，而 `SnapKV` 为 `84.3961`。因此，这组实验表明：如果优先追求推理速度，`StreamingLLM` 最有优势；如果同时考虑速度与输出质量，`SnapKV` 是当前实现中更均衡的选择。


## 总结

这是一个围绕 `pythia-70m` 的 KV Cache 压缩 benchmark 原型，实现了“压缩策略接入 + 数据加载 + 指标评测”的功能。
