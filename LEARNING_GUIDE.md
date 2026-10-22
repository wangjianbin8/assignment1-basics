# 仓库学习导航

你平时学习、运行命令的项目根目录是：

```text
/home/wangjianbin/assignment1-basics/assignment1-basics
```

外面还有一层同名目录，不要把两层混淆。本文的所有命令都在上面这个项目根目录运行。

## 目录：按你想做的事查找

- [想知道文件放在哪里：目录分工](#layout)
- [想运行程序：入口脚本及用途](#entries)
- [想读自己写的实现：核心代码](#core)
- [想看数据和存档：artifacts](#artifacts)
- [想从头操作：训练与生成命令](#commands)
- [想找整理前的文件：移动对照表](#moves)
- [想复习：笔记与当前进度](#progress)

<a id="layout"></a>

## 1. 目录分工

```text
assignment1-basics/
├── LEARNING_GUIDE.md                 ← 先读这里：中文导航
├── cs336_assignment1_basics.pdf       ← 作业讲义
├── cs336_basics/                     ← 核心实现：模型、分词器、优化器等
├── scripts/                          ← 训练、准备数据等入口程序
├── examples/                         ← 生成演示、单独的知识点练习
├── tests/                            ← 作业测试，以及连接测试的 adapters.py
├── notes/                            ← 学习笔记
├── artifacts/                        ← 生成的数据、词表和模型存档
├── README.md                         ← 作业原始说明，新增中文导航链接
├── pyproject.toml                    ← 项目依赖和工具配置
├── uv.lock                           ← 锁定依赖版本
├── make_submission.sh                ← 作业提供的测试、打包脚本
├── CHANGELOG.md                      ← 作业版本变更记录
└── LICENSE                           ← 许可证
```

`.venv/` 是 Python 虚拟环境，`.vscode/` 是编辑器配置，`__pycache__/` 和 `.pytest_cache/` 是自动生成的缓存。这些不是作业实现。

<a id="entries"></a>

## 2. 入口脚本：什么时候运行哪个文件？

统一使用 `uv run python -m 目录名.模块名`。例如 `scripts/train.py` 对应 `uv run python -m scripts.train`，不写 `.py`。

`-m` 表示按模块运行，方便 Python 从项目根目录找到 `cs336_basics`。脚本中的数据相对路径也仍然以项目根目录为起点。

### scripts：准备数据和训练

| 文件 | 用来做什么 | 对应讲义 | 主要输出 |
|---|---|---|---|
| [train.py](scripts/train.py) | 通用训练入口：加载数据、训练、验证、调度学习率、保存与恢复进度 | §5.3 `training_together` | 指定目录下的模型检查点 |
| [train_bpe_debug.py](scripts/train_bpe_debug.py) | 在 `ab、abc、bc` 小语料上训练 BPE，检查保存和读取流程 | §2 BPE 训练练习 | `artifacts/bpe_debug/` |
| [prepare_demo_data.py](scripts/prepare_demo_data.py) | 生成反复循环的 token ID `0～15`，练习最小训练流程 | §5 训练辅助演示 | `artifacts/demo_data/` |
| [prepare_text_data.py](scripts/prepare_text_data.py) | 用已有 BPE 编码几句英语，准备文本训练和验证数据 | §5～6 辅助演示 | `artifacts/text_demo/` |
| [plot_learning_curve.py](scripts/plot_learning_curve.py) | 读取实验 CSV，分别按步数和耗时绘制损失曲线 | §7.1 `experiment_log` | CSV 旁边的同名 PNG |

`train_bpe_debug.py` 只是小语料演示，尚未训练完整 TinyStories，之前的文件名容易让人误会，因此改名。

### examples：演示和独立练习

| 文件 | 用来做什么 | 依赖什么 |
|---|---|---|
| [generate_token_ids.py](examples/generate_token_ids.py) | 输入 `[0, 1, 2]`，生成后续数字 ID | `artifacts/checkpoints/run1/final.pt`，词表大小为 16 的模型 |
| [generate_text.py](examples/generate_text.py) | 输入 `The cat`，经过分词、模型生成和解码得到文本 | BPE 词表和 `artifacts/checkpoints/text_demo/final.pt` |
| [memmap_demo.py](examples/memmap_demo.py) | 用小数组演示内存映射、读取片段和采样 | 无需训练模型；输出在 `artifacts/memmap_demo/` |
| [learning_rate_tuning.py](examples/learning_rate_tuning.py) | 比较 SGD 学习率 10、100、1000 的行为 | §4.2 `learning_rate_tuning`，使用自己的 SGD |

`scripts/__init__.py` 和 `examples/__init__.py` 标识模块目录，本身不是要运行的练习。

<a id="core"></a>

## 3. 核心代码：想复习某个知识点看哪里？

这些文件主要被其他程序导入，不需要逐个运行。

| 学习内容 | 文件 | 作用 |
|---|---|---|
| BPE 训练 | [bpe.py](cs336_basics/bpe.py) | 预分词、相邻对计数、选择和执行合并、构建词表 |
| 分词器 | [tokenizer.py](cs336_basics/tokenizer.py) | 文本编码、ID 解码、特殊 token、流式编码 |
| 预分词参考 | [pretokenization_example.py](cs336_basics/pretokenization_example.py) | 作业提供的预分词辅助示例 |
| 线性层 | [linear.py](cs336_basics/linear.py) | 无偏置线性变换 |
| Embedding | [embedding.py](cs336_basics/embedding.py) | token ID 查表得到向量 |
| RMSNorm | [rmsnorm.py](cs336_basics/rmsnorm.py) | 归一化 |
| SwiGLU、SiLU | [swiglu.py](cs336_basics/swiglu.py) | 前馈网络及其激活函数 |
| RoPE | [rope.py](cs336_basics/rope.py) | 为 Q、K 引入位置信息 |
| Softmax | [softmax.py](cs336_basics/softmax.py) | 将分数归一化为概率 |
| 缩放点积注意力 | [attention.py](cs336_basics/attention.py) | QK 分数、mask、Softmax、对 V 加权 |
| 多头注意力 | [multihead_attention.py](cs336_basics/multihead_attention.py) | QKV 投影、拆头、RoPE、因果注意力、合头 |
| Transformer Block | [transformer_block.py](cs336_basics/transformer_block.py) | RMSNorm、注意力、FFN、残差 |
| 完整语言模型 | [transformer_lm.py](cs336_basics/transformer_lm.py) | Embedding → 多个 Block → RMSNorm → LM Head |
| 交叉熵 | [cross_entropy.py](cs336_basics/cross_entropy.py) | 根据 logits 和目标 ID 计算损失 |
| SGD | [sgd.py](cs336_basics/sgd.py) | 学习优化器接口的简单例子 |
| AdamW | [adamw.py](cs336_basics/adamw.py) | 利用 m、v、偏差修正和权重衰减更新参数 |
| 学习率调度 | [lr_schedule.py](cs336_basics/lr_schedule.py) | 预热 → 余弦下降 → 保持最小学习率 |
| 梯度裁剪 | [gradient_clipping.py](cs336_basics/gradient_clipping.py) | 限制整个模型梯度的 L2 范数 |
| 数据采样 | [data_loader.py](cs336_basics/data_loader.py) | 从一维 token 数组采样错开一位的 x、y |
| 检查点 | [checkpoint.py](cs336_basics/checkpoint.py) | 保存和恢复模型、优化器及已完成步数 |
| 生成函数 | [decoding.py](cs336_basics/decoding.py) | 逐 token 生成、温度、top-p、结束符和文本入口 |

`tests/adapters.py` 负责把作业测试接到这些实现；具体测试在同目录其他 `test_*.py` 中。整理没有移动测试或更改核心计算逻辑。

<a id="artifacts"></a>

## 4. artifacts：这些数据和存档分别属于哪次实验？

| 目录 | 保存什么 | 与什么配套？ |
|---|---|---|
| `artifacts/bpe_debug/` | 示例文本、`vocab.pkl`、`merges.pkl` | 当前文本演示的分词器 |
| `artifacts/demo_data/` | 数字循环的 `train.npy`、`val.npy` | 词表大小为 16 的数字实验 |
| `artifacts/text_demo/` | BPE 编码的 `train.npy`、`val.npy` | 词表大小为 263 的文本实验 |
| `artifacts/checkpoints/run1/` | 数字模型检查点 | `examples.generate_token_ids` |
| `artifacts/checkpoints/text_demo/` | 文本模型检查点 | `examples.generate_text` |
| `artifacts/memmap_demo/` | 内存映射练习的小数组，运行演示后生成 | `examples.memmap_demo` |

`.npy` 是数组数据，`.pkl` 在本项目中是保存的词表和合并规则，`.pt` 是模型训练检查点。

数字模型和文本模型的 token 编号含义不同，不能互换存档。文本生成必须使用训练该模型时的同一套分词器。

<a id="commands"></a>

## 5. 常用运行命令

### 5.1 查看训练脚本接受哪些参数

```bash
uv run python -m scripts.train --help
```

### 5.2 当前最常用：文本训练与生成

已有 `artifacts/bpe_debug/vocab.pkl` 和 `merges.pkl` 时，直接准备文本数据。若需要重新练习小语料 BPE，可运行 `uv run python -m scripts.train_bpe_debug`；它会重写该目录中的演示文件。

```bash
uv run python -m scripts.prepare_text_data

uv run python -m scripts.train \
    --train-data artifacts/text_demo/train.npy \
    --val-data artifacts/text_demo/val.npy \
    --vocab-size 263 \
    --context-length 16 \
    --d-model 64 \
    --num-layers 2 \
    --num-heads 4 \
    --d-ff 192 \
    --batch-size 4 \
    --total-steps 300 \
    --warmup-iters 10 \
    --cosine-cycle-iters 299 \
    --checkpoint-dir artifacts/checkpoints/text_demo \
    --save-interval 100

uv run python -m examples.generate_text
```

263 是现有演示词表大小。换分词器后，以数据准备脚本打印的值为准；生成演示的模型结构也要与训练配置保持一致。

如果现有文本存档已经训练成功，只需要最后一条生成命令，不必重复准备和训练。

### 5.3 回顾数字训练流程

```bash
uv run python -m scripts.prepare_demo_data

uv run python -m scripts.train \
    --train-data artifacts/demo_data/train.npy \
    --val-data artifacts/demo_data/val.npy \
    --checkpoint-dir artifacts/checkpoints/run1 \
    --save-interval 50 \
    --total-steps 100

uv run python -m examples.generate_token_ids
```

### 5.4 从检查点继续训练

例如文本模型已完成 300 次更新，希望继续到累计 400 次，保持原先在 299 步结束的学习率调度：

```bash
uv run python -m scripts.train \
    --train-data artifacts/text_demo/train.npy \
    --val-data artifacts/text_demo/val.npy \
    --vocab-size 263 \
    --total-steps 400 \
    --cosine-cycle-iters 299 \
    --checkpoint-dir artifacts/checkpoints/text_demo \
    --resume artifacts/checkpoints/text_demo/final.pt \
    --save-interval 100
```

这里其余模型参数使用默认值，与上面的文本训练配置相同。若原训练改过结构参数，恢复时也要提供匹配配置。

### 5.5 单独练习和测试

```bash
uv run python -m examples.memmap_demo
uv run python -m examples.learning_rate_tuning

uv run pytest tests/test_tokenizer.py -v
uv run pytest tests/test_model.py -v
uv run pytest tests/test_optimizer.py -v
uv run pytest tests/test_serialization.py -v
uv run pytest
```

<a id="moves"></a>

## 6. 整理前后的文件位置

旧位置已移动，不保留重复入口。以后使用上面的模块运行命令。

| 整理前 | 整理后 |
|---|---|
| 根目录 `train.py` | `scripts/train.py` |
| 根目录 `prepare_text_data.py` | `scripts/prepare_text_data.py` |
| 根目录 `generate_demo.py` | `examples/generate_token_ids.py` |
| 根目录 `generate_text_demo.py` | `examples/generate_text.py` |
| `cs336_basics/prepare_demo_data.py` | `scripts/prepare_demo_data.py` |
| `cs336_basics/train_bpe_tinystories.py` | `scripts/train_bpe_debug.py` |
| `cs336_basics/memmap_demo.py` | `examples/memmap_demo.py` |
| `cs336_basics/learning_rate_tuning.py` | `examples/learning_rate_tuning.py` |

已有 `artifacts/` 中的数据和模型存档保留原路径，不需要重新训练。

<a id="progress"></a>

## 7. 笔记和当前学习位置

- [Transformer LM 流程和参数量](notes/transformer_lm.md)：复习模型各层、张量形状、参数量公式。
- [资源计算笔记](notes/resource_accounting.md)：复习显存、激活值、FLOPs、AdamW 开销和训练耗时，文首有目录。
- [2026-10-22 学习笔记](notes/2026-10-22_learning_notes.md)：串起优化器、训练流程、文本生成、采样和实验记录，含易混淆点与目录。

目前学到讲义第 7.1 节的 `experiment_log` 题，正在建立 CSV 日志、配置记录和学习曲线。第 6 节文本生成演示已跑通；曲线脚本已加入，实际绘图结果仍待运行确认。

第 7 节的正式实验尚未完成；前面跳过的真实语料下载、完整语料 BPE 训练和相关实验也没有被这些小数据演示替代。

日常学习：读讲义 → 在 `cs336_basics/` 找实现 → 在 `scripts/` 或 `examples/` 找入口 → 用 `tests/` 验证 → 在 `notes/` 复习。
