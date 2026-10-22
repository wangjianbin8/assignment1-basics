# 2026-10-22 学习笔记：从优化器到文本生成与实验记录

今天围绕讲义第 4～7 节，完成了小数据训练、文本生成的连接，并开始建立实验日志和学习曲线。

这份笔记记录原理、容易混淆的地方和当前仓库中的入口。资源计算的完整推导见 [资源计算笔记](resource_accounting.md)，目录和文件用途见 [仓库导航](../LEARNING_GUIDE.md)。

## 目录

| 内容 | 解决什么问题 |
|---|---|
| [1. 优化器与参数接口](#optimizer) | parameters、state_dict、step 分别干什么？ |
| [2. AdamW 的计算](#adamw) | m、v、偏差修正、权重衰减怎样组合？ |
| [3. 显存和计算量](#resources) | 装不装得下，与要算多久有什么区别？ |
| [4. 学习率调度与梯度裁剪](#schedule) | 怎样控制每轮参数更新的大小？ |
| [5. 数据采样与内存映射](#data) | 怎样准备 x、y，怎样读取大文件？ |
| [6. 保存与恢复检查点](#checkpoint) | 中断后怎样继续训练？ |
| [7. 训练、验证与模式](#training) | train、eval、no_grad 分别影响什么？ |
| [8. 命令行参数](#arguments) | 怎样不改代码就调整实验？ |
| [9. 自回归生成](#generation) | 怎样逐个生成 token？ |
| [10. 温度与 top-p](#sampling) | 怎样控制采样分布和候选范围？ |
| [11. 从 token ID 到文本](#text) | 分词器怎样与模型配套？ |
| [12. 实验记录与曲线](#logging) | 怎样保存配置、损失、步数和耗时？ |
| [13. 仓库使用与测试](#repository) | 现在各个文件放在哪里，怎样运行？ |
| [14. 当前进度与后续任务](#progress) | 哪些已练习，哪些还未完成？ |

<a id="optimizer"></a>

## 1. 优化器与参数接口

### parameters 和 state_dict

| 写法 | 提供什么 | 主要用途 |
|---|---|---|
| `model.parameters()` | 模型中的 Parameter 对象 | 交给优化器更新，参数对象上有 `.grad` |
| `model.state_dict()` | 名称到张量的字典，包含参数和持久 buffer | 保存和恢复模型状态 |
| `optimizer.state_dict()` | 优化器状态和参数组设置 | 恢复 AdamW 等优化器的训练历史 |

两种模型接口都涉及权重，但提供方式和用途不同。`state_dict()` 默认不是独立的数值快照，其张量与模型共享底层存储。

持久性为 False 的 buffer 不进入模型的 `state_dict()`；buffer 也不属于 `model.parameters()` 中的参数。

### 初始化和更新

```python
defaults = {"lr": lr}
super().__init__(params, defaults)
```

让优化器基类管理参数和设置，建立：

- `self.param_groups`：参数分组，每组包含参数和学习率等设置。
- `self.state[p]`：某个参数张量自己的历史状态。

`__init__()` 准备信息，`step()` 才真正更新参数。

```python
loss.backward()   # 计算梯度，保存在各参数的 .grad 中
optimizer.step()  # 读取梯度并更新参数
```

`step()` 内的可选 `closure` 用于重新计算损失。普通调用 `optimizer.step()` 不传它，内部的局部变量 `loss` 可以一直是 None，不影响通过 `.grad` 更新参数。

<a id="adamw"></a>

## 2. AdamW 的计算

每个参数张量都有同形状的 `m`、`v`，初始为零，并保存实际更新次数 `t`。

### 两份梯度历史

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

`m` 平滑梯度，保留近期的方向信息。

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

`v` 平滑梯度平方，反映近期梯度的大小。平方让正、负梯度都能作为大小信息积累。

### 偏差修正与讲义更新顺序

从零开始的 m、v 在前期偏小，需要修正。讲义将系数合并到学习率，且 t 从 1 开始：

$$
\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

然后按以下顺序执行：

1. 权重衰减：$p\leftarrow(1-\alpha\lambda)p$。
2. 更新 m、v。
3. 梯度更新：$p\leftarrow p-\alpha_t m/(\sqrt v+\varepsilon)$。

权重衰减使用原始学习率 alpha，不使用修正后的 alpha_t。先衰减还是先梯度更新会影响数值，按讲义顺序实现。

直接使用修正后的 m_hat、v_hat 是另一种常见写法，但 epsilon 的缩放位置可能不同，不能在本作业里随意替换公式。

### 原地操作怎么读

| 代码 | 含义 |
|---|---|
| `x.mul_(c)` | `x *= c` |
| `x.add_(a, alpha=c)` | `x += c * a` |
| `x.addcmul_(a, b, value=c)` | `x += c * a * b` |
| `x.addcdiv_(a, b, value=c)` | `x += c * a / b` |

乘除都是对应元素操作。方法名末尾的下划线表示直接修改原张量。

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)
```

先把 m 乘 beta1，再加上 `(1-beta1)*grad`。由于 m 指向 `state["m"]` 中保存的张量，原地修改也更新了保存的历史。

<a id="resources"></a>

## 3. 显存和计算量

训练显存主要统计四部分，全部按 float32 计算：

| 内容 | 字节数 |
|---|---:|
| 模型参数 | $4P$ |
| 参数梯度 | $4P$ |
| AdamW 状态 m、v | $8P$ |
| 激活值 | $4A$ |

$$
M\approx16P+4A
$$

参数和优化器状态跨轮次保留；激活值是当前批次计算产生的中间结果，用于反向传播。实际显存还包括临时张量和运行开销。

计算量是浮点运算次数，单位 FLOPs。矩阵乘法 $(m,n)\times(n,p)$ 约需要 $2mnp$ FLOPs。计算速度单位是 FLOP/s，不要与 FLOPs 混淆。

AdamW 每个参数元素约需 14 次运算，故更新约为 $14P$。修正学习率按参数张量计算一次，不是按张量里的每个元素计算，作为小项忽略。这个计数还依赖具体公式和运算统计约定。

讲义假设反向计算量约为前向两倍：

$$
F_{\text{step}}\approx3F_{\text{forward}}+14P
$$

$$
\text{训练时间}=\frac{\text{每步计算量}\times\text{步数}}{\text{有效计算速度}}
$$

完整的参数量、激活值公式和 GPT-2 XL 示例见 [resource_accounting.md](resource_accounting.md)。其中批大小 3、约 202 天等结论都依赖笔记中明确的简化口径和题设，不是实测。

<a id="schedule"></a>

## 4. 学习率调度与梯度裁剪

### 余弦调度

学习率分三段：

1. 预热：从零线性增加到最大值。
2. 余弦下降：从最大值平滑降到最小值。
3. 下降结束后：保持最小值。

预热结束步数为 Tw，余弦下降结束步数为 Tc，下降持续时间为 Tc-Tw。Tc 是结束位置，不是持续长度。

调度函数只返回数字，需要训练循环把它写进优化器：

```python
for group in optimizer.param_groups:
    group["lr"] = lr
```

循环 `range(N)` 的下标是 0～N-1；若要最后一次更新达到最小学习率，可以把 Tc 设置为 N-1。

### 梯度裁剪

把所有参数梯度合起来，计算：

$$
\|g\|_2=\sqrt{\sum_i g_i^2}
$$

如果超过上限 M，所有梯度都乘同一个正系数：

$$
g\leftarrow g\frac{M}{\|g\|_2+10^{-6}}
$$

例如梯度 `[3,4]` 长度为 5，上限为 1 时缩成约 `[0.6,0.8]`。方向不变，长度缩小。

修改的是 `.grad`，不是直接修改权重；顺序是 backward → 裁剪 → optimizer.step。

<a id="data"></a>

## 5. 数据采样与内存映射

### x、y 错开一个 token

假设原始数据是：

```text
10, 20, 30, 40, 50, 60
```

从下标 1 开始，长度为 3：

```python
x = data[1:4]  # [20, 30, 40]
y = data[2:5]  # [30, 40, 50]
```

输入和目标都是 `(B,T)`。长度为 T 的输入需要能访问连续 T+1 个 token，才能得到最后一个目标。

`np.random.randint(0, len(data)-T, size=B)` 的上限不包含在内，因此合法起点最大为 `len(data)-T-1`。

### 转成模型需要的张量

```python
x = torch.tensor(x, dtype=torch.long, device=device)
```

- 把 NumPy 数组转换为 PyTorch 张量。
- `torch.long` 是 int64，适合 token 索引和交叉熵标签。
- `device` 决定放 CPU 内存还是 GPU 显存；输入应与模型处于同一设备。

磁盘上可以用 uint16 节省空间，前提是 ID 在 0～65535；准备本批输入时再转成 long。

### mmap 的大白话

```python
data = np.load(path, mmap_mode="r")
```

不是先把整个文件塞进内存，而是用到哪部分，再按需访问对应的数据页。仍然使用内存，读过的数据也可能被操作系统缓存。

`r` 表示只读。`.npy` 文件包含形状和类型信息；如果文件是原始二进制数据，用 np.memmap 时则需要指定匹配的 dtype 等信息。

<a id="checkpoint"></a>

## 6. 保存与恢复检查点

作业要求保存三样东西，放在一个文件里：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
}
```

恢复顺序：创建结构匹配的模型 → 创建绑定其参数的优化器 → 加载两份状态 → 取得已完成步数 → 继续循环。

```python
for step in range(start_step, total_steps):
    ...
```

`total_steps=200` 表示累计目标 200 次，不是额外训练 200 次。

### iteration 与 t

- iteration：整体已完成多少次更新。
- AdamW 的 t：某个参数实际被更新多少次。

所有参数每轮都有梯度时通常相等；若某参数没有梯度被跳过，它的 t 就可能更小。

保存发生在更新后，因此已完成次数是 `step+1`。恢复时目标总步数不能小于存档步数，否则可能不执行训练却写入错误的进度。

当前检查点没有保存随机数状态，恢复不保证与不中断训练逐步完全一致。生成时只需要恢复模型状态，不需要恢复优化器。

<a id="training"></a>

## 7. 训练、验证与模式

训练流程：

```text
设置学习率 → 采样 x、y → 清除旧梯度
    → 模型前向 → 交叉熵 → backward
    → 梯度裁剪 → optimizer.step
    → 定期验证、记录日志、保存检查点
```

| 写法 | 作用 |
|---|---|
| `model.train()` | 设置模型和子模块为训练模式 |
| `model.eval()` | 设置为评估模式，相当于 `model.train(False)` |
| `torch.no_grad()` | 不记录用于反向传播的计算图 |
| `loss.backward()` | 计算梯度 |
| `optimizer.step()` | 更新参数 |

train 和 eval 影响 Dropout、BatchNorm 等层的行为，本身不开始训练，也不控制梯度记录。当前模型没有这些受模式影响的层，但仍保留标准流程。

验证时通常同时用 eval 和 no_grad，计算多个批次的平均损失，不调用 backward 或 step，结束后恢复原模式。

训练损失和验证损失不必相同：数据、平均批次数及计算时机都有区别。小幅波动正常。训练损失降低而验证损失恶化，可能提示过拟合。

人工数字循环和几句重复英语只验证流程，不足以证明对真实新文本的泛化能力。

<a id="arguments"></a>

## 8. 命令行参数

argparse 让实验配置不必写死在代码里：

```python
parser.add_argument("--batch-size", type=int, default=4)
```

```bash
uv run python -m scripts.train --batch-size 8 ...
```

`--batch-size` 会变成 `args.batch_size`。未提供的选项使用默认值，required=True 的选项必须填写。

```bash
uv run python -m scripts.train --help
```

当前支持模型结构、优化器、学习率调度、数据路径、设备、恢复路径、保存目录及保存间隔等参数。

模型配置需满足头数整除模型维度、RoPE 的头维度为偶数；数据 ID 小于词表大小。恢复时模型结构和调度配置要匹配原实验。

<a id="generation"></a>

## 9. 自回归生成

```text
输入 [0,1,2] → 生成 3
输入 [0,1,2,3] → 生成 4
继续将新 token 接回输入……
```

如果输入为 `(1,T)`，模型输出 logits 为 `(1,T,V)`。为了接在完整输入之后生成，只取最后一个位置：

```python
next_logits = logits[:, -1, :]  # (1,V)
```

然后 Softmax → 按概率抽样 → 拼接：

```python
next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
generated = torch.cat([generated, next_token], dim=-1)
```

multinomial 是按概率抽签，不保证总选最大概率候选。

`generated` 保存完整序列；`generated[:, -context_length:]` 只把末尾窗口送进模型，避免超出位置缓存。超出窗口的更早内容不参与本次预测。

停止条件：额外生成达到 max_new_tokens，或抽到指定的 eos_token_id。当前实现一次生成一条非空序列，返回值包含原 prompt 和结束符。

```python
was_training = model.training
model.eval()
# 生成……
model.train(was_training)
```

记录和恢复原来的模式。关闭梯度靠函数上的 `@torch.no_grad()`。

<a id="sampling"></a>

## 10. 温度与 top-p

### 温度 temperature

```python
probs = torch.softmax(next_logits / temperature, dim=-1)
```

- temperature=1：使用原分布。
- 小于 1：拉大分数差距，概率更集中。
- 大于 1：缩小分数差距，概率更平均。

当前实现要求温度大于零，不直接除以 0。温度只改变生成概率，不改变模型权重。

### top-p：保留累计概率达到阈值的候选

假设排序后的概率为：

```text
概率：    0.50, 0.25, 0.15, 0.07, 0.03
累计：    0.50, 0.75, 0.90, 0.97, 1.00
top_p：0.8
```

保留前三项，因为加上第三项才首次达到 0.8。剩余置零，并把保留部分重新归一化。top-p 不是固定保留 p 个，也不是保留单个概率大于 p 的 token。

核心过程：

```python
sorted_probs, sorted_indices = torch.sort(
    probs, dim=-1, descending=True,
)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

remove_mask = cumulative_probs >= top_p
remove_mask[..., 1:] = remove_mask[..., :-1].clone()
remove_mask[..., 0] = False
```

True 表示删除。右移删除标记，是为了保留首次达到阈值的那项。

第一项也要手动设为 False，因为右移赋值只修改位置 1 到末尾。如果最大概率已经超过阈值，原来第一项就是 True；不改就可能删光所有候选。

clone 先复制源数据，避免读写重叠存储。

### 排序位置不等于 token ID

```text
排序概率：[0.50, 0.25, 0.15, 0.07, 0.03]
原始 ID：[1,    3,    0,    4,    2]
```

若从排序概率里抽到位置 1，真实 ID 是 3，需要查询回来：

```python
sampled_position = torch.multinomial(filtered_probs, 1)
next_token = sorted_indices.gather(-1, sampled_position)
```

顺序：温度调整 → Softmax → top-p 筛选 → 归一化 → 采样位置 → 找回 ID。

<a id="text"></a>

## 11. 从 token ID 到文本

```text
文本 prompt → tokenizer.encode → token ID
    → 模型生成 → 完整 ID 列表 → tokenizer.decode → 文本
```

分词器与模型必须配套：相同 ID 要代表同一个 token，模型词表大小要覆盖所有 ID。词表大小相等也不意味着任意两个分词器可以互换。

数字演示模型只在 0～15 循环上训练，不适合直接接 BPE 文本词表。文本模型使用当前 263 大小的 BPE 词表重新训练。

当前 BPE 即使只在小语料上训练，也因含有完整基础字节而能编码其他文本，只是压缩效果有限。

特殊 token 的查询使用 bytes 键：

```python
eos_token_id = tokenizer.token_to_id[b"<|endoftext|>"]
```

文本演示已成功运行，示例输出为 `The cat is small.<|endoftext|>`。它反映小语料规律的学习，不代表已经具备通用语言能力。

<a id="logging"></a>

## 12. 实验记录与曲线

对应 §7.1 的 experiment_log，要求记录损失随训练步数和实际时间的变化，并保留实验尝试的说明。

### 每次运行保存什么

| 文件 | 内容 |
|---|---|
| `metrics_*.csv` | step、elapsed_seconds、train_loss、val_loss、lr |
| `metrics_*.config.json` | 本次启动解析的命令行设置 |
| `step_*.pt`、`final.pt` | 模型、优化器和已完成更新次数 |

`time.perf_counter()` 测量经过时间。`time.time_ns()` 用于区分本次运行的日志文件名。

CSV 用追加模式 a 写入新行。JSON 中 `vars(args)` 把参数对象转成字典，`indent=2` 便于阅读。

恢复训练时创建新日志：step 接着存档，耗时从本次启动重新计时。配置 JSON 记录命令行输入；优化器恢复后，部分实际设置可能已由存档覆盖，比较实验时需要注意。

### 学习曲线

- 横轴 step：看多少次更新达到某个损失。
- 横轴 elapsed_seconds：看多少时间达到某个损失。
- 纵轴 loss：同时画训练和验证曲线。

CSV 中读出的内容是字符串，需要 int/float 转换。Matplotlib 的 Agg 后端可以在 WSL 等没有绘图窗口的环境直接保存图片。

```bash
uv add --dev matplotlib
uv run python -m scripts.plot_learning_curve --log 实际CSV路径
```

画图脚本读取日志，输出同名 PNG。刚讲完并加入脚本，画图结果仍需用实际运行输出确认。

实验日志代码不等于完成正式实验：之后还要记录每次尝试、理由、结果，并比较曲线。

<a id="repository"></a>

## 13. 仓库使用与测试

| 目录 | 用途 |
|---|---|
| `cs336_basics/` | 核心计算实现 |
| `scripts/` | 数据准备、训练、画图入口 |
| `examples/` | 生成演示和独立知识点练习 |
| `notes/` | 中文笔记 |
| `artifacts/` | 数据、词表、检查点、日志、图片 |
| `tests/` | 作业测试和 adapters |

在项目根目录运行模块命令：

```bash
uv run python -m scripts.train --help
uv run python -m scripts.prepare_text_data
uv run python -m examples.generate_text
uv run pytest
```

核心实现文件一般用于 import，不需要逐个直接执行。更多命令、数据配套关系和移动对照表见 [仓库导航](../LEARNING_GUIDE.md)。

### 今天遇到的测试问题

- tiktoken 的 SSL 报错来自参考分词器下载资源；本次运行已成功下载和缓存，但不能由此断定所有网络环境的证书问题都永久解决。
- SiLU 已实现但测试接口未连接，现已补上 `run_silu()`。
- 全量测试最后记录为 46 passed、1 xfailed、1 failed；失败项是已有 BPE 速度测试，耗时约 1.503 秒，略超 1.5 秒上限，尚未优化。
- XFAIL 是预期失败标记；XPASS 表示原来标记为预期失败的测试这次反而通过，不等于新增的实现失败。

<a id="progress"></a>

## 14. 当前进度与后续任务

已练习：AdamW、资源估算、学习率调度、梯度裁剪、数据加载、检查点、训练验证、命令行配置、文本生成、温度和 top-p。

当前学习位置：第 7.1 节 experiment_log，已讲解 CSV 日志、配置 JSON 和曲线脚本；正式实验尚未完成。

后续：

1. 实际生成并查看损失曲线，补好实验记录文档。
2. 进入 §7.2 TinyStories，准备真实语料与配套分词器。
3. 按讲义进行学习率等超参数实验，再开展后续对比实验。
4. 真实语料 BPE 实验之前主动跳过的部分，仍需补做；小数据演示没有替代这些任务。

每天复习可以沿这条主线：

```text
文本 → 分词 → token 数据 → 随机批次
    → Transformer → 损失 → 梯度 → AdamW 更新
    → 验证与实验记录 → 保存模型
    → 输入 prompt → 逐 token 采样 → 解码为文本
```
