# Transformer LM：整体流程与参数量计算

## 1. 整体流程

设输入批大小为 $B$，序列长度为 $T$，词表大小为 $V$，模型维度为 $d$，Transformer Block 数量为 $L$。

```text
token ID                         (B, T)
   ↓ Embedding：查表转换成向量
隐藏向量                         (B, T, d)
   ↓ 连续经过 L 个 Transformer Block
隐藏向量                         (B, T, d)
   ↓ 最终 RMSNorm
归一化后的向量                   (B, T, d)
   ↓ LM Head：线性投影到词表大小
logits                           (B, T, V)
```

每个位置的 logits，表示该位置对**下一个 token**的预测分数。模型直接输出 logits；需要概率时再使用 Softmax。

## 2. 一个 Transformer Block 的流程

当前实现采用 Pre-Norm 结构：

$$
z = x + \operatorname{Attention}(\operatorname{RMSNorm}_1(x))
$$

$$
y = z + \operatorname{SwiGLU}(\operatorname{RMSNorm}_2(z))
$$

### 注意力内部

```text
输入 → 三个 Linear，生成 Q、K、V
     → 拆成多个头
     → 对 Q、K 应用 RoPE
     → 计算缩放点积分数，应用因果 mask
     → Softmax 得到权重，对 V 加权求和
     → 合头 → 输出 Linear
```

因果 mask 保证每个位置只能关注自己及之前的位置。

### SwiGLU 内部

$$
\operatorname{SwiGLU}(x)
= W_2\bigl(\operatorname{SiLU}(W_1x)\odot W_3x\bigr)
$$

其中，$\odot$ 表示逐元素乘法。

各位置共享同一套 FFN 参数，FFN 本身不混合不同位置的信息。

## 3. 参数量计算

记：

- $V$：词表大小。
- $d$：模型维度。
- $f$：FFN 中间维度。
- $L$：Block 数量。

当前实现**没有偏置，Embedding 与 LM Head 不共享权重**。

| 组件 | 权重形状或构成 | 参数量 |
| --- | --- | --- |
| Token Embedding | $(V,d)$ | $Vd$ |
| 一个 Block 的注意力 | Q、K、V、输出投影，各为 $(d,d)$ | $4d^2$ |
| 一个 Block 的 SwiGLU | 两个 $(f,d)$，一个 $(d,f)$ | $3df$ |
| 一个 Block 的两个 RMSNorm | 两个长度为 $d$ 的增益向量 | $2d$ |
| 最终 RMSNorm | 长度为 $d$ 的增益向量 | $d$ |
| LM Head | $(V,d)$ | $Vd$ |

因此，单个 Block 的参数量为：

$$
\boxed{P_{\text{block}} = 4d^2 + 3df + 2d}
$$

完整模型的参数量为：

$$
\boxed{P_{\text{total}} = 2Vd + L(4d^2 + 3df + 2d) + d}
$$

注意：

- 注意力参数量**不再乘头数**，各头已包含在投影矩阵中。
- RoPE、Softmax、残差加法、mask 没有可训练参数。
- 参数量不随批大小 $B$ 或序列长度 $T$ 改变。
