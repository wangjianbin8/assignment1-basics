import torch

class CrossEntropyLoss:  # 计算交叉熵损失
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def forward(self, inputs, targets):

        '''
        inputs: logits，形状 [B, V]
        targets: 正确类别，形状 [B]
        '''

        # log_probs = torch.log(softmax(self.inputs, -1))

        '''
        不能先 softmax 再 log：
        softmax 后很小的概率可能变成 0
        log(0) = -inf，导致数值溢出
        '''

        '''
        稳定计算 log(softmax(x))：
        log_softmax(x) = x - logsumexp(x)
        '''

        log_probs = self.inputs - torch.logsumexp(
            self.inputs,
            dim=-1,
            keepdim=True
        )

        '''
        log_probs: [B, V]
        每一行都是该样本所有类别的 log 概率

        torch.arange(B)：
        选择第 0、1、2...个样本

        self.targets：
        选择每个样本对应的正确类别

        最终得到每个样本的正确类别 log 概率
        '''

        target_log_probs = log_probs[
            torch.arange(self.inputs.shape[0]),
            self.targets
        ]

        '''
        交叉熵：
        CE = -log(P(correct))

        对所有样本取平均
        '''

        return -torch.mean(target_log_probs)