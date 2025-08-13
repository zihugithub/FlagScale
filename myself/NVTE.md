# NVTE
NVTE (全称 NeMo Megatron (NM) / NeMo Tensor Core (NvTe))

## NVTE_APPLY_QK_LAYER_SCALING
该参数用于控制 NVTE 框架下注意力机制行为
- 0 → 禁用 query-key layer scaling: 沿用传统的注意力计算方式，不进行额外缩放
- 1 → 启用 query-key layer scaling: 系统会自动为每个注意力层添加这样的可训练参数
```
Query-Key Layer Scaling 是一种针对自注意力模块的内部归一化技术，用于动态调整不同层的激活强度，以改善训练稳定性和收敛速度

- 具体来说，它会对每个Transformer层的 Q×Kᵀ 矩阵乘积结果进行额外的缩放操作
- 标准的Transformer架构中，注意力分数由以下公式计算得出：
  - Attention Score = softmax(Q·K^T / √dₖ)
  -  dₖ 是键向量维度, 随着模型深度增加，多层堆叠可能导致数值不稳定或梯度消失等问题
- "LayerScale" 初始化策略应用于注意力层，即引入可学习的标量参数 αᵢ（每层独立），使得实际使用的相似度变为：
  - Scaled Attention Score = αᵢ · softmax(Q·K^T / √dₖ)
```