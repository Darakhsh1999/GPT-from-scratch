# GPT-from-scratch

Implementing Generative Pretrained Transformer (GPT) from scratch with limited external ML libraries usage:
- PyTorch for training loop and model architecture
- tiktoken for GPT2 tokenizer

## Similarities to OpenAI's GPT
- Flash attention
- Same tokenizer
- GELU activation functions
- LayerNorm before each sublayer in the transformer blocks
- AdamW optimizer

## Main differences to OpenAI's GPT
- Fixed learning rate with not decay
- No accumulated gradients, each batch corresponds to a step
- My dataset is limited to a single .txt file for text data

## Model Architecture

```
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
├─ModuleDict: 1                                         []                        --
|    └─Embedding: 2-1                                   [-1, 768]                 786,432
|    └─Embedding: 2-2                                   [-1, 100, 768]            38,597,376
|    └─ModuleList: 2                                    []                        --
|    |    └─TransformerDecoderBlock: 3-1                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-2                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-3                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-4                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-5                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-6                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-7                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-8                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-9                [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-10               [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-11               [-1, 100, 768]            7,087,872
|    |    └─TransformerDecoderBlock: 3-12               [-1, 100, 768]            7,087,872
|    └─LayerNorm: 2-3                                   [-1, 100, 768]            1,536
├─Linear: 1-1                                           [-1, 100, 50257]          38,597,376
=========================================================================================================
Total params: 163,037,184
Trainable params: 163,037,184
Non-trainable params: 0
Total mult-adds (M): 247.89
=========================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 53.58
Params size (MB): 621.94
Estimated Total Size (MB): 675.52
=========================================================================================================
```