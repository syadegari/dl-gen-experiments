# Fine-Tuning of GPT-2 for Sentiment Analysis

This repository showcases multiple methods for fine-tuning language models, using **GPT-2** as a base model for binary sentiment classification on the [IMDB dataset](https://huggingface.co/datasets/imdb): 

- **Base GPT-2**: Fine-tunes only the final classification head.
- **LoRA-GPT2**: Fine-tunes using Low-Rank Adaptation (LoRA) for efficient parameter tuning.
- **Quantized LoRA-GPT2 with Checkpointing**: Combines LoRA with 4-bit quantization (using BitsAndBytes) and gradient checkpointing to reduce memory footprint and allow for larger batch sizes or deeper models.

---

## Table of Contents and Structure

- [`project_peft.ipynb`](./project_peft.ipynb) – The notebook including all code and experiments
- Imports and Setup
- Load and Tokenize IMDB Dataset
- Define Evaluation Metrics
- Evaluate Pretrained GPT-2 Classifier
  - Some helper functions
  - Prepare Tokenizer, Tokenized Datasets and Models
    - Base Model with the classifier layer
    - LoRA
    - LoRA with Quantization
- Some Notes and Explanation of LoRA and Quantized Model Architectures
- Training and Evaluation
    - Training Base Model (Only the Classifier Layer)
    - Training LoRA Model
    - Training Quantized-LoRA Model
- Conclusions/Suggestions


- [`saved_f32_lora/`](./saved_f32_lora/) – Saved weights of the LoRA fine-tuned model. 
- [`saved_nf4_lora/`](./saved_nf4_lora/) – Saved weights of the quantized LoRA model.

## Experiments

Three variants are trained and evaluated:

| Model                          | Accuracy | Notes |
|-------------------------------|----------|-------|
| Base GPT-2 (`model_f32`)                    | ~83%     | Only classifier head trained |
| LoRA-GPT2 (`model_f32_lora`)                     | ~90%     | 600k+ trainable params via LoRA |
| Quantized LoRA w/ Checkpoint (`model_nf4_with_chkpoint`)  | ~88%     | 600k+ trainable params plus 4-bit quantized weights, efficient memory |

---

## Some Notes and Explanation of LoRA and Quantized Model Architectures

In the above mentioned experiments, we prepared 3 different models for training:

- `model_f32` is the vanilla GPT2 with a classifier head. It contains only a handful of trainable parameters since we only have two categories and the output is `768` dimensional (about `1.5 K` parameters)
- `model_f32_lora` is the LoRA adjust models. This models has about `600_000` trainable parameters since we use a low rank approximation of dimension `4` 

```python
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    fan_in_fan_out=True,
)
```

which is then attached to all the frozen modules in both attention and mlp layers in all the block (GPT2 has 12 block in total). 

One can inspect one of these blocks more closely

```python

>>> block_lora_f32 = model_f32_lora.base_model.model.transformer.h[0].attn.c_attn
>>> block_lora_f32

lora.Linear(
  (base_layer): Conv1D(nf=2304, nx=768)
  (lora_dropout): ModuleDict(
    (default): Dropout(p=0.1, inplace=False)
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=768, out_features=4, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=4, out_features=2304, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)
```

with `lora_A` and `lora_B` being the low rank approximation, in this case, of 4. These `lora_A` and `lora_B` matrices are the trainable parameters of the model.

- `model_nf4_with_chkpoint` uses quantization as well as LoRA, plus recomputing the intermediate activations to save memory (called checkpointing), thus enabling using larger batch sizes and/or larger models. This stores certain layer weights in 4-bit and it appears that the number of parameters are reduces, but the reduction stems from the fact that under the hood, weights are stored in 8-bit variables (so two 4-bit parameters can be packed inside a single 8-bit bit parameter):

```python
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

 Similar to LoRA model, one can peek into individual attention heads to see how the quantization is performed (not all the details but at least gain some insight into the workings of the bitsandbytes module):

```python
>>> block_nf4 = model_nf4_lora_with_chkpoint.base_model.model.transformer.h[0].attn.c_attn
>>> block_nf4

lora.Linear4bit(
  (base_layer): Linear4bit(in_features=768, out_features=2304, bias=True)
  (lora_dropout): ModuleDict(
    (default): Dropout(p=0.1, inplace=False)
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=768, out_features=4, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=4, out_features=2304, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
  (lora_magnitude_vector): ModuleDict()
)
```

The LoRA, trainable parameters, remain 32-bit:

```python
>>> for param in block_nf4.lora_A.parameters():
...    print(param.shape, param.dtype)

torch.Size([4, 768]) torch.float32
```

whereas the non-trainable parameters have been quantized to 4bit values:

```python
>>> for param in block_nf4.base_layer.parameters():
...    print(param.shape, param.dtype)

torch.Size([884736, 1]) torch.uint8
torch.Size([2304]) torch.float32
```

interestingly, the bias term remains 32-bit where the matrix weights have been quantized and packed into a 1D tensor. I am not sure why this has been done like this (in terms of the 1D storation of the tensor). The number of parameters and the dtype are consistent with 4-bit quantization, since we originally had `[768, 2304]` float-32 parameters. That's `1769472` parameters, and in 4-bit, we can store 2 of such parameters inside a byte (uint8), thus halving the number of parameters in each quantized tensor, matching the `[884736, 1]` shape.

## Coclusions and Suggestions

- Both LoRA and Quantized models achieved higher accuracy than the base model on the test set, 90% and 88% respectively versus the accuracy of 83% for the base model. 
- Quantized model with checkpointing, `model_nf4_lora_with_chkpoint`, can handle bigger batch sizes or a bigger model. It is better to use an even larger model in this case and compare it to GPT2-LoRA. 
- Comparing the three presented variants is non-trivial. While all three are trained on the same number of epochs, a better and more involved approach would be to match them by compute budget (GPU hours and memory).

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```
