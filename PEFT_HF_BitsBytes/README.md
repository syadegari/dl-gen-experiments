# Fine-Tuning of GPT-2 for Sentiment Analysis

This repository showcases multiple methods for fine-tuning language models, using **GPT-2** as a base model for binary sentiment classification on the [IMDB dataset](https://huggingface.co/datasets/imdb): 

- **Base GPT-2**: Fine-tunes only the final classification head.
- **LoRA-GPT2**: Fine-tunes using Low-Rank Adaptation (LoRA) for efficient parameter tuning.
- **Quantized LoRA-GPT2 with Checkpointing**: Combines LoRA with 4-bit quantization (using BitsAndBytes) and gradient checkpointing to reduce memory footprint and allow for larger batch sizes or deeper models.

---

## Table of Contents and Structure

- [`project_peft.ipynb`](./project_peft.ipynb) – The main notebook that includes all code and experiments
- [Imports and Setup](project_peft.ipynb#imports-and-setup)
- [Load and Tokenize IMDB Dataset](project_peft.ipynb#load-and-tokenize-imdb-dataset)
- [Define Evaluation Metrics](project_peft.ipynb#define-evaluation-metrics)
- [Evaluate Pretrained GPT-2 Classifier](project_peft.ipynb#evaluate-pretrained-gpt-2-classifier)
  - [Some helper functions](project_peft.ipynb#some-helper-functions)
  - [Prepare Tokenizer, Tokenized Datasets and Models](project_peft.ipynb#prepare-tokenizer-tokenized-datasets-and-models)
    - [Base Model with the classifier layer](project_peft.ipynb#base-model-with-the-classifier-layer)
    - [LoRA](project_peft.ipynb#lora)
    - [LoRA with Quantization](project_peft.ipynb#lora-with-quantization)
- [Some Notes and Explanation of LoRA and Quantized Model Architectures](project_peft.ipynb#some-notes-and-explanation-of-lora-and-quantized-model-architectures)
- [Training and Evaluation](project_peft.ipynb#training-and-evaluation)
    - [Training Base Model (Only the Classifier Layer)](project_peft.ipynb#training-base-model-only-the-classifier-layer)
    - [Training LoRA Model](project_peft.ipynb#training-lora-model)
    - [Training Quantized-LoRA Model](project_peft.ipynb#training-quantized-lora-model)
- [Conclusions/Suggestions](project_peft.ipynb#conclusionssuggestions)


- [`saved_f32_lora/`](./saved_f32_lora/) – Saved weights of the LoRA fine-tuned model. 
- [`saved_nf4_lora/`](./saved_nf4_lora/) – Saved weights of the quantized LoRA model.

## Experiments

Three variants are trained and evaluated:

| Model                          | Accuracy | Notes |
|-------------------------------|----------|-------|
| Base GPT-2                    | ~83%     | Only classifier head trained |
| LoRA-GPT2                     | ~90%     | 600k+ trainable params via LoRA |
| Quantized LoRA w/ Checkpoint  | ~88%     | 4-bit quantized weights, efficient memory |

---

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
