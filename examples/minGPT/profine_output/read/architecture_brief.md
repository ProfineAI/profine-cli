## Architecture summary

This script trains a **character-level GPT language model** using PyTorch and the `mingpt` package (imports at `chargpt.py:8-14`). The model is instantiated as `GPT(config.model)` and trained via `Trainer(config.trainer, model, train_dataset)` (`chargpt.py:102`, `chargpt.py:105`).

### Data pipeline
- The dataset is `CharDataset`, a custom `torch.utils.data.Dataset` that tokenizes raw text at the character level (`chargpt.py:42-82`).
- It reads the full corpus from `input.txt` (`chargpt.py:96`) and builds a vocabulary from the unique characters in that file (`chargpt.py:56-63`).
- The context length / block size is **128** (`chargpt.py:50`, `chargpt.py:101`).

### Model
- The model family is **GPT** (`chargpt.py:12`, `chargpt.py:31`).
- The script sets `model_type = 'gpt-mini'` (`chargpt.py:32`), but the exact internal depth/width/head counts are **not specified in this file** and likely come from `mingpt.model.GPT` defaults. Those fields are therefore marked as guessed/unknown in the record.
- Attention is inferred to be **causal multi-head self-attention** because this is an autoregressive GPT LM, but the exact attention implementation is not visible here.

### Training
- The trainer learning rate is **5e-4** (`chargpt.py:36`).
- No explicit optimizer construction is shown in this script; the optimizer type is inferred from `mingpt.Trainer` defaults and should be treated cautiously.
- No scheduler, mixed precision, gradient scaler, compile, or gradient checkpointing is configured in this file (`chargpt.py:35`, `chargpt.py:133`).

### Runtime behavior
- The script seeds randomness with `3407` and logs to `./out/chargpt` (`chargpt.py:24-25`, `chargpt.py:92-93`).
- Every 500 iterations it samples 500 tokens from the model starting from the prompt `"O God, O God!"` and saves a checkpoint to `./out/chargpt/model.pt` (`chargpt.py:113-126`).

### Caveats
- Several architecture fields are **guessed** because they are not explicitly defined in this script and depend on `mingpt` internals (notably `num_layers`, `num_heads`, `hidden_size`, optimizer type, and possibly batch size / gradient accumulation).