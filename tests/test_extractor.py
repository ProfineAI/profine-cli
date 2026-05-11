"""Tests for the AST fact extractor.

The extractor is the deterministic input layer to the LLM analyzer —
its observations need to be exact (no hallucination, no missed calls).
"""

from __future__ import annotations

from profine.reader.extractor import extract


def test_extract_minimal_source():
    facts = extract("x = 1\n", "test.py")
    assert facts.file_path == "test.py"
    assert facts.total_lines == 1


def test_imports_are_categorized():
    src = "import torch\nfrom transformers import AutoModel\n"
    facts = extract(src)
    assert any("torch" in i.module for i in facts.imports)
    assert any(i.style == "from" for i in facts.imports)


def test_optimizer_call_classified():
    src = "import torch\noptimizer = torch.optim.Adam(params, lr=1e-3)\n"
    facts = extract(src)
    assert any("Adam" in c.name for c in facts.optimizer_calls)


def test_dataloader_call_classified():
    src = "from torch.utils.data import DataLoader\nloader = DataLoader(ds, batch_size=64)\n"
    facts = extract(src)
    assert any("DataLoader" in c.name for c in facts.dataloader_calls)


def test_model_loader_classified():
    src = "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('gpt2')\n"
    facts = extract(src)
    assert any("from_pretrained" in c.name for c in facts.model_loader_calls)


def test_loss_call_classified():
    src = "import torch.nn as nn\nloss_fn = nn.CrossEntropyLoss()\n"
    facts = extract(src)
    assert any("CrossEntropyLoss" in c.name for c in facts.loss_calls)


def test_compile_call_classified():
    src = "import torch\nmodel = torch.compile(model)\n"
    facts = extract(src)
    assert any("compile" in c.name for c in facts.compile_calls)


def test_autocast_call_classified():
    src = "import torch\nwith torch.autocast('cuda', dtype=torch.bfloat16):\n    pass\n"
    facts = extract(src)
    assert any("autocast" in c.name for c in facts.autocast_calls)


def test_distributed_call_classified():
    src = "import torch.distributed as dist\ndist.init_process_group('nccl')\n"
    facts = extract(src)
    assert any("init_process_group" in c.name for c in facts.distributed_calls)


def test_classes_and_functions_are_recorded():
    src = "class Foo:\n    pass\n\ndef bar():\n    pass\n"
    facts = extract(src)
    assert any(c.name == "Foo" for c in facts.classes)
    assert any(f.name == "bar" for f in facts.functions)


def test_line_numbers_are_tracked():
    src = "import os\nimport sys\n"
    facts = extract(src)
    by_module = {i.module: i.line for i in facts.imports}
    assert by_module["os"] == 1
    assert by_module["sys"] == 2


def test_empty_source_does_not_crash():
    facts = extract("")
    assert facts.total_lines == 0
    assert facts.imports == []
