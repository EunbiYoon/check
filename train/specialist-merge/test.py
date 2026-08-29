from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from merge_adapters import merge_lora_adapters


def _write_adapter(path: Path, tensors: dict[str, torch.Tensor], config: str | None = "{}") -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    if config is not None:
        (path / "adapter_config.json").write_text(config, encoding="utf-8")


class SpecialistMergeTests(unittest.TestCase):
    def test_weighted_sum_and_config_copied(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            aux = {"lora_A": torch.ones(2, 2), "lora_B": torch.full((2, 2), 4.0)}
            all_ = {"lora_A": torch.zeros(2, 2), "lora_B": torch.full((2, 2), 8.0)}
            _write_adapter(root / "aux", aux, config='{"peft_type": "LORA"}')
            _write_adapter(root / "all", all_, config='{"peft_type": "LORA"}')

            out = merge_lora_adapters(root / "aux", root / "all", root / "merge", alpha=0.25)

            merged = load_file(str(out / "adapter_model.safetensors"))
            # 0.25*1 + 0.75*0 = 0.25 ; 0.25*4 + 0.75*8 = 7.0
            self.assertTrue(torch.allclose(merged["lora_A"], torch.full((2, 2), 0.25)))
            self.assertTrue(torch.allclose(merged["lora_B"], torch.full((2, 2), 7.0)))
            self.assertEqual(
                (out / "adapter_config.json").read_text(encoding="utf-8"),
                '{"peft_type": "LORA"}',
            )

    def test_key_only_in_aux_is_passed_through(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            _write_adapter(root / "aux", {"shared": torch.ones(1), "aux_only": torch.full((1,), 3.0)})
            _write_adapter(root / "all", {"shared": torch.full((1,), 5.0)})

            out = merge_lora_adapters(root / "aux", root / "all", root / "merge", alpha=0.5)
            merged = load_file(str(out / "adapter_model.safetensors"))

            self.assertTrue(torch.allclose(merged["shared"], torch.full((1,), 3.0)))
            self.assertTrue(torch.allclose(merged["aux_only"], torch.full((1,), 3.0)))

    def test_missing_safetensors_raises(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            (root / "aux").mkdir()
            _write_adapter(root / "all", {"shared": torch.ones(1)})
            with self.assertRaises(FileNotFoundError):
                merge_lora_adapters(root / "aux", root / "all", root / "merge")


if __name__ == "__main__":
    unittest.main()
