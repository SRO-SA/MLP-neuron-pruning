import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:  # laptop dry-run environment
    torch = nn = None


@unittest.skipIf(torch is None, "torch is only available in the GPU/server environment")
class HeterogeneousCheckpointTests(unittest.TestCase):
    def _model(self):
        from types import SimpleNamespace

        class Packed(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.arange(2 * 8 * 3).reshape(2, 8, 3).float())
                self.down_proj = nn.Parameter(torch.arange(2 * 3 * 4).reshape(2, 3, 4).float())
                self.intermediate_size = 4

        class MLP(nn.Module):
            def __init__(self):
                super().__init__(); self.experts = Packed(); self.intermediate_size = 4

        class Layer(nn.Module):
            def __init__(self):
                super().__init__(); self.mlp = MLP()

        class Toy(nn.Module):
            def __init__(self):
                super().__init__(); self.model = nn.Module(); self.model.layers = nn.ModuleList([Layer()])

        return Toy()

    def test_packed_plan_is_physically_sliced_without_padding(self):
        from src.heterogeneous_moe_checkpoint import apply_plan_physical, inspect_plan_shapes
        model = self._model()
        plan = {"layers": [{"layer_idx": 0, "old_intermediate": 4,
                            "prune_idx": [1]}]}
        audit = apply_plan_physical(model, plan)
        self.assertEqual(audit["removed_layer_channels"], 1)
        self.assertEqual(audit["removed_expert_neurons"], 2)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(tuple(experts.gate_up_proj.shape), (2, 6, 3))
        self.assertEqual(tuple(experts.down_proj.shape), (2, 3, 3))
        self.assertTrue(inspect_plan_shapes(model, plan)[0]["no_original_width_padding"])

    def test_invalid_index_fails_before_mutation(self):
        from src.heterogeneous_moe_checkpoint import apply_plan_physical
        with self.assertRaises(ValueError):
            apply_plan_physical(self._model(), {"layers": [{
                "layer_idx": 0, "old_intermediate": 4, "prune_idx": [4]
            }]})

    def test_auto_dispatch_keeps_decoder_and_moe_classes_atomic(self):
        from src.heterogeneous_moe_checkpoint import dispatch_no_split_module_classes

        model = self._model()
        model._no_split_modules = ["DeclaredAtomicLayer"]
        classes = dispatch_no_split_module_classes(model)
        self.assertIn("DeclaredAtomicLayer", classes)
        self.assertIn(model.model.layers[0].__class__.__name__, classes)
        self.assertIn(model.model.layers[0].mlp.__class__.__name__, classes)

    def test_execution_device_audit_accepts_colocated_moe(self):
        from src.heterogeneous_moe_checkpoint import inspect_plan_execution_devices

        audit = inspect_plan_execution_devices(self._model(), {"layers": [{
            "layer_idx": 0, "old_intermediate": 4, "prune_idx": [1]
        }]})
        self.assertEqual(audit, [{"layer_idx": 0, "mlp_devices": ["cpu"]}])


if __name__ == "__main__":
    unittest.main()
