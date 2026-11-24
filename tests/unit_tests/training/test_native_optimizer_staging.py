"""Compatibility of Bridge's checkpoint option with native Core CPU staging."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.training import config as config_module
from megatron.bridge.training.config import (
    _sync_optimizer_checkpoint_staging,
    _validate_and_sync_distributed_optimizer_settings,
)
from megatron.bridge.training.optim import memory_efficient_precision_aware_optimizer_state_checkpointing


def _config(*, requested=True, native=False):
    return SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer="adam",
            use_distributed_optimizer=True,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
            optimizer_cpu_offload=False,
            use_layer_wise_distributed_optimizer=False,
            stage_precision_aware_optimizer_state_on_cpu=native,
        ),
        ddp=SimpleNamespace(
            use_megatron_fsdp=False,
            use_distributed_optimizer=True,
            num_distributed_optimizer_instances=1,
            overlap_param_gather=False,
        ),
        dist=SimpleNamespace(use_torch_fsdp2=False),
        checkpoint=SimpleNamespace(stage_precision_aware_optimizer_state_on_cpu=requested, ckpt_format="torch_dist"),
    )


@pytest.fixture
def native_core(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "MCoreOptimizerConfig",
        SimpleNamespace(stage_precision_aware_optimizer_state_on_cpu=False),
    )


@pytest.mark.parametrize(
    "requested,native,expected", [(False, False, False), (True, False, True), (False, True, True)]
)
def test_translates_legacy_option_and_preserves_native_opt_in(native_core, requested, native, expected):
    cfg = _config(requested=requested, native=native)
    _sync_optimizer_checkpoint_staging(cfg)
    _sync_optimizer_checkpoint_staging(cfg)
    assert cfg.optimizer.stage_precision_aware_optimizer_state_on_cpu is expected


@pytest.mark.parametrize("mode", ["ordinary", "offload", "layerwise", "fsdp", "fsdp2", "no_dist", "muon"])
def test_preserves_legacy_noop_modes(native_core, mode):
    cfg = _config()
    if mode == "ordinary":
        cfg.optimizer.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
    elif mode == "offload":
        cfg.optimizer.optimizer_cpu_offload = True
    elif mode == "layerwise":
        cfg.optimizer.use_layer_wise_distributed_optimizer = True
    elif mode == "fsdp":
        cfg.ddp.use_megatron_fsdp = True
    elif mode == "fsdp2":
        cfg.dist.use_torch_fsdp2 = True
    elif mode == "no_dist":
        cfg.optimizer.use_distributed_optimizer = False
    else:
        cfg.optimizer.optimizer = "muon"
    _sync_optimizer_checkpoint_staging(cfg)
    assert not cfg.optimizer.stage_precision_aware_optimizer_state_on_cpu


def test_old_core_does_not_receive_phantom_field(monkeypatch):
    monkeypatch.setattr(config_module, "MCoreOptimizerConfig", SimpleNamespace())
    cfg = _config()
    del cfg.optimizer.stage_precision_aware_optimizer_state_on_cpu
    _sync_optimizer_checkpoint_staging(cfg)
    assert not hasattr(cfg.optimizer, "stage_precision_aware_optimizer_state_on_cpu")


@pytest.mark.parametrize("mode", ["torch", "fsdp", "fsdp2"])
def test_native_option_validates_checkpoint_contract(native_core, mode):
    cfg = _config(requested=False, native=True)
    if mode == "torch":
        cfg.checkpoint.ckpt_format = "torch"
    elif mode == "fsdp":
        cfg.ddp.use_megatron_fsdp = True
    else:
        cfg.dist.use_torch_fsdp2 = True
    with pytest.raises(ValueError, match="torch_dist without FSDP"):
        _sync_optimizer_checkpoint_staging(cfg)


def test_real_optimizer_config_uses_native_staging():
    if not hasattr(OptimizerConfig, "stage_precision_aware_optimizer_state_on_cpu"):
        pytest.skip("Installed Core predates native CPU staging")
    cfg = _config()
    cfg.optimizer = OptimizerConfig(
        optimizer="adam",
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        bf16=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    _validate_and_sync_distributed_optimizer_settings(cfg)
    cfg.optimizer.validate()
    assert cfg.optimizer.stage_precision_aware_optimizer_state_on_cpu


@pytest.mark.run_only_on("gpu")
@pytest.mark.parametrize("legacy_option", [False, True])
def test_native_core_exports_cpu_state_without_patching_te(legacy_option):
    from megatron.core.optimizer import DistributedOptimizer

    if not hasattr(OptimizerConfig, "stage_precision_aware_optimizer_state_on_cpu"):
        pytest.skip("Installed Core predates native CPU staging")
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers").FusedAdam
    param = torch.nn.Parameter(torch.arange(32, device="cuda", dtype=torch.bfloat16) / 32)
    kwargs = dict(
        lr=0.001,
        master_weights=True,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
        use_decoupled_grad=True,
        store_param_remainders=False,
    )
    inner = fused_adam([param], **kwargs)
    for iteration in range(2):
        param.decoupled_grad = torch.full_like(param, (iteration + 1) * 0.125)
        inner.step()
    wrapper = DistributedOptimizer.__new__(DistributedOptimizer)
    wrapper.optimizer = inner
    wrapper.config = OptimizerConfig(stage_precision_aware_optimizer_state_on_cpu=True)
    original = inner.get_unscaled_state
    with memory_efficient_precision_aware_optimizer_state_checkpointing(wrapper, enabled=legacy_option) as patched:
        assert patched == 0
        portable = wrapper._get_inner_state_dict()
        for key, value in portable["state"][0].items():
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, original(param, key).cpu(), rtol=0, atol=0)
        assert inner.get_unscaled_state == original
        assert "get_unscaled_state" not in inner.__dict__

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = fused_adam([restored_param], **kwargs)
    restored.load_state_dict(portable)
    for iteration in range(3):
        for weight, optimizer in ((param, inner), (restored_param, restored)):
            weight.decoupled_grad = torch.full_like(weight, (iteration + 3) * -0.0625)
            optimizer.step()
        torch.testing.assert_close(param, restored_param, rtol=0, atol=0)
        for key in inner.state[param]:
            torch.testing.assert_close(
                inner.get_unscaled_state(param, key),
                restored.get_unscaled_state(restored_param, key),
                rtol=0,
                atol=0,
            )
        assert inner.param_groups[0]["step"] == restored.param_groups[0]["step"]
