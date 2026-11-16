"""Architecture registry.

Adding an architecture is two edits: a new file in this package, and one line
in `ARCHS`. Everything else -- CLI, training, evaluation, generation and the
contract test suite -- picks it up automatically.
"""

import inspect
from typing import Any, Dict, Type

from model.device import quiet_tensorflow_import
from model.interfaces import BaseLM
import model.schedules  # noqa: F401  (registers LinearWarmup, saved in checkpoints' optimizer config)

quiet_tensorflow_import()  # keras_hub (ALiBi archs) imports TensorFlow, which logs from C++ on import

from model.archs.dual_sphere import DualSphereLM  # noqa: E402
from model.archs.dual_sphere_dual_readout import DualSphereDualReadoutLM
from model.archs.dual_sphere_mid_gate import DualSphereMidGateLM
from model.archs.dual_sphere_hyper_token_transform import DualSphereHyperTokenTransformLM
from model.archs.dual_sphere_hyper_gate import DualSphereHyperGateLM
from model.archs.dual_sphere_householder import DualSphereHouseholderLM
from model.archs.dual_sphere_householder_per_token import DualSphereHouseholderPerTokenLM
from model.archs.dual_sphere_householder_rank8 import DualSphereHouseholderRank8LM
from model.archs.dual_sphere_linear_recurrence import DualSphereLinearRecurrenceLM
from model.archs.sine_sphere import SineSphereLM
from model.archs.alibi_sphere import AlibiSphereLM
from model.archs.alibi_sphere_two_block import AlibiSphereTwoBlockLM
from model.archs.alibi_retriever_sphere import AlibiRetrieverSphereLM
from model.archs.alibi_token_retriever_sphere import AlibiTokenRetrieverSphereLM
from model.archs.alibi_token_retriever_sphere_bidirectional import AlibiTokenRetrieverSphereBidirectionalLM
from model.archs.alibi_token_retriever_sphere_bidirectional_raw import AlibiTokenRetrieverSphereBidirectionalRawLM
from model.archs.ngpt import NGPTLM
from model.archs.simple_gpt import SimpleGPTLM
from model.archs.rope_logit_sum import RopeLogitSumLM
from model.archs.rope_log_gaussian_sum import RopeLogGaussianSumLM
from model.archs.rope_kernel_attention import RopeKernelAttentionLM
from model.archs.rope_kernel_hops import RopeKernelHopsLM
from model.archs.rope_token_kernel import RopeTokenKernelLM
from model.archs.rope_token_banks import RopeTokenBanksLM
from model.archs.wave_arrow import WaveArrowLM
from model.archs.rope_state_memory import RopeStateMemoryLM
from model.archs.wave_field import WaveFieldLM
from model.archs.min_gru import MinGRULM
from model.archs.min_gru_cuda import MinGRUCudaLM
from model.archs.gated_deltanet import GatedDeltaNetLM
from model.archs.lru_banks import LRUBanksLM

ARCHS: Dict[str, Type[BaseLM]] = {
    "dual-sphere": DualSphereLM,
    "dual-sphere-dual-readout": DualSphereDualReadoutLM,
    "dual-sphere-mid-gate": DualSphereMidGateLM,
    "dual-sphere-hyper-token-transform": DualSphereHyperTokenTransformLM,
    "dual-sphere-hyper-gate": DualSphereHyperGateLM,
    "dual-sphere-householder": DualSphereHouseholderLM,
    "dual-sphere-householder-per-token": DualSphereHouseholderPerTokenLM,
    "dual-sphere-householder-rank8": DualSphereHouseholderRank8LM,
    "dual-sphere-linear-recurrence": DualSphereLinearRecurrenceLM,
    "sine-sphere": SineSphereLM,
    "alibi-sphere": AlibiSphereLM,
    "alibi-sphere-two-block": AlibiSphereTwoBlockLM,
    "alibi-retriever-sphere": AlibiRetrieverSphereLM,
    "alibi-token-retriever-sphere": AlibiTokenRetrieverSphereLM,
    "alibi-token-retriever-sphere-bidirectional": AlibiTokenRetrieverSphereBidirectionalLM,
    "alibi-token-retriever-sphere-bidirectional-raw": AlibiTokenRetrieverSphereBidirectionalRawLM,
    "ngpt": NGPTLM,
    "simple-gpt": SimpleGPTLM,
    "rope-logit-sum": RopeLogitSumLM,
    "rope-log-gaussian-sum": RopeLogGaussianSumLM,
    "rope-kernel-attention": RopeKernelAttentionLM,
    "rope-kernel-hops": RopeKernelHopsLM,
    "rope-token-kernel": RopeTokenKernelLM,
    "rope-token-banks": RopeTokenBanksLM,
    "wave-field": WaveFieldLM,
    "wave-arrow": WaveArrowLM,
    "rope-state-memory": RopeStateMemoryLM,
    "min-gru": MinGRULM,
    "min-gru-cuda": MinGRUCudaLM,
    "gated-deltanet": GatedDeltaNetLM,
    "lru-banks": LRUBanksLM,
}

__all__ = [
    "ARCHS",
    "BaseLM",
    "DualSphereLM",
    "DualSphereDualReadoutLM",
    "DualSphereMidGateLM",
    "DualSphereHyperTokenTransformLM",
    "DualSphereHyperGateLM",
    "DualSphereHouseholderLM",
    "DualSphereHouseholderPerTokenLM",
    "DualSphereHouseholderRank8LM",
    "DualSphereLinearRecurrenceLM",
    "SineSphereLM",
    "AlibiSphereLM",
    "AlibiSphereTwoBlockLM",
    "AlibiRetrieverSphereLM",
    "AlibiTokenRetrieverSphereLM",
    "AlibiTokenRetrieverSphereBidirectionalLM",
    "AlibiTokenRetrieverSphereBidirectionalRawLM",
    "NGPTLM",
    "SimpleGPTLM",
    "RopeLogitSumLM",
    "RopeLogGaussianSumLM",
    "RopeKernelAttentionLM",
    "RopeKernelHopsLM",
    "build_model",
    "arch_defaults",
]


def _resolve(arch: str) -> Type[BaseLM]:
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Available: {sorted(ARCHS)}")
    return ARCHS[arch]


def build_model(config: Dict[str, Any]) -> BaseLM:
    """Instantiates the architecture named by `config["arch"]`.

    `arch_params` is passed straight through to the constructor, so the
    architecture's own signature is the only validator of its own parameters
    -- there is no second, drift-prone copy of that knowledge anywhere.
    """
    cls = _resolve(config["arch"])
    return cls(**config.get("arch_params", {}))


def arch_defaults(arch: str) -> Dict[str, Any]:
    """Default `arch_params` for an architecture, read from its signature.

    Used by `model create` to write a complete, explicit `config.json`, so
    every value a run depends on is visible in the file rather than hidden in
    a constructor default that might change later.
    """
    cls = _resolve(arch)
    defaults: Dict[str, Any] = {}
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name in ("self", "kwargs") or param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults
