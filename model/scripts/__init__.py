"""One-off analysis scripts, exposed through `model scripts <name>`.

Each entry maps a dash-case CLI name to a module under `model.scripts` that
defines `build_parser()` and `main(argv)` -- the same shape as `model.cli`'s
own subcommands, so `model scripts <name> --help` works for free.

To add a script: write `model/scripts/<module>.py` with those two functions,
then add one line to `SCRIPTS` below.
"""

from typing import Dict

SCRIPTS: Dict[str, str] = {
    "pga-dual-embeddings": "model.scripts.pga_dual_embeddings",
    "spherical-k-means": "model.scripts.spherical_kmeans",
    "table-neighbor-agreement": "model.scripts.table_neighbor_agreement",
    "query-recall-trace": "model.scripts.query_recall_trace",
    "retrieval-behavior-trace": "model.scripts.retrieval_behavior_trace",
    "retrieval-ablation-loss": "model.scripts.retrieval_ablation_loss",
    "retrieval-ablation-loss-tokenid": "model.scripts.retrieval_ablation_loss_tokenid",
    "token-retrieval-trace": "model.scripts.token_retrieval_trace",
    "order-sensitivity-loss": "model.scripts.order_sensitivity_loss",
    "name-swap-trace": "model.scripts.name_swap_trace",
    "token-frequency": "model.scripts.token_frequency",
    "compare-runs": "model.scripts.compare_runs",
    "log-gaussian-kernel-trace": "model.scripts.log_gaussian_kernel_trace",
    "log-gaussian-kernel-stepper": "model.scripts.log_gaussian_kernel_stepper",
    "conv-kernels": "model.scripts.conv_kernels",
    "token-kernel-report": "model.scripts.token_kernel_report",
    "token-kernel-stepper": "model.scripts.token_kernel_stepper",
    "token-banks-stepper": "model.scripts.token_banks_stepper",
    "token-banks-clusters": "model.scripts.token_banks_clusters",
    "wave-field-stepper": "model.scripts.wave_field_stepper",
    "wave-arrow-stepper": "model.scripts.wave_arrow_stepper",
    "rsm-bank-stepper": "model.scripts.rsm_bank_stepper",
}
