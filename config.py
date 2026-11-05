"""Regression configuration for ARG-MSA and TFN on four MSA datasets."""

import json
import os
from pathlib import Path
from typing import Optional, Union

from easydict import EasyDict as edict

SUPPORTED_MODELS = ("arg_msa", "tfn")
SUPPORTED_DATASETS = ("mosi", "mosei", "sims", "simsv2")


def _load_json(config_file: Union[str, Path]) -> dict:
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_feature_path(
    config_all: dict,
    feature_path: str,
    data_root: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve local feature paths; this function never downloads data.

    Explicit absolute feature paths take precedence. Relative feature paths use
    the data_root argument, MSA_DATA_ROOT, dataset_root_dir in the JSON, or the
    repository's data directory, in that order. Relative roots use the current
    working directory, matching normal command-line path handling.
    """
    feature = Path(str(feature_path)).expanduser()
    if feature.is_absolute():
        return str(feature.resolve())
    root = (
        data_root
        or os.environ.get("MSA_DATA_ROOT")
        or config_all.get("datasetCommonParams", {}).get("dataset_root_dir")
        or Path(__file__).resolve().parent / "data"
    )
    return str((Path(root).expanduser() / feature).resolve())


def _merge_regression_config(config_all: dict, model_name: str, dataset_name: str) -> dict:
    if model_name not in config_all:
        raise KeyError(f"Model {model_name} not found in config file")
    model_config = config_all[model_name]
    model_common = model_config["commonParams"]
    model_dataset = model_config["datasetParams"][dataset_name]
    dataset_common = config_all["datasetCommonParams"][dataset_name]

    # Preserve the original experiment's selection order. MOSEI's ARG-MSA
    # dataset override sets need_data_aligned=True after unaligned selection.
    # Changing this order would select a different input file.
    alignment = "aligned" if model_common.get("need_data_aligned", False) else "unaligned"
    if alignment not in dataset_common:
        raise KeyError(f"Dataset {dataset_name} does not provide {alignment} features")
    return {**dataset_common[alignment], **model_common, **model_dataset}


def _apply_common_defaults(config):
    def _set_default(k, v):
        if k not in config:
            config[k] = v

    _set_default("seed", 1111)
    _set_default("num_workers", 0)

    _set_default("hidden_dim", 128)
    _set_default("num_heads", 4)
    _set_default("num_iters", 4)
    _set_default("lambda_role", 0.2)
    _set_default("lambda_q", 0.2)
    _set_default("q_floor", 0.05)
    _set_default("q_floor_mode", "absolute")
    _set_default("q_logit_centering", False)
    _set_default("q_logit_centering_mode", "sample")
    _set_default("q_logit_centering_momentum", 0.0)
    _set_default("q_logit_norm", False)
    _set_default("q_logit_scale", 1.0)
    _set_default("local_residual_scale", 0.1)
    _set_default("node_residual_keep", 0.1)
    _set_default("global_residual_keep", 0.1)
    _set_default("node_init_residual_scale", 0.1)
    _set_default("global_init_residual_scale", 0.1)
    _set_default("relation_evidence_residual_scale", 0.1)
    _set_default("relation_temperature", 1.0)
    _set_default("relation_confidence_residual", False)
    _set_default("relation_confidence_scale", 0.0)
    _set_default("gate_temperature", 1.0)
    _set_default("undercut_gate_balance", False)
    _set_default("undercut_gate_scale", 0.0)
    _set_default("relation_aware_gate", False)
    _set_default("attack_gate_boost", 0.0)
    _set_default("undercut_gate_boost", 0.0)
    _set_default("q_temperature", 1.0)
    _set_default("q_state_floor", 0.0)
    _set_default("q_relation_aware", False)
    _set_default("q_relation_aware_v2", False)
    _set_default("q_relation_input", "message_gate_relation")
    _set_default("q_relation_feature_scale", 0.0)
    _set_default("q_preserve_delta", False)
    _set_default("q_preserve_delta_scale", 0.0)
    _set_default("q_delta_temperature", 1.0)
    _set_default("node_input_residual_scale", 0.1)
    _set_default("global_pooled_residual_scale", 0.1)
    _set_default("final_residual_scale", 0.0)
    _set_default("final_output_residual_scale", 0.0)
    _set_default("negative_output_residual_scale", 0.0)
    _set_default("negative_residual_threshold", 0.0)
    _set_default("pos_output_residual_scale", 0.0)
    _set_default("neg_output_residual_scale", 0.0)
    _set_default("negative_residual_warmup_epochs", 0)
    _set_default("negative_residual_max_abs", 0.0)
    _set_default("final_q_normalize", False)
    _set_default("sign_aware_final", False)
    _set_default("sign_negative_scale", 0.0)
    _set_default("sign_positive_scale", 0.0)
    _set_default("segment_window", 4)
    _set_default("segment_pooling", "mean")
    _set_default("dropout", 0.1)
    _set_default("use_soft_role", True)
    _set_default("batch_size", 16)
    _set_default("lr", 1e-3)
    if "learning_rate" not in config:
        config["learning_rate"] = config["lr"]
    if "lr" not in config and "learning_rate" in config:
        config["lr"] = config["learning_rate"]



    _set_default("pooling", "mean")
    _set_default("loss_name", "mse")
    _set_default("corr_loss_weight", 0.0)
    _set_default("mae_loss_weight", 0.0)
    _set_default("intensity_weighted_loss", False)
    _set_default("intensity_weight_alpha", 0.0)
    _set_default("intensity_weight_threshold", 1.5)
    _set_default("intensity_margin_loss_weight", 0.0)
    _set_default("intensity_margin_loss_warmup_epochs", 0)
    _set_default("intensity_margin_loss_warmup_start", 0.0)
    _set_default("intensity_threshold", 1.5)
    _set_default("intensity_margin_ratio", 0.85)
    _set_default("strong_negative_margin_loss_weight", 0.0)
    _set_default("strong_negative_margin_loss_warmup_epochs", 0)
    _set_default("strong_negative_margin_loss_warmup_start", 0.0)
    _set_default("strong_negative_threshold", -1.5)
    _set_default("strong_negative_margin_ratio", 0.85)
    _set_default("strong_negative_contrast_loss_weight", 0.0)
    _set_default("strong_negative_contrast_loss_warmup_epochs", 0)
    _set_default("strong_negative_contrast_loss_warmup_start", 0.0)
    _set_default("strong_negative_separation_loss_weight", 0.0)
    _set_default("strong_negative_loss_warmup_epochs", 0)
    _set_default("strong_negative_loss_warmup_start", 0.0)
    _set_default("weak_negative_lower", -1.0)
    _set_default("weak_negative_upper", 0.0)
    _set_default("strong_negative_margin", 0.3)
    _set_default("strong_negative_loss_type", "smooth_l1")
    _set_default("neutral_margin", 0.5)
    _set_default("pred_scale_loss_weight", 0.0)
    _set_default("pred_scale_loss_warmup_epochs", 0)
    _set_default("pred_scale_loss_warmup_start", 0.0)
    _set_default("pred_scale_ratio", 0.85)
    _set_default("pred_scale_upper_ratio", 0.0)
    _set_default("pred_scale_margin", 0.0)
    _set_default("rank_corr_loss_weight", 0.0)
    _set_default("rank_corr_loss_warmup_epochs", 0)
    _set_default("rank_corr_loss_warmup_start", 0.0)
    _set_default("rank_pair_sample_size", 64)
    _set_default("rank_margin", 0.05)
    _set_default("weighted_corr_loss_weight", 0.0)
    _set_default("weighted_corr_loss_warmup_epochs", 0)
    _set_default("weighted_corr_loss_warmup_start", 0.0)
    _set_default("weighted_corr_threshold", 0.5)
    _set_default("weighted_corr_alpha", 0.5)
    _set_default("q_entropy_loss_weight", 0.0)
    _set_default("q_entropy_loss_warmup_epochs", 0)
    _set_default("q_entropy_loss_warmup_start", 0.0)
    _set_default("q_entropy_target", 0.0)
    _set_default("q_entropy_margin", 0.0)
    _set_default("attention_entropy_reg_weight", 0.0)
    _set_default("attention_entropy_warmup_epochs", 0)
    _set_default("attention_entropy_warmup_start", 0.0)
    _set_default("attention_entropy_target", 0.0)
    _set_default("attention_temperature", 1.0)
    _set_default("final_amplitude_gate", False)
    _set_default("final_amplitude_scale", 0.0)
    _set_default("final_amplitude_temperature", 1.0)
    _set_default("final_amplitude_warmup_epochs", 0)
    _set_default("final_amplitude_warmup_start", 0.0)
    _set_default("negative_evidence_gate", False)
    _set_default("negative_evidence_scale", 0.0)
    _set_default("negative_evidence_threshold", 0.0)
    _set_default("positive_amplitude_recovery", False)
    _set_default("positive_amplitude_scale", 0.0)
    _set_default("positive_threshold", 1.0)
    _set_default("positive_amplitude_temperature", 1.0)
    _set_default("negative_amplitude_recovery", False)
    _set_default("negative_amplitude_scale", 0.0)
    _set_default("negative_threshold", -1.0)
    _set_default("negative_amplitude_temperature", 1.0)
    _set_default("attack_aware_negative_final", False)
    _set_default("attack_evidence_scale", 0.0)
    _set_default("undercut_evidence_scale", 0.0)
    _set_default("negative_final_temperature", 1.0)
    _set_default("amplitude_preserving_fusion", False)
    _set_default("local_amplitude_residual_scale", 0.0)
    _set_default("global_amplitude_residual_scale", 0.0)
    _set_default("final_fusion_retention_repair", False)
    _set_default("local_retention_residual_scale", 0.0)
    _set_default("global_retention_residual_scale", 0.0)
    _set_default("retention_gate_temperature", 1.0)
    _set_default("sign_preserve_gate", False)
    _set_default("local_global_balance_gate", False)
    _set_default("local_preserve_scale", 0.0)
    _set_default("global_preserve_scale", 0.0)
    _set_default("positive_final_retention", False)
    _set_default("negative_output_amplitude_scale", 0.0)
    _set_default("positive_output_amplitude_scale", 0.0)
    _set_default("retention_output_residual_scale", 0.0)
    _set_default("positive_retention_output_scale", 0.0)
    _set_default("negative_retention_output_scale", 0.0)
    _set_default("retention_output_residual", False)
    _set_default("retention_output_mode", "signed")
    _set_default("negative_retention_scale", 0.0)
    _set_default("positive_retention_scale", 0.0)
    _set_default("negative_confidence_threshold", 0.6)
    _set_default("positive_confidence_threshold", 0.6)
    _set_default("positive_residual_max_abs", 0.0)
    _set_default("positive_residual_warmup_epochs", 0)
    _set_default("retention_residual_safe_guard", False)
    _set_default("retention_guard_metric", "mae_acc2_f1_corr")
    _set_default("retention_acc2_drop_limit", 0.001)
    _set_default("retention_f1_drop_limit", 0.001)
    _set_default("retention_corr_drop_limit", 0.001)
    _set_default("negative_evidence_consistency_residual", False)
    _set_default("negative_evidence_consistency_residual_v2", False)
    _set_default("negative_evidence_consistency_residual_v3", False)
    _set_default("negative_evidence_consistency_residual_v4", False)
    _set_default("negative_evidence_residual_v6", False)
    _set_default("negative_v6_use_hlocal", True)
    _set_default("negative_v6_use_q_node", True)
    _set_default("negative_v6_use_residual_direction", True)
    _set_default("negative_v6_use_attack_role", True)
    _set_default("negative_v6_use_positive_suppression", True)
    _set_default("negative_v6_use_neutral_suppression", True)
    _set_default("negative_v6_score_norm", "batch_zscore")
    _set_default("negative_v6_threshold", 0.50)
    _set_default("negative_v6_scale", 0.0)
    _set_default("negative_v6_max_abs", 0.0)
    _set_default("negative_v6_warmup_epochs", 0)
    _set_default("negative_v6_trace", False)
    _set_default("negative_evidence_residual_v7", False)
    _set_default("negative_v7_use_hlocal", True)
    _set_default("negative_v7_use_q_node", True)
    _set_default("negative_v7_use_residual_direction", True)
    _set_default("negative_v7_use_attack_role", True)
    _set_default("negative_v7_use_positive_suppression", True)
    _set_default("negative_v7_use_neutral_suppression", True)
    _set_default("negative_v7_use_weak_negative_suppression", True)
    _set_default("negative_v7_score_norm", "batch_zscore")
    _set_default("negative_v7_threshold", 0.55)
    _set_default("negative_v7_strong_threshold", 0.65)
    _set_default("negative_v7_neutral_suppress_threshold", 0.45)
    _set_default("negative_v7_weak_suppress_threshold", 0.40)
    _set_default("negative_v7_scale", 0.0)
    _set_default("negative_v7_max_abs", 0.0)
    _set_default("negative_v7_warmup_epochs", 0)
    _set_default("negative_v7_trace", False)
    _set_default("negative_residual_path_gain", False)
    _set_default("negative_path_gain_mode", "gate_confidence")
    _set_default("negative_path_gain_scale", 1.5)
    _set_default("negative_path_gain_max", 2.5)
    _set_default("negative_path_gain_min_gate", 0.70)
    _set_default("negative_path_gain_require_neutral_block", True)
    _set_default("negative_path_gain_require_weak_block", True)
    _set_default("negative_path_gain_trace", False)
    _set_default("negative_confidence_source", "legacy")
    _set_default("negative_confidence_use_attack", True)
    _set_default("negative_confidence_use_hlocal", True)
    _set_default("negative_confidence_use_q_node", True)
    _set_default("negative_confidence_use_residual_direction", False)
    _set_default("negative_confidence_use_role", True)
    _set_default("negative_confidence_use_undercut", True)
    _set_default("negative_confidence_neutral_suppress", False)
    _set_default("negative_evidence_consistency_scale", 0.0)
    _set_default("negative_neutral_suppress_threshold", 0.4)
    _set_default("negative_consistency_warmup_epochs", 0)
    _set_default("positive_evidence_consistency_residual", False)
    _set_default("positive_evidence_consistency_residual_v2", False)
    _set_default("positive_evidence_consistency_residual_v3", False)
    _set_default("positive_evidence_consistency_residual_v4", False)
    _set_default("positive_confidence_source", "legacy")
    _set_default("positive_confidence_use_support", True)
    _set_default("positive_confidence_use_hlocal", True)
    _set_default("positive_confidence_use_q_node", True)
    _set_default("positive_confidence_use_role", True)
    _set_default("positive_confidence_neutral_suppress", False)
    _set_default("positive_intensity_gate", False)
    _set_default("positive_intensity_threshold", 0.55)
    _set_default("positive_weak_suppress_threshold", 0.35)
    _set_default("positive_evidence_consistency_scale", 0.0)
    _set_default("positive_evidence_threshold", 0.65)
    _set_default("positive_neutral_suppress_threshold", 0.4)
    _set_default("positive_consistency_warmup_epochs", 0)
    _set_default("smooth_l1_beta", 1.0)
    _set_default("return_debug_tensors", False)
    _set_default("show_progress", True)
    _set_default("enable_iter_analysis", False)
    _set_default("iter_analysis_split", "valid")
    _set_default("iter_analysis_every_epoch", 1)
    _set_default("enable_arg_msa_error_diagnostics", False)
    _set_default("enable_internal_diagnostics", False)
    _set_default("internal_diag_every_epoch", 1)
    _set_default("enable_layerwise_diagnostics", False)
    _set_default("enable_method_chain_diagnostics", False)
    _set_default("method_chain_diag_every_epoch", 1)
    _set_default("skip_epoch_test", False)
    _set_default("enable_val_scale_calibration", False)
    _set_default("calibration_scale_min", 0.5)
    _set_default("calibration_scale_max", 2.0)
    _set_default("calibration_blend", 1.0)
    _set_default("calibration_apply_to_test", True)
    _set_default("calibration_clip_outputs", False)
    _set_default("calibration_affects_eval_loss", True)
    _set_default("calibration_min_pred_std", 0.0001)
    _set_default("calibration_allow_negative_scale", False)
    _set_default("calibration_sign_preserve", False)
    _set_default("calibration_bias_limit", 0.3)
    _set_default("calibration_warmup_epochs", 0)
    _set_default("calibration_bias_freeze_epochs", 0)
    _set_default("calibration_acc2_drop_limit", None)
    _set_default("calibration_f1_drop_limit", None)
    _set_default("calibration_corr_drop_limit", None)
    _set_default("calibration_acc7_drop_limit", None)
    _set_default("calibration_acc5_drop_limit", None)
    _set_default("calibration_acc3_drop_limit", None)
    _set_default("calibration_mae_drop_guard", False)
    _set_default("calibration_acc3_target", None)
    _set_default("q_min_entropy_guard", 0.0)
    _set_default("calibration_acc2_drop_tolerance", 0.005)
    _set_default("calibration_f1_drop_tolerance", 0.005)
    _set_default("calibration_corr_drop_tolerance", 0.005)
    _set_default("calibration_metric_acc7_weight", 1.0)
    _set_default("calibration_metric_acc5_weight", 1.0)
    _set_default("calibration_metric_acc3_weight", 0.5)
    _set_default("calibration_metric_acc2_weight", 1.0)
    _set_default("calibration_metric_f1_weight", 1.0)
    _set_default("calibration_metric_corr_weight", 2.0)
    _set_default("calibration_metric_mae_weight", 2.0)
    _set_default("calibration_min_acc7", None)
    _set_default("calibration_min_acc5", None)
    _set_default("calibration_min_acc3", None)
    _set_default("calibration_min_acc2", None)
    _set_default("calibration_min_f1", None)
    _set_default("calibration_min_corr", None)
    _set_default("calibration_max_mae", None)
    _set_default("calibration_threshold_penalty_weight", 100.0)
    _set_default("negative_scale_min", 1.0)
    _set_default("negative_scale_max", 1.6)
    _set_default("positive_scale_max", 1.05)
    _set_default("bias_correction_objective", "none")
    _set_default("bias_correction_limit", 0.0)
    _set_default("bias_correction_steps", 21)
    _set_default("calibration_objective", "mse")
    _set_default("calibration_grid_steps", 11)
    _set_default("calibration_bias_steps", 21)
    _set_default("calibration_scale_span", 0.25)
    _set_default("calibration_bias_span", 0.25)
    return config


def get_config_regression(
    model_name: str,
    dataset_name: str,
    config_file: Union[str, Path] = "",
    data_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Load the original experiment settings, with portable local data paths."""
    model_name = model_name.lower().replace("-", "_").strip()
    dataset_name = dataset_name.lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}; choose from {SUPPORTED_DATASETS}")
    config_path = config_file or Path(__file__).parent / "config" / "config_regression.json"
    config_all = _load_json(config_path)
    config = _merge_regression_config(config_all, model_name, dataset_name)
    config["model_name"] = model_name
    config["dataset_name"] = dataset_name
    if "featurePath" in config:
        config["featurePath"] = _resolve_feature_path(config_all, config["featurePath"], data_root)
    return edict(_apply_common_defaults(config))


def get_citations() -> dict:
    """Return the citations supplied by the original repository."""
    return _load_json(Path(__file__).parent / "config" / "citations.json")
