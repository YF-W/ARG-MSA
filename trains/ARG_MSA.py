import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger("MSA")

def _get_arg(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)

class ARG_MSA:
    METRIC_ALIASES = {
        "Acc_7": ("Acc_7", "acc_7"),
        "Acc_5": ("Acc_5", "acc_5"),
        "Acc_3": ("Acc_3", "acc_3"),
        "Acc_2": ("Acc_2", "acc_2"),
        "F1_score": ("F1_score", "f1_score"),
        "Corr": ("Corr", "corr"),
        "MAE": ("MAE", "mae"),
    }

    def __init__(self, args):
        self.args = args
        self.criterion = self._build_criterion()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self._last_val_calibration = None
        self._current_epoch = 0

    def _build_criterion(self):
        if self.args.train_mode == "classification":
            return nn.CrossEntropyLoss()

        loss_name = str(_get_arg(self.args, "loss_name", "mse")).lower()
        if loss_name in {"l1", "mae"}:
            return nn.L1Loss()
        if loss_name == "smooth_l1":
            beta = float(_get_arg(self.args, "smooth_l1_beta", 1.0))
            return nn.SmoothL1Loss(beta=beta)
        return nn.MSELoss()

    def _warmup_factor(self, warmup_epochs, warmup_start=0.0):
        warmup_epochs = int(max(0, warmup_epochs))
        if warmup_epochs <= 1 or self._current_epoch <= 0:
            return 1.0
        warmup_start = float(max(0.0, min(1.0, warmup_start)))
        progress = min(
            1.0,
            max(0.0, (self._current_epoch - 1) / float(max(1, warmup_epochs - 1))),
        )
        return warmup_start + (1.0 - warmup_start) * progress

    def _is_stage1(self):
        if not bool(_get_arg(self.args, "stagewise_training", False)):
            return False
        stage1_epochs = int(max(0, _get_arg(self.args, "stage1_epochs", 0) or 0))
        return self._current_epoch > 0 and self._current_epoch <= stage1_epochs

    def _stagewise_weight_enabled(self, weight_key):
        if not self._is_stage1():
            return True
        rank_keys = {"rank_corr_loss_weight"}
        negative_fix_keys = {
            "strong_negative_margin_loss_weight",
            "strong_negative_contrast_loss_weight",
            "strong_negative_separation_loss_weight",
            "negative_argument_consistency_loss_weight",
            "residual_debias_weight",
        }
        if bool(_get_arg(self.args, "stage2_enable_rank", False)) and weight_key in rank_keys:
            return False
        if (
            bool(_get_arg(self.args, "stage2_enable_negative_fix", False))
            and weight_key in negative_fix_keys
        ):
            return False
        return True

    def _scheduled_aux_weight(self, weight_key, warmup_key, start_key):
        weight = float(_get_arg(self.args, weight_key, 0.0))
        if weight <= 0.0:
            return 0.0
        if not self._stagewise_weight_enabled(weight_key):
            return 0.0
        return weight * self._warmup_factor(
            _get_arg(self.args, warmup_key, 0),
            _get_arg(self.args, start_key, 0.0),
        )

    def _compute_base_loss(self, pred, labels):
        if (
            self.args.train_mode == "regression"
            and self.args.model_name == "arg_msa"
            and bool(_get_arg(self.args, "intensity_weighted_loss", False))
        ):
            loss_name = str(_get_arg(self.args, "loss_name", "mse")).lower()
            threshold = float(_get_arg(self.args, "intensity_weight_threshold", 1.5))
            alpha = float(_get_arg(self.args, "intensity_weight_alpha", 0.0))
            if alpha > 0.0 and loss_name in {"mse", "l1", "mae", "smooth_l1"}:
                if loss_name in {"l1", "mae"}:
                    element_loss = torch.abs(pred - labels)
                elif loss_name == "smooth_l1":
                    beta = float(_get_arg(self.args, "smooth_l1_beta", 1.0))
                    element_loss = F.smooth_l1_loss(pred, labels, beta=beta, reduction="none")
                else:
                    element_loss = (pred - labels) ** 2
                excess = (labels.detach().abs() - threshold).clamp(min=0.0)
                weights = 1.0 + alpha * excess
                return (element_loss * weights).mean()
        return self.criterion(pred, labels)

    def _q_entropy_loss(self, q_tensor):
        weight = self._scheduled_aux_weight(
            "q_entropy_loss_weight",
            "q_entropy_loss_warmup_epochs",
            "q_entropy_loss_warmup_start",
        )
        if weight <= 0.0 or not torch.is_tensor(q_tensor):
            return None
        q = q_tensor.float().squeeze(-1)
        if q.dim() != 2 or q.size(1) <= 1:
            return None
        probs = q.clamp(min=0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)
        entropy = entropy / torch.log(torch.as_tensor(float(q.size(1)), device=q.device))
        target = float(_get_arg(self.args, "q_entropy_target", 0.0))
        margin = float(_get_arg(self.args, "q_entropy_margin", 0.0))
        if target > 0.0:
            penalty = F.relu(entropy - target)
        elif margin > 0.0:
            penalty = F.relu(entropy - (1.0 - margin))
        else:
            penalty = entropy
        return weight * penalty.mean()

    def _attention_entropy_loss(self, attention_entropy):
        weight = self._scheduled_aux_weight(
            "attention_entropy_reg_weight",
            "attention_entropy_warmup_epochs",
            "attention_entropy_warmup_start",
        )
        if weight <= 0.0 or not torch.is_tensor(attention_entropy):
            return None
        entropy = attention_entropy.float().squeeze(-1)
        target = float(_get_arg(self.args, "attention_entropy_target", 0.0))
        if target > 0.0:
            penalty = (entropy - target).pow(2)
        else:
            penalty = entropy
        return weight * penalty.mean()

    def _pred_scale_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "pred_scale_loss_weight",
            "pred_scale_loss_warmup_epochs",
            "pred_scale_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        pred_vec = pred.float().view(-1)
        true_vec = labels.float().view(-1)
        if pred_vec.numel() <= 1:
            return None
        pred_std = pred_vec.std(unbiased=False)
        true_std = true_vec.detach().std(unbiased=False)
        ratio = float(_get_arg(self.args, "pred_scale_ratio", 0.85))
        margin = float(_get_arg(self.args, "pred_scale_margin", 0.0))
        target_std = true_std * max(0.0, ratio) + margin
        penalty = F.relu(target_std - pred_std).pow(2)
        upper_ratio = float(_get_arg(self.args, "pred_scale_upper_ratio", 0.0))
        if upper_ratio > 0.0:
            upper_std = true_std * upper_ratio + margin
            penalty = penalty + F.relu(pred_std - upper_std).pow(2)
        return weight * penalty

    def _intensity_margin_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "intensity_margin_loss_weight",
            "intensity_margin_loss_warmup_epochs",
            "intensity_margin_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        threshold = float(_get_arg(self.args, "intensity_threshold", 1.5))
        ratio = float(_get_arg(self.args, "intensity_margin_ratio", 0.85))
        true_abs = labels.detach().abs()
        strong_mask = true_abs >= threshold
        if not bool(strong_mask.any().item()):
            return None
        target_abs = true_abs * max(0.0, ratio)
        penalty = F.relu(target_abs - pred.abs()).pow(2)
        return weight * penalty[strong_mask].mean()

    def _strong_negative_margin_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "strong_negative_margin_loss_weight",
            "strong_negative_margin_loss_warmup_epochs",
            "strong_negative_margin_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        threshold = float(_get_arg(self.args, "strong_negative_threshold", -1.5))
        ratio = float(_get_arg(self.args, "strong_negative_margin_ratio", 0.85))
        mask = labels.detach() <= threshold
        if not bool(mask.any().item()):
            return None
        target = labels.detach().abs() * max(0.0, ratio)
        penalty = F.relu(target - pred.abs()).pow(2)
        wrong_side = F.relu(pred)
        return weight * (penalty[mask].mean() + wrong_side[mask].pow(2).mean())

    def _strong_negative_contrast_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "strong_negative_contrast_loss_weight",
            "strong_negative_contrast_loss_warmup_epochs",
            "strong_negative_contrast_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        pred_vec = pred.float().view(-1)
        label_vec = labels.detach().float().view(-1)
        threshold = float(_get_arg(self.args, "strong_negative_threshold", -1.5))
        neutral_margin = max(0.0, float(_get_arg(self.args, "neutral_margin", 0.5)))
        strong_mask = label_vec <= threshold
        if not bool(strong_mask.any().item()):
            return None
        strong_pred = pred_vec[strong_mask]
        loss = F.relu(strong_pred + neutral_margin).pow(2).mean()
        contrast_mask = label_vec > threshold
        if bool(contrast_mask.any().item()):
            reference = pred_vec[contrast_mask].detach().mean()
            loss = loss + F.relu(strong_pred - reference + neutral_margin).pow(2).mean()
        return weight * loss

    def _strong_negative_separation_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "strong_negative_separation_loss_weight",
            "strong_negative_loss_warmup_epochs",
            "strong_negative_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        pred_vec = pred.float().view(-1)
        label_vec = labels.detach().float().view(-1)
        threshold = float(_get_arg(self.args, "strong_negative_threshold", -1.5))
        weak_lower = float(_get_arg(self.args, "weak_negative_lower", -1.0))
        weak_upper = float(_get_arg(self.args, "weak_negative_upper", 0.0))
        margin = max(0.0, float(_get_arg(self.args, "strong_negative_margin", 0.3)))
        strong_mask = label_vec <= threshold
        weak_mask = (label_vec >= weak_lower) & (label_vec < weak_upper)
        if not bool(strong_mask.any().item()) or not bool(weak_mask.any().item()):
            return None
        strong_pred = pred_vec[strong_mask]
        weak_ref = pred_vec[weak_mask].detach().mean()
        violation = strong_pred - weak_ref + margin
        loss_type = str(_get_arg(self.args, "strong_negative_loss_type", "smooth_l1")).lower()
        if loss_type == "smooth_l1":
            zeros = torch.zeros_like(violation)
            penalty = F.smooth_l1_loss(violation, zeros, reduction="none")
            penalty = torch.where(violation > 0.0, penalty, zeros)
        else:
            penalty = F.relu(violation).pow(2)
        sign_penalty = F.relu(strong_pred).pow(2)
        return weight * (penalty.mean() + 0.25 * sign_penalty.mean())

    def _aux_regression_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "aux_regression_loss_weight",
            "aux_regression_loss_warmup_epochs",
            "aux_regression_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        loss_name = str(_get_arg(self.args, "aux_regression_loss", "none")).lower()
        if loss_name == "smooth_l1":
            beta = float(_get_arg(self.args, "smooth_l1_beta", 1.0))
            return weight * F.smooth_l1_loss(pred, labels, beta=beta)
        if loss_name in {"l1", "mae"}:
            return weight * F.l1_loss(pred, labels)
        if loss_name == "mse":
            return weight * F.mse_loss(pred, labels)
        return None

    def _negative_argument_consistency_loss(self, pred, labels, outputs):
        weight = self._scheduled_aux_weight(
            "negative_argument_consistency_loss_weight",
            "negative_argument_consistency_loss_warmup_epochs",
            "negative_argument_consistency_loss_warmup_start",
        )
        if weight <= 0.0 or outputs is None:
            return None
        argument_score = outputs.get("argument_sentiment")
        if not torch.is_tensor(argument_score):
            return None
        threshold = float(_get_arg(self.args, "negative_argument_threshold", -1.5))
        margin = float(_get_arg(self.args, "negative_argument_margin", 0.5))
        label_vec = labels.detach().float().view(-1)
        pred_vec = pred.float().view(-1)
        argument_vec = argument_score.float().view(-1)
        strong_mask = label_vec <= threshold
        if not bool(strong_mask.any().item()):
            return None
        strong_argument = argument_vec[strong_mask]
        strong_pred = pred_vec[strong_mask]
        consistency = F.smooth_l1_loss(strong_argument, strong_pred.detach())
        negative_floor = F.relu(strong_argument + margin).pow(2).mean()
        final_alignment = F.relu(strong_pred - strong_argument.detach() + margin).pow(2).mean()
        return weight * (consistency + negative_floor + 0.5 * final_alignment)

    def _residual_sign_debias_loss(self, labels, outputs):
        if not bool(_get_arg(self.args, "residual_sign_debias", False)):
            return None
        weight = float(_get_arg(self.args, "residual_debias_weight", 0.0))
        if weight <= 0.0 or not self._stagewise_weight_enabled("residual_debias_weight"):
            return None
        if outputs is None:
            return None
        layer_trace = outputs.get("layer_trace", {})
        residual = layer_trace.get("final_output_residual") if isinstance(layer_trace, dict) else None
        if not torch.is_tensor(residual):
            return None
        threshold = float(_get_arg(self.args, "negative_argument_threshold", -1.5))
        target_ratio = float(_get_arg(self.args, "negative_residual_target_ratio", 0.85))
        label_vec = labels.detach().float().view(-1)
        residual_vec = residual.float().view(-1)
        strong_mask = label_vec <= threshold
        if not bool(strong_mask.any().item()):
            return None
        strong_residual = residual_vec[strong_mask]
        strong_label = label_vec[strong_mask]
        target = -target_ratio * strong_label.abs()
        sign_penalty = F.relu(strong_residual).pow(2).mean()
        target_penalty = F.relu(strong_residual - target).pow(2).mean()
        return weight * (sign_penalty + target_penalty)

    def _rank_corr_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "rank_corr_loss_weight",
            "rank_corr_loss_warmup_epochs",
            "rank_corr_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        pred_vec = pred.float().view(-1)
        true_vec = labels.detach().float().view(-1)
        n = pred_vec.numel()
        if n <= 1:
            return None
        max_pairs = max(1, int(_get_arg(self.args, "rank_pair_sample_size", 64)))
        pair_count = min(max_pairs, n * (n - 1))
        idx_i = torch.randint(0, n, (pair_count,), device=pred_vec.device)
        idx_j = torch.randint(0, n, (pair_count,), device=pred_vec.device)
        valid = true_vec[idx_i] > true_vec[idx_j]
        if not bool(valid.any().item()):
            return None
        margin = float(_get_arg(self.args, "rank_margin", 0.05))
        diffs = pred_vec[idx_i[valid]] - pred_vec[idx_j[valid]]
        return weight * F.relu(margin - diffs).mean()

    def _weighted_corr_loss(self, pred, labels):
        weight = self._scheduled_aux_weight(
            "weighted_corr_loss_weight",
            "weighted_corr_loss_warmup_epochs",
            "weighted_corr_loss_warmup_start",
        )
        if weight <= 0.0:
            return None
        pred_vec = pred.float().view(-1)
        true_vec = labels.float().view(-1)
        if pred_vec.numel() <= 1:
            return None
        threshold = float(_get_arg(self.args, "weighted_corr_threshold", 0.5))
        alpha = float(_get_arg(self.args, "weighted_corr_alpha", 0.5))
        sample_weight = 1.0 + alpha * F.relu(true_vec.detach().abs() - threshold)
        sample_weight = sample_weight / sample_weight.mean().clamp(min=1e-8)
        pred_mean = torch.sum(sample_weight * pred_vec) / sample_weight.sum().clamp(min=1e-8)
        true_mean = torch.sum(sample_weight * true_vec) / sample_weight.sum().clamp(min=1e-8)
        pred_centered = pred_vec - pred_mean
        true_centered = true_vec - true_mean
        cov = torch.sum(sample_weight * pred_centered * true_centered) / sample_weight.sum().clamp(min=1e-8)
        pred_std = torch.sqrt(torch.sum(sample_weight * pred_centered.pow(2)) / sample_weight.sum().clamp(min=1e-8)).clamp(min=1e-8)
        true_std = torch.sqrt(torch.sum(sample_weight * true_centered.pow(2)) / sample_weight.sum().clamp(min=1e-8)).clamp(min=1e-8)
        corr = cov / (pred_std * true_std).clamp(min=1e-8)
        return weight * (1.0 - corr.clamp(min=-1.0, max=1.0))

    # ARG_MSA_31 / Candidate 1: SIMS Soft Bucket-Distance Auxiliary Loss
    # default-OFF, SIMS-only. Activated when:
    #   args.dataset_name == 'sims'
    #   args.sims_bucket_aux_enable is True
    #   args.sims_bucket_aux_weight > 0
    # Mechanism: KL(p_true || p_pred) over 7-bucket soft distribution on SIMS
    # label centers [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]. Each sample's prediction
    # and label produce a soft probability over the 7 buckets via gaussian-like
    # kernel softmax(-(c - x)^2 / tau^2). The KL term encourages prediction
    # and label to fall in the same bucket (penalizing adjacent-bucket leakage),
    # without forcing prediction to bucket centers (preserves Corr/MAE).
    # Differs from rank_corr_loss (rank ordering), mae_loss (L1 magnitude), and
    # pred_scale_loss (global scale): this is per-sample soft bucket KL only.
    def _sims_bucket_distance_loss(self, pred, labels):
        if str(_get_arg(self.args, "dataset_name", "")).lower() != "sims":
            return None
        if not bool(_get_arg(self.args, "sims_bucket_aux_enable", False)):
            return None
        weight = float(_get_arg(self.args, "sims_bucket_aux_weight", 0.0))
        if weight <= 0.0:
            return None
        tau = float(_get_arg(self.args, "sims_bucket_aux_tau", 0.1))
        if tau <= 0.0:
            return None
        pred_vec = pred.float().view(-1, 1)
        true_vec = labels.float().view(-1, 1)
        if pred_vec.numel() == 0:
            return None
        centers = torch.tensor(
            [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        ).view(1, -1)
        inv_tau_sq = 1.0 / (tau * tau)
        d_true = -((centers - true_vec) ** 2) * inv_tau_sq
        d_pred = -((centers - pred_vec) ** 2) * inv_tau_sq
        log_p_true = F.log_softmax(d_true, dim=-1)
        log_p_pred = F.log_softmax(d_pred, dim=-1)
        p_true = log_p_true.exp()
        kl = (p_true * (log_p_true - log_p_pred)).sum(dim=-1).mean()
        return weight * kl

    def _compute_loss(self, pred, labels, use_auxiliary=False, outputs=None):
        self._last_aux_loss_values = {}
        loss = self._compute_base_loss(pred, labels)
        if (
            not use_auxiliary
            or self.args.train_mode != "regression"
            or self.args.model_name != "arg_msa"
        ):
            return loss

        corr_weight = float(_get_arg(self.args, "corr_loss_weight", 0.0))
        mae_weight = float(_get_arg(self.args, "mae_loss_weight", 0.0))

        if mae_weight > 0.0:
            loss = loss + mae_weight * F.l1_loss(pred, labels)

        aux_regression_loss = self._aux_regression_loss(pred, labels)
        if aux_regression_loss is not None:
            loss = loss + aux_regression_loss

        if corr_weight > 0.0:
            pred_vec = pred.float().view(-1)
            true_vec = labels.float().view(-1)
            pred_centered = pred_vec - pred_vec.mean()
            true_centered = true_vec - true_vec.mean()
            pred_std = torch.sqrt(torch.mean(pred_centered * pred_centered).clamp(min=1e-8))
            true_std = torch.sqrt(torch.mean(true_centered * true_centered).clamp(min=1e-8))
            corr = torch.mean(pred_centered * true_centered) / (pred_std * true_std).clamp(min=1e-8)
            loss = loss + corr_weight * (1.0 - corr.clamp(min=-1.0, max=1.0))
        scale_loss = self._pred_scale_loss(pred, labels)
        if scale_loss is not None:
            loss = loss + scale_loss
        intensity_loss = self._intensity_margin_loss(pred, labels)
        if intensity_loss is not None:
            loss = loss + intensity_loss
        strong_negative_loss = self._strong_negative_margin_loss(pred, labels)
        if strong_negative_loss is not None:
            loss = loss + strong_negative_loss
        strong_negative_contrast_loss = self._strong_negative_contrast_loss(pred, labels)
        if strong_negative_contrast_loss is not None:
            loss = loss + strong_negative_contrast_loss
        strong_negative_separation_loss = self._strong_negative_separation_loss(pred, labels)
        if strong_negative_separation_loss is not None:
            loss = loss + strong_negative_separation_loss
            self._last_aux_loss_values = {
                **getattr(self, "_last_aux_loss_values", {}),
                "strong_negative_separation_loss": float(
                    strong_negative_separation_loss.detach().cpu().item()
                ),
            }
        rank_loss = self._rank_corr_loss(pred, labels)
        if rank_loss is not None:
            loss = loss + rank_loss
        weighted_corr_loss = self._weighted_corr_loss(pred, labels)
        if weighted_corr_loss is not None:
            loss = loss + weighted_corr_loss
        # ARG_MSA_31 Candidate 1: SIMS soft bucket-distance KL aux loss
        # default-off; SIMS-only gating inside the helper. No effect for MOSI / MOSEI / SIMSV2.
        sims_bucket_loss = self._sims_bucket_distance_loss(pred, labels)
        if sims_bucket_loss is not None:
            loss = loss + sims_bucket_loss
            self._last_aux_loss_values = {
                **getattr(self, "_last_aux_loss_values", {}),
                "sims_bucket_distance_loss": float(
                    sims_bucket_loss.detach().cpu().item()
                ),
            }
        if outputs is not None:
            negative_argument_loss = self._negative_argument_consistency_loss(
                pred, labels, outputs
            )
            if negative_argument_loss is not None:
                loss = loss + negative_argument_loss
            residual_debias_loss = self._residual_sign_debias_loss(labels, outputs)
            if residual_debias_loss is not None:
                loss = loss + residual_debias_loss
            q_loss = self._q_entropy_loss(outputs.get("segment_acceptability"))
            if q_loss is not None:
                loss = loss + q_loss
            attn_loss = self._attention_entropy_loss(outputs.get("attention_entropy"))
            if attn_loss is not None:
                loss = loss + attn_loss
        return loss

    def _build_optimizer(self, model):
        learning_rate = float(
            _get_arg(self.args, "learning_rate", _get_arg(self.args, "lr", 1e-3))
        )
        weight_decay = float(_get_arg(self.args, "weight_decay", 0.0))
        no_decay_terms = ("bias", "norm", "layernorm", "input_norm")
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(term in name.lower() for term in no_decay_terms):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return optim.Adam(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )

    def _prepare_labels(self, labels):
        if self.args.train_mode == "classification":
            return labels.view(-1).long()
        return labels.view(-1, 1)

    def _forward_model(
        self,
        model,
        batch_data,
        collect_iter_history=False,
        collect_layer_trace=False,
    ):
        text = batch_data["text"].to(self.args.device)
        audio = batch_data["audio"].to(self.args.device)
        vision = batch_data["vision"].to(self.args.device)
        text_lengths = batch_data.get("text_lengths")
        audio_lengths = batch_data.get("audio_lengths")
        vision_lengths = batch_data.get("vision_lengths")
        if text_lengths is not None and not torch.is_tensor(text_lengths):
            text_lengths = torch.tensor(text_lengths)
        if audio_lengths is not None and not torch.is_tensor(audio_lengths):
            audio_lengths = torch.tensor(audio_lengths)
        if vision_lengths is not None and not torch.is_tensor(vision_lengths):
            vision_lengths = torch.tensor(vision_lengths)
        if text_lengths is not None:
            text_lengths = text_lengths.to(self.args.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.args.device)
        if vision_lengths is not None:
            vision_lengths = vision_lengths.to(self.args.device)
        return model(
            text,
            audio,
            vision,
            text_lengths=text_lengths,
            audio_lengths=audio_lengths,
            vision_lengths=vision_lengths,
            collect_iter_history=collect_iter_history,
            collect_layer_trace=collect_layer_trace,
        )

    def _is_arg_msa_sims_family(self):
        return (
            self.args.model_name == "arg_msa"
            and str(self.args.dataset_name).lower() in {"mosi", "mosei", "sims", "simsv2"}
        )

    def _should_log_internal_diagnostics(self, mode, epoch_num=None):
        if not self._is_arg_msa_sims_family():
            return False
        if not bool(_get_arg(self.args, "enable_internal_diagnostics", True)):
            return False
        every_epoch = max(1, int(_get_arg(self.args, "internal_diag_every_epoch", 1)))
        if epoch_num is not None and epoch_num % every_epoch != 0:
            return False
        return mode.upper() in {"TRAIN", "VAL", "VALID", "TEST"}

    def _should_log_method_chain_diagnostics(self, mode, epoch_num=None):
        if not self._is_arg_msa_sims_family():
            return False
        if not bool(_get_arg(self.args, "enable_method_chain_diagnostics", False)):
            return False
        every_epoch = max(1, int(_get_arg(self.args, "method_chain_diag_every_epoch", 1)))
        if epoch_num is not None and epoch_num % every_epoch != 0:
            return False
        return mode.upper() in {"VAL", "VALID", "TEST"}

    def _should_log_layerwise_trace(self, mode, epoch_num=None):
        if not self._should_log_internal_diagnostics(mode, epoch_num):
            return False
        return bool(_get_arg(self.args, "enable_layerwise_diagnostics", True))

    def _canonicalize_metrics(self, metric_dict):
        canonical = {}
        for canonical_key, aliases in self.METRIC_ALIASES.items():
            for alias in aliases:
                if alias in metric_dict:
                    canonical[canonical_key] = float(metric_dict[alias])
                    break
        if "Loss" in metric_dict:
            canonical["Loss"] = float(metric_dict["Loss"])
        return canonical

    def _calibration_enabled(self):
        if (
            self._is_stage1()
            and bool(_get_arg(self.args, "stage1_disable_calibration", False))
        ):
            return False
        return (
            self.args.model_name == "arg_msa"
            and self.args.train_mode == "regression"
            and bool(_get_arg(self.args, "enable_val_scale_calibration", False))
        )

    def _calibration_bounds(self):
        dataset = str(self.args.dataset_name).lower()
        if dataset in {"sims", "simsv2"}:
            return -1.0, 1.0
        return -3.0, 3.0

    def _calibration_drop_tolerance(self, metric_name, default):
        metric_key = str(metric_name).lower().replace("_", "")
        aliases = {
            "acc7": ("calibration_acc7_drop_limit",),
            "acc5": ("calibration_acc5_drop_limit",),
            "acc2": ("calibration_acc2_drop_limit", "calibration_acc2_drop_tolerance"),
            "acc3": ("calibration_acc3_drop_limit",),
            "f1": ("calibration_f1_drop_limit", "calibration_f1_drop_tolerance"),
            "f1score": ("calibration_f1_drop_limit", "calibration_f1_drop_tolerance"),
            "corr": ("calibration_corr_drop_limit", "calibration_corr_drop_tolerance"),
        }
        for key in aliases.get(metric_key, ()):
            value = _get_arg(self.args, key, None)
            if value is not None:
                return float(value)
        return float(default)

    def _apply_retention_residual_safe_guard(self, raw_pred, residual_pred, true, mode):
        if (
            self.args.model_name != "arg_msa"
            or self.args.train_mode != "regression"
            or not bool(_get_arg(self.args, "retention_residual_safe_guard", False))
            or str(mode).upper() not in {"VAL", "VALID"}
        ):
            return residual_pred, None
        raw_metrics = self._canonicalize_metrics(self.metrics(raw_pred, true))
        residual_metrics = self._canonicalize_metrics(self.metrics(residual_pred, true))
        acc2_drop = raw_metrics.get("Acc_2", 0.0) - residual_metrics.get("Acc_2", 0.0)
        f1_drop = raw_metrics.get("F1_score", 0.0) - residual_metrics.get("F1_score", 0.0)
        corr_drop = raw_metrics.get("Corr", 0.0) - residual_metrics.get("Corr", 0.0)
        mae_delta = residual_metrics.get("MAE", float("inf")) - raw_metrics.get("MAE", float("inf"))
        failed = (
            mae_delta >= -1e-12
            or acc2_drop > float(_get_arg(self.args, "retention_acc2_drop_limit", 0.001))
            or f1_drop > float(_get_arg(self.args, "retention_f1_drop_limit", 0.001))
            or corr_drop > float(_get_arg(self.args, "retention_corr_drop_limit", 0.001))
        )
        guard_info = {
            "epoch": int(self._current_epoch),
            "mode": str(mode).upper(),
            "guard_metric": str(_get_arg(self.args, "retention_guard_metric", "mae_acc2_f1_corr")),
            "fallback": bool(failed),
            "raw_Acc_2": raw_metrics.get("Acc_2", np.nan),
            "residual_Acc_2": residual_metrics.get("Acc_2", np.nan),
            "raw_F1_score": raw_metrics.get("F1_score", np.nan),
            "residual_F1_score": residual_metrics.get("F1_score", np.nan),
            "raw_Corr": raw_metrics.get("Corr", np.nan),
            "residual_Corr": residual_metrics.get("Corr", np.nan),
            "raw_MAE": raw_metrics.get("MAE", np.nan),
            "residual_MAE": residual_metrics.get("MAE", np.nan),
            "acc2_drop": acc2_drop,
            "f1_drop": f1_drop,
            "corr_drop": corr_drop,
            "mae_delta": mae_delta,
        }
        return (raw_pred if failed else residual_pred), guard_info

    def _fit_prediction_calibration(self, pred, true, mode, epoch_num=None):
        pred_vec = pred.detach().float().view(-1)
        true_vec = true.detach().float().view(-1)
        eps = 1e-8
        pred_mean = pred_vec.mean()
        true_mean = true_vec.mean()
        centered_pred = pred_vec - pred_mean
        centered_true = true_vec - true_mean
        pred_var = torch.mean(centered_pred * centered_pred)
        pred_std = torch.sqrt(pred_var.clamp(min=0.0))
        true_std = torch.sqrt(torch.mean(centered_true * centered_true).clamp(min=0.0))
        objective = str(_get_arg(self.args, "calibration_objective", "mse")).lower()
        warmup_epochs = max(0, int(_get_arg(self.args, "calibration_warmup_epochs", 0) or 0))
        if epoch_num is not None and warmup_epochs > 0 and int(epoch_num) <= warmup_epochs:
            return {
                "scale": 1.0,
                "bias": 0.0,
                "objective": objective,
                "mode": str(mode).upper(),
                "epoch": int(epoch_num),
                "pred_mean": float(pred_mean.detach().cpu().item()),
                "pred_std": float(pred_std.detach().cpu().item()),
                "true_mean": float(true_mean.detach().cpu().item()),
                "true_std": float(true_std.detach().cpu().item()),
                "used_identity": True,
                "warmup_identity": True,
            }
        min_pred_std = float(_get_arg(self.args, "calibration_min_pred_std", 1e-4))
        extra_calibration = {}
        if float(pred_std.item()) <= min_pred_std:
            scale = torch.ones((), device=pred.device, dtype=pred.dtype)
            bias = torch.zeros((), device=pred.device, dtype=pred.dtype)
            used_identity = True
        else:
            covariance = torch.mean(centered_pred * centered_true)
            scale = covariance / pred_var.clamp(min=eps)
            if not bool(_get_arg(self.args, "calibration_allow_negative_scale", False)):
                scale = scale.clamp(min=0.0)
            scale_min = float(_get_arg(self.args, "calibration_scale_min", 0.5))
            scale_max = float(_get_arg(self.args, "calibration_scale_max", 2.0))
            scale = scale.clamp(min=scale_min, max=scale_max)
            bias = true_mean - scale * pred_mean
            used_identity = False
            if objective in {
                "mae",
                "mae_grid",
                "safe_mae_grid",
                "sign_safe_scale",
            }:
                scale, bias = self._refine_calibration_for_mae(
                    pred_vec,
                    true_vec,
                    scale,
                    bias,
                )
            if objective in {
                "metric_grid",
                "safe_metric_grid",
                "acc3_safe_grid",
            }:
                if objective == "acc3_safe_grid":
                    scale, bias = self._refine_calibration_for_acc3_safe_grid(
                        pred_vec,
                        true_vec,
                        scale,
                        bias,
                    )
                else:
                    scale, bias = self._refine_calibration_for_metric_grid(
                        pred_vec,
                        true_vec,
                        scale,
                        bias,
                    )
            if objective == "threshold_mae_grid":
                scale, bias = self._refine_calibration_for_threshold_mae_grid(
                    pred_vec,
                    true_vec,
                    scale,
                    bias,
                )
            if objective == "threshold_mae_grid_v2":
                scale, bias = self._refine_calibration_for_threshold_mae_grid_v2(
                    pred_vec,
                    true_vec,
                )
            if objective == "negative_safe_scale":
                negative_scale, positive_scale = self._refine_calibration_for_negative_safe_scale(
                    pred_vec,
                    true_vec,
                )
                scale = torch.ones((), device=pred.device, dtype=pred.dtype)
                bias = torch.zeros((), device=pred.device, dtype=pred.dtype)
                extra_calibration = {
                    "negative_scale": float(negative_scale.detach().cpu().item()),
                    "positive_scale": float(positive_scale.detach().cpu().item()),
                }
        bias_freeze_epochs = max(0, int(_get_arg(self.args, "calibration_bias_freeze_epochs", 0) or 0))
        if (
            epoch_num is not None
            and bias_freeze_epochs > 0
            and int(epoch_num) <= bias_freeze_epochs
        ):
            bias = torch.zeros((), device=pred.device, dtype=pred.dtype)
        bias_objective = str(_get_arg(self.args, "bias_correction_objective", "none")).lower()
        if bias_objective == "val_safe_bias":
            bias_delta = self._refine_bias_correction(pred_vec, true_vec, scale, bias)
            bias = bias + bias_delta
            extra_calibration["bias_correction"] = float(bias_delta.detach().cpu().item())
            extra_calibration["bias_correction_objective"] = bias_objective
        calibration = {
            "scale": float(scale.detach().cpu().item()),
            "bias": float(bias.detach().cpu().item()),
            "objective": objective,
            "mode": str(mode).upper(),
            "epoch": int(epoch_num) if epoch_num is not None else -1,
            "pred_mean": float(pred_mean.detach().cpu().item()),
            "pred_std": float(pred_std.detach().cpu().item()),
            "true_mean": float(true_mean.detach().cpu().item()),
            "true_std": float(true_std.detach().cpu().item()),
            "used_identity": bool(used_identity),
        }
        calibration.update(extra_calibration)
        return calibration

    def _refine_calibration_for_mae(self, pred_vec, true_vec, scale, bias):
        scale_steps = max(3, int(_get_arg(self.args, "calibration_grid_steps", 11)))
        bias_steps = max(3, int(_get_arg(self.args, "calibration_bias_steps", 21)))
        scale_span = max(0.0, float(_get_arg(self.args, "calibration_scale_span", 0.25)))
        bias_span = max(0.0, float(_get_arg(self.args, "calibration_bias_span", 0.25)))
        scale_min = float(_get_arg(self.args, "calibration_scale_min", 0.5))
        scale_max = float(_get_arg(self.args, "calibration_scale_max", 2.0))

        base_scale = float(scale.detach().cpu().item())
        base_bias = float(bias.detach().cpu().item())
        lo = max(scale_min, base_scale * (1.0 - scale_span))
        hi = min(scale_max, base_scale * (1.0 + scale_span))
        if hi < lo:
            lo, hi = hi, lo
        scales = torch.linspace(lo, hi, steps=scale_steps, device=pred_vec.device, dtype=pred_vec.dtype)

        true_std = float(true_vec.std(unbiased=False).detach().cpu().item())
        bias_radius = 0.0 if bias_span <= 0.0 else max(1e-6, bias_span * max(true_std, 1e-6))
        biases = torch.linspace(
            base_bias - bias_radius,
            base_bias + bias_radius,
            steps=bias_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )

        clip_outputs = bool(_get_arg(self.args, "calibration_clip_outputs", False))
        clip_min, clip_max = self._calibration_bounds()
        best_scale = scale
        best_bias = bias
        best_mae = None
        for candidate_scale in scales:
            calibrated = pred_vec.unsqueeze(0) * candidate_scale + biases.unsqueeze(1)
            if clip_outputs:
                calibrated = calibrated.clamp(min=clip_min, max=clip_max)
            mae = (calibrated - true_vec.unsqueeze(0)).abs().mean(dim=1)
            cur_mae, cur_idx = torch.min(mae, dim=0)
            if best_mae is None or float(cur_mae.item()) < float(best_mae.item()):
                best_mae = cur_mae
                best_scale = candidate_scale
                best_bias = biases[cur_idx]
        if str(_get_arg(self.args, "calibration_objective", "mse")).lower() in {
            "safe_mae_grid",
            "sign_safe_scale",
        }:
            bias_limit = float(_get_arg(self.args, "calibration_bias_limit", 0.3))
            if bias_limit >= 0.0:
                best_bias = best_bias.clamp(min=-bias_limit, max=bias_limit)
        return best_scale, best_bias

    def _refine_calibration_for_negative_safe_scale(self, pred_vec, true_vec):
        scale_steps = max(3, int(_get_arg(self.args, "calibration_grid_steps", 11)))
        negative_min = float(_get_arg(self.args, "negative_scale_min", 1.0))
        negative_max = float(_get_arg(self.args, "negative_scale_max", 1.6))
        positive_max = float(_get_arg(self.args, "positive_scale_max", 1.05))
        negative_min = max(0.0, negative_min)
        negative_max = max(negative_min, negative_max)
        positive_max = max(1.0, positive_max)
        negative_scales = torch.linspace(
            negative_min,
            negative_max,
            steps=scale_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )
        positive_scales = torch.linspace(
            1.0,
            positive_max,
            steps=scale_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )

        best_negative_scale = torch.ones((), device=pred_vec.device, dtype=pred_vec.dtype)
        best_positive_scale = torch.ones((), device=pred_vec.device, dtype=pred_vec.dtype)
        best_mae = None
        for negative_scale in negative_scales:
            negative_candidate = torch.where(pred_vec < 0.0, pred_vec * negative_scale, pred_vec)
            for positive_scale in positive_scales:
                calibrated = torch.where(
                    pred_vec >= 0.0,
                    negative_candidate * positive_scale,
                    negative_candidate,
                )
                mae = (calibrated - true_vec).abs().mean()
                if best_mae is None or float(mae.item()) < float(best_mae.item()):
                    best_mae = mae
                    best_negative_scale = negative_scale
                    best_positive_scale = positive_scale
        return best_negative_scale, best_positive_scale

    def _refine_bias_correction(self, pred_vec, true_vec, scale, bias):
        limit = max(0.0, float(_get_arg(self.args, "bias_correction_limit", 0.0)))
        if limit <= 0.0:
            return torch.zeros((), device=pred_vec.device, dtype=pred_vec.dtype)
        steps = max(3, int(_get_arg(self.args, "bias_correction_steps", 21)))
        calibrated = pred_vec * scale + bias
        bias_candidates = torch.linspace(
            -limit,
            limit,
            steps=steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )
        mae = (calibrated.unsqueeze(0) + bias_candidates.unsqueeze(1) - true_vec.unsqueeze(0)).abs().mean(dim=1)
        _, best_idx = torch.min(mae, dim=0)
        return bias_candidates[best_idx]

    def _calibration_metric_score(self, pred, true):
        metrics = self._canonicalize_metrics(self.metrics(pred, true))
        corr = metrics.get("Corr", 0.0)
        if not np.isfinite(corr):
            corr = -1.0
        weights = {
            "Acc_7": float(_get_arg(self.args, "calibration_metric_acc7_weight", 1.0)),
            "Acc_5": float(_get_arg(self.args, "calibration_metric_acc5_weight", 1.0)),
            "Acc_3": float(_get_arg(self.args, "calibration_metric_acc3_weight", 0.5)),
            "Acc_2": float(_get_arg(self.args, "calibration_metric_acc2_weight", 1.0)),
            "F1_score": float(_get_arg(self.args, "calibration_metric_f1_weight", 1.0)),
            "Corr": float(_get_arg(self.args, "calibration_metric_corr_weight", 2.0)),
            "MAE": float(_get_arg(self.args, "calibration_metric_mae_weight", 2.0)),
        }
        score = 0.0
        score += weights["Acc_7"] * metrics.get("Acc_7", 0.0)
        score += weights["Acc_5"] * metrics.get("Acc_5", 0.0)
        score += weights["Acc_3"] * metrics.get("Acc_3", 0.0)
        score += weights["Acc_2"] * metrics.get("Acc_2", 0.0)
        score += weights["F1_score"] * metrics.get("F1_score", 0.0)
        score += weights["Corr"] * corr
        score -= weights["MAE"] * metrics.get("MAE", 0.0)
        return float(score)

    def _calibration_threshold_shortfall(self, metrics):
        checks = [
            ("Acc_7", "calibration_min_acc7", "min"),
            ("Acc_5", "calibration_min_acc5", "min"),
            ("Acc_3", "calibration_min_acc3", "min"),
            ("Acc_2", "calibration_min_acc2", "min"),
            ("F1_score", "calibration_min_f1", "min"),
            ("Corr", "calibration_min_corr", "min"),
            ("MAE", "calibration_max_mae", "max"),
        ]
        shortfall = 0.0
        for metric_name, arg_name, direction in checks:
            threshold = _get_arg(self.args, arg_name, None)
            if threshold is None:
                continue
            value = float(metrics.get(metric_name, 0.0))
            threshold = float(threshold)
            if direction == "min":
                shortfall += max(0.0, threshold - value)
            else:
                shortfall += max(0.0, value - threshold)
        return float(shortfall)

    def _refine_calibration_for_metric_grid(self, pred_vec, true_vec, scale, bias):
        scale_steps = max(3, int(_get_arg(self.args, "calibration_grid_steps", 11)))
        bias_steps = max(3, int(_get_arg(self.args, "calibration_bias_steps", 21)))
        scale_span = max(0.0, float(_get_arg(self.args, "calibration_scale_span", 0.25)))
        bias_span = max(0.0, float(_get_arg(self.args, "calibration_bias_span", 0.25)))
        scale_min = float(_get_arg(self.args, "calibration_scale_min", 0.5))
        scale_max = float(_get_arg(self.args, "calibration_scale_max", 2.0))

        base_scale = float(scale.detach().cpu().item())
        base_bias = float(bias.detach().cpu().item())
        lo = max(scale_min, base_scale * (1.0 - scale_span))
        hi = min(scale_max, base_scale * (1.0 + scale_span))
        if hi < lo:
            lo, hi = hi, lo
        scales = torch.linspace(lo, hi, steps=scale_steps, device=pred_vec.device, dtype=pred_vec.dtype)

        true_std = float(true_vec.std(unbiased=False).detach().cpu().item())
        bias_radius = max(0.0, bias_span * max(true_std, 1e-6))
        biases = torch.linspace(
            base_bias - bias_radius,
            base_bias + bias_radius,
            steps=bias_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )

        clip_outputs = bool(_get_arg(self.args, "calibration_clip_outputs", False))
        clip_min, clip_max = self._calibration_bounds()
        best_scale = scale
        best_bias = bias
        best_score = None
        for candidate_scale in scales:
            for candidate_bias in biases:
                calibrated = pred_vec * candidate_scale + candidate_bias
                if clip_outputs:
                    calibrated = calibrated.clamp(min=clip_min, max=clip_max)
                score = self._calibration_metric_score(calibrated, true_vec)
                if best_score is None or score > best_score:
                    best_score = score
                    best_scale = candidate_scale
                    best_bias = candidate_bias
        if str(_get_arg(self.args, "calibration_objective", "mse")).lower() == "safe_metric_grid":
            bias_limit = float(_get_arg(self.args, "calibration_bias_limit", 0.3))
            if bias_limit >= 0.0:
                best_bias = best_bias.clamp(min=-bias_limit, max=bias_limit)
        return best_scale, best_bias

    def _refine_calibration_for_acc3_safe_grid(self, pred_vec, true_vec, scale, bias):
        scale_steps = max(3, int(_get_arg(self.args, "calibration_grid_steps", 11)))
        bias_steps = max(3, int(_get_arg(self.args, "calibration_bias_steps", 21)))
        scale_span = max(0.0, float(_get_arg(self.args, "calibration_scale_span", 0.25)))
        bias_span = max(0.0, float(_get_arg(self.args, "calibration_bias_span", 0.25)))
        scale_min = float(_get_arg(self.args, "calibration_scale_min", 0.5))
        scale_max = float(_get_arg(self.args, "calibration_scale_max", 2.0))

        base_scale = float(scale.detach().cpu().item())
        base_bias = float(bias.detach().cpu().item())
        lo = max(scale_min, base_scale * (1.0 - scale_span))
        hi = min(scale_max, base_scale * (1.0 + scale_span))
        if hi < lo:
            lo, hi = hi, lo
        scales = torch.linspace(lo, hi, steps=scale_steps, device=pred_vec.device, dtype=pred_vec.dtype)

        true_std = float(true_vec.std(unbiased=False).detach().cpu().item())
        bias_radius = max(0.0, bias_span * max(true_std, 1e-6))
        biases = torch.linspace(
            base_bias - bias_radius,
            base_bias + bias_radius,
            steps=bias_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )

        clip_outputs = bool(_get_arg(self.args, "calibration_clip_outputs", False))
        clip_min, clip_max = self._calibration_bounds()
        target = _get_arg(self.args, "calibration_acc3_target", None)
        target = None if target is None else float(target)
        base_pred = pred_vec * scale + bias
        if clip_outputs:
            base_pred = base_pred.clamp(min=clip_min, max=clip_max)
        base_metrics = self._canonicalize_metrics(self.metrics(base_pred, true_vec))
        base_mae = float(base_metrics.get("MAE", 0.0))
        best_scale = scale
        best_bias = bias
        best_tuple = None
        for candidate_scale in scales:
            for candidate_bias in biases:
                calibrated = pred_vec * candidate_scale + candidate_bias
                if clip_outputs:
                    calibrated = calibrated.clamp(min=clip_min, max=clip_max)
                metrics = self._canonicalize_metrics(self.metrics(calibrated, true_vec))
                acc3 = float(metrics.get("Acc_3", 0.0))
                mae = float(metrics.get("MAE", 0.0))
                if bool(_get_arg(self.args, "calibration_mae_drop_guard", False)) and mae > base_mae:
                    continue
                target_shortfall = max(0.0, (target if target is not None else acc3) - acc3)
                shortfall = self._calibration_threshold_shortfall(metrics)
                metric_score = self._calibration_metric_score(calibrated, true_vec)
                objective_tuple = (
                    target_shortfall,
                    shortfall,
                    -acc3,
                    mae,
                    -metric_score,
                )
                if best_tuple is None or objective_tuple < best_tuple:
                    best_tuple = objective_tuple
                    best_scale = candidate_scale
                    best_bias = candidate_bias
        return best_scale, best_bias

    def _refine_calibration_for_threshold_mae_grid(self, pred_vec, true_vec, scale, bias):
        scale_steps = max(3, int(_get_arg(self.args, "calibration_grid_steps", 11)))
        bias_steps = max(3, int(_get_arg(self.args, "calibration_bias_steps", 21)))
        scale_span = max(0.0, float(_get_arg(self.args, "calibration_scale_span", 0.25)))
        bias_span = max(0.0, float(_get_arg(self.args, "calibration_bias_span", 0.25)))
        scale_min = float(_get_arg(self.args, "calibration_scale_min", 0.5))
        scale_max = float(_get_arg(self.args, "calibration_scale_max", 2.0))
        penalty_weight = max(
            1.0,
            float(_get_arg(self.args, "calibration_threshold_penalty_weight", 100.0)),
        )

        base_scale = float(scale.detach().cpu().item())
        base_bias = float(bias.detach().cpu().item())
        lo = max(scale_min, base_scale * (1.0 - scale_span))
        hi = min(scale_max, base_scale * (1.0 + scale_span))
        if hi < lo:
            lo, hi = hi, lo
        scales = torch.linspace(lo, hi, steps=scale_steps, device=pred_vec.device, dtype=pred_vec.dtype)

        true_std = float(true_vec.std(unbiased=False).detach().cpu().item())
        bias_radius = max(0.0, bias_span * max(true_std, 1e-6))
        biases = torch.linspace(
            base_bias - bias_radius,
            base_bias + bias_radius,
            steps=bias_steps,
            device=pred_vec.device,
            dtype=pred_vec.dtype,
        )

        clip_outputs = bool(_get_arg(self.args, "calibration_clip_outputs", False))
        clip_min, clip_max = self._calibration_bounds()
        best_scale = scale
        best_bias = bias
        best_tuple = None
        for candidate_scale in scales:
            for candidate_bias in biases:
                calibrated = pred_vec * candidate_scale + candidate_bias
                if clip_outputs:
                    calibrated = calibrated.clamp(min=clip_min, max=clip_max)
                metrics = self._canonicalize_metrics(self.metrics(calibrated, true_vec))
                mae = float(metrics.get("MAE", 0.0))
                shortfall = self._calibration_threshold_shortfall(metrics)
                metric_score = self._calibration_metric_score(calibrated, true_vec)
                objective_tuple = (
                    shortfall,
                    mae + penalty_weight * shortfall,
                    -metric_score,
                )
                if best_tuple is None or objective_tuple < best_tuple:
                    best_tuple = objective_tuple
                    best_scale = candidate_scale
                    best_bias = candidate_bias
        return best_scale, best_bias

    def _refine_calibration_for_threshold_mae_grid_v2(self, pred_vec, true_vec):
        step = max(1e-6, float(_get_arg(self.args, "threshold_mae_grid_step", 0.005)))
        scale_min = float(_get_arg(self.args, "threshold_mae_grid_scale_min", 0.95))
        scale_max = float(_get_arg(self.args, "threshold_mae_grid_scale_max", 1.10))
        bias_min = float(_get_arg(self.args, "threshold_mae_grid_bias_min", -0.03))
        bias_max = float(_get_arg(self.args, "threshold_mae_grid_bias_max", 0.03))
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        if bias_max < bias_min:
            bias_min, bias_max = bias_max, bias_min

        scale_count = int(round((scale_max - scale_min) / step)) + 1
        bias_count = int(round((bias_max - bias_min) / step)) + 1
        grid_dtype = torch.float64
        pred_work = pred_vec.to(dtype=grid_dtype)
        true_work = true_vec.to(dtype=grid_dtype)
        scale_values = scale_min + torch.arange(
            max(1, scale_count),
            device=pred_vec.device,
            dtype=grid_dtype,
        ) * step
        bias_values = bias_min + torch.arange(
            max(1, bias_count),
            device=pred_vec.device,
            dtype=grid_dtype,
        ) * step
        scale_values = scale_values.clamp(min=scale_min, max=scale_max)
        bias_values = bias_values.clamp(min=bias_min, max=bias_max)

        clip_outputs = bool(_get_arg(self.args, "calibration_clip_outputs", False))
        clip_min, clip_max = self._calibration_bounds()
        best_scale = scale_values[0]
        best_bias = bias_values[0]
        best_tuple = None
        for candidate_scale in scale_values:
            calibrated = pred_work.unsqueeze(0) * candidate_scale + bias_values.unsqueeze(1)
            if clip_outputs:
                calibrated = calibrated.clamp(min=clip_min, max=clip_max)
            for row_idx, candidate_bias in enumerate(bias_values):
                candidate = calibrated[row_idx]
                metrics = self._canonicalize_metrics(self.metrics(candidate, true_work))
                mae = float(metrics.get("MAE", 0.0))
                shortfall = self._calibration_threshold_shortfall(metrics)
                metric_score = self._calibration_metric_score(candidate, true_work)
                objective_tuple = (shortfall, mae, -metric_score)
                if best_tuple is None or objective_tuple < best_tuple:
                    best_tuple = objective_tuple
                    best_scale = candidate_scale
                    best_bias = candidate_bias
        return best_scale, best_bias

    def _apply_prediction_calibration(self, pred, calibration):
        if not calibration:
            return pred
        scale = torch.as_tensor(
            float(calibration["scale"]), device=pred.device, dtype=pred.dtype
        )
        bias = torch.as_tensor(
            float(calibration["bias"]), device=pred.device, dtype=pred.dtype
        )
        objective = str(calibration.get("objective", "")).lower()
        if objective == "negative_safe_scale":
            negative_scale = torch.as_tensor(
                float(calibration.get("negative_scale", 1.0)),
                device=pred.device,
                dtype=pred.dtype,
            )
            positive_scale = torch.as_tensor(
                float(calibration.get("positive_scale", 1.0)),
                device=pred.device,
                dtype=pred.dtype,
            )
            calibrated = torch.where(pred < 0.0, pred * negative_scale, pred * positive_scale)
            if float(scale.item()) != 1.0 or float(bias.item()) != 0.0:
                calibrated = calibrated * scale + bias
        else:
            calibrated = pred * scale + bias
        blend = float(_get_arg(self.args, "calibration_blend", 1.0))
        blend = max(0.0, min(1.0, blend))
        if blend < 1.0:
            calibrated = pred + blend * (calibrated - pred)
        if bool(_get_arg(self.args, "calibration_clip_outputs", False)):
            clip_min, clip_max = self._calibration_bounds()
            calibrated = calibrated.clamp(min=clip_min, max=clip_max)
        if bool(_get_arg(self.args, "calibration_sign_preserve", False)):
            sign_changed = (pred.view_as(calibrated) * calibrated) < 0
            calibrated = torch.where(sign_changed, pred, calibrated)
        return calibrated

    def _should_log_error_diagnostics(self, mode):
        if not self._is_arg_msa_sims_family():
            return False
        if not bool(_get_arg(self.args, "enable_arg_msa_error_diagnostics", False)):
            return False
        return str(mode).upper() in {"VAL", "VALID"}

    def _sanitize_metric_dict(self, metric_dict, mode, epoch_num=None, pred=None, true=None):
        sanitized = {}
        pred_std = None
        true_std = None
        if pred is not None and torch.is_tensor(pred):
            pred_std = float(pred.detach().float().std(unbiased=False).cpu().item())
        if true is not None and torch.is_tensor(true):
            true_std = float(true.detach().float().std(unbiased=False).cpu().item())

        for key, value in metric_dict.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()

            try:
                value_float = float(value)
            except (TypeError, ValueError):
                sanitized[key] = value
                continue

            if not np.isfinite(value_float):
                logger.warning(
                    "Non-finite metric detected and replaced with 0.0: mode=%s epoch=%s key=%s value=%s pred_std=%s true_std=%s",
                    mode,
                    epoch_num if epoch_num is not None else "NA",
                    key,
                    value,
                    pred_std,
                    true_std,
                )
                value_float = 0.0
            sanitized[key] = value_float
        return sanitized

    def _iter_analysis_matches_mode(self, mode):
        split = str(_get_arg(self.args, "iter_analysis_split", "valid")).lower()
        split_aliases = {
            "val": {"VAL", "VALID"},
            "valid": {"VAL", "VALID"},
            "test": {"TEST"},
            "train": {"TRAIN"},
        }
        valid_modes = split_aliases.get(split, {split.upper()})
        return mode.upper() in valid_modes

    def _should_run_iter_analysis(self, mode, epoch_num=None):
        if self.args.model_name != "arg_msa":
            return False
        if not bool(_get_arg(self.args, "enable_iter_analysis", False)):
            return False
        if not self._iter_analysis_matches_mode(mode):
            return False
        every_epoch = max(1, int(_get_arg(self.args, "iter_analysis_every_epoch", 1)))
        if epoch_num is not None and epoch_num % every_epoch != 0:
            return False
        return True

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = self._build_optimizer(model)
        grad_clip = float(_get_arg(self.args, "grad_clip", 1.0))
        show_progress = bool(_get_arg(self.args, "show_progress", True))

        best_epoch = 0
        if return_epoch_results:
            epoch_results = {"train": [], "valid": [], "test": []}

        min_or_max = "min" if self.args.KeyEval in ["Loss"] else "max"
        best_valid = 1e8 if min_or_max == "min" else 0.0

        for epoch_idx in range(self.args.epochs):
            epoch_num = epoch_idx + 1
            self._current_epoch = epoch_num
            target_model = model.module if hasattr(model, "module") else model
            if hasattr(target_model, "set_training_epoch"):
                target_model.set_training_epoch(epoch_num)
            model.train()
            y_pred, y_true = [], []
            train_loss = 0.0
            iterator = tqdm(dataloader["train"], disable=not show_progress)
            for batch_data in iterator:
                labels = self._prepare_labels(batch_data["labels"]["M"].to(self.args.device))
                optimizer.zero_grad()
                outputs = self._forward_model(
                    model,
                    batch_data,
                    collect_layer_trace=self._should_log_layerwise_trace("TRAIN", epoch_num),
                )
                loss = self._compute_loss(
                    outputs["M"],
                    labels,
                    use_auxiliary=True,
                    outputs=outputs,
                )
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                optimizer.step()

                train_loss += loss.item()
                y_pred.append(outputs["M"].detach().cpu())
                y_true.append(labels.detach().cpu())

            train_loss = train_loss / max(1, len(dataloader["train"]))
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self._sanitize_metric_dict(
                self.metrics(pred, true),
                mode="TRAIN",
                epoch_num=epoch_num,
                pred=pred,
                true=true,
            )
            logger.info(
                f">> Epoch:{epoch_num} :"
                f"TRAIN-({self.args.model_name}) [{epoch_num - best_epoch}/{epoch_num}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 8)} {dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader["valid"], mode="VAL", epoch_num=epoch_num)
            cur_valid = val_results[self.args.KeyEval]
            if epoch_num == self.args.epochs:
                save_path = (
                    self.args.model_save_dir
                    / str(self.args.dataset_name)
                    / f"{self.args.model_name}_{self.args.dataset_name}_{epoch_num}.pth"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

            is_better = (
                cur_valid <= (best_valid - 1e-6)
                if min_or_max == "min"
                else cur_valid >= (best_valid + 1e-6)
            )
            if is_better:
                best_valid, best_epoch = cur_valid, epoch_num
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                train_results["LossMean"] = train_loss
                train_results["TrainBatches"] = len(dataloader["train"])
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(val_results)
                if bool(_get_arg(self.args, "skip_epoch_test", False)):
                    test_results = {}
                else:
                    test_results = self.do_test(
                        model, dataloader["test"], mode="TEST", epoch_num=epoch_num
                    )
                epoch_results["test"].append(test_results)

        return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, epoch_num=None):
        show_progress = bool(_get_arg(self.args, "show_progress", True))
        model.eval()
        y_pred, y_true = [], []
        retention_raw_pred = []
        eval_loss = 0.0
        feature_keys = ["Feature_t", "Feature_a", "Feature_v", "Feature_f"]
        iter_analysis_enabled = self._should_run_iter_analysis(mode, epoch_num)
        error_diag_enabled = self._should_log_error_diagnostics(mode)
        # Preserve the original trace switches: optional losses and validation
        # guards consume these tensors even though reporting has been removed.
        collect_layer_trace = (
            self._should_log_layerwise_trace(mode, epoch_num)
            or error_diag_enabled
            or self._should_log_method_chain_diagnostics(mode, epoch_num)
        )

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {key: [] for key in feature_keys}

        with torch.no_grad():
            iterator = tqdm(dataloader, disable=not show_progress)
            for batch_data in iterator:
                labels = self._prepare_labels(batch_data["labels"]["M"].to(self.args.device))
                outputs = self._forward_model(
                    model,
                    batch_data,
                    collect_iter_history=iter_analysis_enabled,
                    collect_layer_trace=collect_layer_trace,
                )
                loss = self._compute_loss(outputs["M"], labels, use_auxiliary=False)

                eval_loss += loss.item()
                y_pred.append(outputs["M"].detach().cpu())
                y_true.append(labels.detach().cpu())
                layer_trace = outputs.get("layer_trace", {})
                raw_retention = layer_trace.get("retention_output_raw_pred") if isinstance(layer_trace, dict) else None
                if torch.is_tensor(raw_retention):
                    retention_raw_pred.append(raw_retention.detach().cpu())
                if return_sample_results:
                    ids.extend(batch_data["id"])
                    all_labels.extend(labels.cpu().tolist())
                    sample_results.extend(np.atleast_1d(outputs["M"].cpu().numpy().squeeze()))
                    for key in feature_keys:
                        if key in outputs:
                            features[key].append(outputs[key].cpu().numpy())

        eval_loss = eval_loss / max(1, len(dataloader))
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        raw_retention_pred = None
        if retention_raw_pred:
            raw_retention_pred = torch.cat(retention_raw_pred)
            if raw_retention_pred.shape == pred.shape:
                pred, guard_info = self._apply_retention_residual_safe_guard(
                    raw_retention_pred,
                    pred,
                    true,
                    mode,
                )
                if guard_info and guard_info.get("fallback"):
                    eval_loss = float(self.criterion(pred, true).item())
        calibration = None
        if self._calibration_enabled() and mode.upper() in {"VAL", "VALID", "TEST"}:
            if mode.upper() in {"VAL", "VALID"}:
                calibration = self._fit_prediction_calibration(
                    pred, true, mode, epoch_num=epoch_num
                )
            elif bool(_get_arg(self.args, "calibration_apply_to_test", True)):
                calibration = self._last_val_calibration
            if calibration is not None:
                candidate_pred = self._apply_prediction_calibration(pred, calibration)
                objective = str(calibration.get("objective", "")).lower()
                bias_objective = str(
                    calibration.get(
                        "bias_correction_objective",
                        _get_arg(self.args, "bias_correction_objective", "none"),
                    )
                ).lower()
                if mode.upper() in {"VAL", "VALID"} and (objective in {
                    "safe_mae_grid",
                    "sign_safe_scale",
                    "safe_metric_grid",
                    "threshold_mae_grid",
                    "threshold_mae_grid_v2",
                    "negative_safe_scale",
                    "acc3_safe_grid",
                } or bias_objective == "val_safe_bias"):
                    raw_metrics = self._canonicalize_metrics(self.metrics(pred, true))
                    candidate_metrics = self._canonicalize_metrics(
                        self.metrics(candidate_pred, true)
                    )
                    acc_tol = self._calibration_drop_tolerance("acc2", 0.005)
                    acc7_tol = self._calibration_drop_tolerance("acc7", 1.0)
                    acc5_tol = self._calibration_drop_tolerance("acc5", 1.0)
                    acc3_tol = self._calibration_drop_tolerance("acc3", 1.0)
                    f1_tol = self._calibration_drop_tolerance("f1", 0.005)
                    corr_tol = self._calibration_drop_tolerance("corr", 1.0)
                    acc7_drop = raw_metrics.get("Acc_7", 0.0) - candidate_metrics.get("Acc_7", 0.0)
                    acc5_drop = raw_metrics.get("Acc_5", 0.0) - candidate_metrics.get("Acc_5", 0.0)
                    acc_drop = raw_metrics.get("Acc_2", 0.0) - candidate_metrics.get("Acc_2", 0.0)
                    acc3_drop = raw_metrics.get("Acc_3", 0.0) - candidate_metrics.get("Acc_3", 0.0)
                    f1_drop = raw_metrics.get("F1_score", 0.0) - candidate_metrics.get("F1_score", 0.0)
                    corr_drop = raw_metrics.get("Corr", 0.0) - candidate_metrics.get("Corr", 0.0)
                    mae_drop = candidate_metrics.get("MAE", 0.0) - raw_metrics.get("MAE", 0.0)
                    mae_guard_failed = (
                        bool(_get_arg(self.args, "calibration_mae_drop_guard", False))
                        and mae_drop > 1e-12
                    )
                    if (
                        acc7_drop > acc7_tol
                        or acc5_drop > acc5_tol
                        or acc_drop > acc_tol
                        or acc3_drop > acc3_tol
                        or f1_drop > f1_tol
                        or corr_drop > corr_tol
                        or mae_guard_failed
                    ):
                        calibration = dict(calibration)
                        calibration["scale"] = 1.0
                        calibration["bias"] = 0.0
                        calibration["negative_scale"] = 1.0
                        calibration["positive_scale"] = 1.0
                        calibration["bias_correction"] = 0.0
                        calibration["used_identity"] = True
                        calibration["safe_fallback"] = True
                        candidate_pred = pred
                    else:
                        calibration["safe_fallback"] = False
                pred = candidate_pred
                if bool(_get_arg(self.args, "calibration_affects_eval_loss", True)):
                    eval_loss = float(self.criterion(pred, true).item())
                if mode.upper() in {"VAL", "VALID"}:
                    # Freeze the validation decision, including any safe fallback.
                    # Test labels are used only to report metrics and loss.
                    self._last_val_calibration = calibration
        eval_results = self._sanitize_metric_dict(
            self.metrics(pred, true),
            mode=mode,
            epoch_num=epoch_num,
            pred=pred,
            true=true,
        )
        eval_results["Loss"] = round(eval_loss, 8)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["RawSResults"] = sample_results
            eval_results["SResults"] = list(np.atleast_1d(pred.cpu().numpy().squeeze()))
            eval_results["Features"] = {
                key: np.concatenate(value, axis=0) if value else np.array([])
                for key, value in features.items()
            }
            eval_results["Labels"] = all_labels

        return eval_results
