from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ["ARG_MSA"]


def _get_arg(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _config_bool_or_mode(value, default=True):
    if value is None:
        value = default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return text


def _config_enabled(value) -> bool:
    return bool(value)


def _config_auto_filter(value) -> bool:
    return isinstance(value, str) and value.lower() == "auto_filter"


def _resolve_num_heads(hidden_dim: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    for heads in range(requested_heads, 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def _sanitize_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return lengths.long().clamp(min=1, max=max_len)


def _infer_mask_and_lengths(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = sequence.abs().sum(dim=-1) > 1e-6
    lengths = mask.long().sum(dim=1)
    zero_length = lengths == 0
    if zero_length.any():
        mask = mask.clone()
        mask[zero_length, 0] = True
        lengths = lengths.clamp(min=1)
    return mask, lengths


def _prepare_mask_and_lengths(
    sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = sequence.size(1)
    if lengths is None:
        return _infer_mask_and_lengths(sequence)
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, device=sequence.device)
    lengths = _sanitize_lengths(lengths.to(sequence.device), max_len)
    mask = _lengths_to_mask(lengths, max_len)
    return mask, lengths


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def _masked_weighted_mean(x: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = weights * mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1e-6)
    return (x * weights).sum(dim=1) / denom


def _apply_q_floor(
    q: torch.Tensor,
    q_floor: float,
    mode: str = "absolute",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_floor = float(max(0.0, min(1.0, q_floor)))
    if q_floor <= 0.0:
        return q
    mode = str(mode).lower()
    if mode == "relative":
        if mask is None:
            q_min = q.amin(dim=1, keepdim=True)
            q_max = q.amax(dim=1, keepdim=True)
        else:
            valid = mask.bool().unsqueeze(-1)
            q_min = q.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
            q_max = q.masked_fill(~valid, float("-inf")).amax(dim=1, keepdim=True)
            q_min = torch.where(torch.isfinite(q_min), q_min, torch.zeros_like(q_min))
            q_max = torch.where(torch.isfinite(q_max), q_max, q_min)
        q_relative = (q - q_min) / (q_max - q_min).clamp(min=1e-6)
        out = q_floor + (1.0 - q_floor) * q_relative
        if mask is not None:
            out = out * mask.float().unsqueeze(-1)
        return out
    # Preserve q's sample-level differences while preventing all-near-zero
    # acceptability from erasing local evidence in weighted aggregation.
    return q_floor + (1.0 - q_floor) * q


def _expand_global(g: torch.Tensor, steps: int) -> torch.Tensor:
    return g.unsqueeze(1).expand(-1, steps, -1)


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class UnimodalSequenceEncoder(nn.Module):
    """
    Input:
        sequence: [batch, seq_len, input_dim]
        lengths:  [batch]
    Output:
        encoded:  [batch, seq_len, hidden_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        rnn_hidden = max(1, hidden_dim // 2)
        # SIMS raw visual/audio features can have very large outliers. Normalizing
        # each timestep before projection prevents GRU gate saturation while
        # preserving the unimodal-local-encoder design.
        # This is an input stabilizer, not a learnable modeling component.
        # Keep affine disabled so the model cannot learn gamma/beta ~= 0 and
        # silently shut off an entire modality on SIMS/SIMSV2.
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.output_proj = nn.Linear(rnn_hidden * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        return_trace: bool = False,
    ) -> torch.Tensor:
        raw_sequence = sequence
        clean_sequence = torch.nan_to_num(
            raw_sequence.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        normed_sequence = self.input_norm(clean_sequence)
        projected = self.input_proj(normed_sequence)
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.rnn(packed)
        rnn_outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=sequence.size(1)
        )
        outputs = self.output_proj(rnn_outputs)
        outputs = self.norm(outputs)
        outputs = self.dropout(F.relu(outputs))
        if return_trace:
            return outputs, {
                "raw": raw_sequence,
                "clean": clean_sequence,
                "norm": normed_sequence,
                "gru_input": projected,
                "gru_output": rnn_outputs,
                "encoded": outputs,
            }
        return outputs


class LocalMultimodalFusion(nn.Module):
    """
    Inputs:
        text_feat:   [batch, text_len, hidden_dim]
        audio_feat:  [batch, audio_len, hidden_dim]
        vision_feat: [batch, vision_len, hidden_dim]
        text_mask:   [batch, text_len]
        audio_mask:  [batch, audio_len]
        vision_mask: [batch, vision_len]
    Outputs:
        h_t:         [batch, text_len, hidden_dim]
        h_mask:      [batch, text_len]
    """

    def __init__(self, hidden_dim: int, dropout: float, num_heads: int):
        super().__init__()
        heads = _resolve_num_heads(hidden_dim, num_heads)
        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.fuse = _build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _resize_sequence(
        sequence: torch.Tensor, mask: torch.Tensor, target_steps: int
    ) -> torch.Tensor:
        if sequence.size(1) == target_steps:
            return sequence * mask.float().unsqueeze(-1)

        mask_float = mask.float().unsqueeze(-1)
        masked_sequence = sequence * mask_float
        pooled_feat = F.adaptive_avg_pool1d(masked_sequence.transpose(1, 2), target_steps)
        pooled_mask = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), target_steps).squeeze(1)
        pooled_feat = pooled_feat.transpose(1, 2)

        # Average only over valid frames. Without this normalization, zero padding in
        # unaligned audio/vision can shrink modality features toward 0 and make
        # validation predictions look artificially constant on SIMS/SIMSV2.
        pooled_feat = pooled_feat / pooled_mask.unsqueeze(-1).clamp(min=1e-6)
        pooled_feat = pooled_feat * (pooled_mask > 0).unsqueeze(-1).float()
        return pooled_feat

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target_steps: int) -> torch.Tensor:
        if mask.size(1) == target_steps:
            return mask
        resized = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), target_steps).squeeze(1)
        return resized > 0.0

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_steps = text_feat.size(1)
        audio_aligned = self._resize_sequence(audio_feat, audio_mask, target_steps)
        vision_aligned = self._resize_sequence(vision_feat, vision_mask, target_steps)
        audio_aligned_mask = self._resize_mask(audio_mask, target_steps)
        vision_aligned_mask = self._resize_mask(vision_mask, target_steps)
        audio_aligned = audio_aligned * audio_aligned_mask.float().unsqueeze(-1)
        vision_aligned = vision_aligned * vision_aligned_mask.float().unsqueeze(-1)

        # h_t comes from per-position multimodal fusion after all three modalities
        # are projected into the same hidden space.
        fusion_input = torch.cat([text_feat, audio_aligned, vision_aligned], dim=-1)
        gate = torch.softmax(self.gate(fusion_input), dim=-1)
        gated_local = (
            gate[..., 0:1] * text_feat
            + gate[..., 1:2] * audio_aligned
            + gate[..., 2:3] * vision_aligned
        )
        fused = self.fuse(fusion_input) + gated_local
        attn_out, _ = self.attn(
            fused,
            fused,
            fused,
            key_padding_mask=~text_mask,
            need_weights=False,
        )
        h_t = self.norm(fused + self.dropout(attn_out))
        return h_t, text_mask


class SegmentBuilder(nn.Module):
    """
    Inputs:
        h_t:   [batch, seq_len, hidden_dim]
        mask:  [batch, seq_len]
    Outputs:
        a_i:   [batch, num_segments, hidden_dim]
        m_i:   [batch, num_segments]
        c_i:   [batch, num_segments, 1]
    """

    def __init__(
        self,
        segment_window: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        pooling: str = "mean",
        attention_temperature: float = 1.0,
    ):
        super().__init__()
        self.segment_window = max(1, int(segment_window))
        self.pooling = str(pooling).lower().strip()
        self.attention_temperature = float(max(1e-6, attention_temperature))
        if self.pooling not in {"mean", "attention"}:
            self.pooling = "mean"
        if self.pooling == "attention":
            if hidden_dim is None:
                raise ValueError("hidden_dim is required when segment pooling is attention")
            scorer_hidden = max(16, int(hidden_dim) // 2)
            self.attn_score = nn.Sequential(
                nn.Linear(int(hidden_dim), scorer_hidden),
                nn.Tanh(),
                nn.Dropout(float(dropout)),
                nn.Linear(scorer_hidden, 1),
            )
        else:
            self.attn_score = None

    def _pool_span(self, span_feat: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        if self.attn_score is None:
            return _masked_mean(span_feat, span_mask), None

        logits = self.attn_score(span_feat).squeeze(-1)
        valid = span_mask.bool()
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits / self.attention_temperature, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (span_feat * weights.unsqueeze(-1)).sum(dim=1)
        valid_count = valid.float().sum(dim=1).clamp(min=1.0)
        entropy = -(weights * torch.log(weights.clamp(min=1e-12))).sum(dim=1)
        norm = torch.log(valid_count).clamp(min=1e-6)
        entropy = torch.where(valid_count > 1.0, entropy / norm, torch.zeros_like(entropy))
        return pooled, entropy.unsqueeze(-1)

    def forward(self, h_t: torch.Tensor, mask: torch.Tensor):
        # a_i is built with fixed contiguous windows. The optional attention
        # pooling keeps the same argument-unit definition while letting the
        # model emphasize the most sentiment-bearing frames inside each unit.
        _, seq_len, _ = h_t.shape
        segment_reprs = []
        segment_masks = []
        clarity_scores = []
        attention_entropies = []
        for start in range(0, seq_len, self.segment_window):
            end = min(seq_len, start + self.segment_window)
            span_feat = h_t[:, start:end, :]
            span_mask = mask[:, start:end]
            pooled, attention_entropy = self._pool_span(span_feat, span_mask)
            segment_reprs.append(pooled)
            segment_masks.append(span_mask.any(dim=1))
            clarity_scores.append(span_mask.float().mean(dim=1, keepdim=True))
            if attention_entropy is not None:
                attention_entropies.append(attention_entropy)
        a_i = torch.stack(segment_reprs, dim=1)
        m_i = torch.stack(segment_masks, dim=1)
        c_i = torch.stack(clarity_scores, dim=1)
        attn_entropy = (
            torch.stack(attention_entropies, dim=1) if attention_entropies else None
        )
        return a_i, m_i, c_i, attn_entropy


class RoleClassifier(nn.Module):
    """
    Inputs:
        a_i: [batch, num_segments, hidden_dim]
        g:   [batch, hidden_dim]
    Output:
        p_i: [batch, num_segments, 3]
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = _build_mlp(hidden_dim * 4, hidden_dim, 3, dropout)

    def forward(self, a_i: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_expand = _expand_global(g, a_i.size(1))
        role_input = torch.cat(
            [a_i, g_expand, a_i * g_expand, torch.abs(a_i - g_expand)], dim=-1
        )
        return torch.softmax(self.mlp(role_input), dim=-1)


class RoleSetBuilder(nn.Module):
    """
    Inputs:
        a_i:     [batch, num_segments, hidden_dim]
        p_i:     [batch, num_segments, 3]
        segmask: [batch, num_segments]
    Outputs:
        role_sets:    [batch, 3, hidden_dim]
        role_weights: [batch, num_segments, 3]
    """

    def __init__(self, use_soft_role: bool = True):
        super().__init__()
        self.use_soft_role = use_soft_role

    def forward(self, a_i: torch.Tensor, p_i: torch.Tensor, segmask: torch.Tensor):
        if self.use_soft_role:
            role_weights = p_i
        else:
            role_weights = F.one_hot(p_i.argmax(dim=-1), num_classes=3).float()
        role_weights = role_weights * segmask.float().unsqueeze(-1)

        role_sets = []
        for role_idx in range(3):
            cur_weights = role_weights[..., role_idx : role_idx + 1]
            role_sets.append(_masked_weighted_mean(a_i, cur_weights, segmask))
        role_sets = torch.stack(role_sets, dim=1)
        return role_sets, role_weights


class NodeInitializer(nn.Module):
    """
    Inputs:
        a_i: [batch, num_segments, hidden_dim]
        p_i: [batch, num_segments, 3]
        g:   [batch, hidden_dim]
        c_i: [batch, num_segments, 1]
    Output:
        v_i: [batch, num_segments, hidden_dim]
    """

    def __init__(self, hidden_dim: int, dropout: float, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(max(0.0, residual_scale))
        self.mlp = _build_mlp(hidden_dim * 3 + 4, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def build_input(
        self, a_i: torch.Tensor, p_i: torch.Tensor, g: torch.Tensor, c_i: torch.Tensor
    ) -> torch.Tensor:
        g_expand = _expand_global(g, a_i.size(1))
        return torch.cat(
            [a_i, p_i, a_i * g_expand, torch.abs(a_i - g_expand), c_i], dim=-1
        )

    def forward(
        self,
        a_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        c_i: torch.Tensor,
        return_input: bool = False,
    ):
        node_input = self.build_input(a_i, p_i, g, c_i)
        projected = self.mlp(node_input)
        # Sample-level argument evidence is preserved as a configurable residual.
        # residual_scale=0 restores the original node initializer exactly.
        if self.residual_scale > 0.0:
            projected = projected + self.residual_scale * a_i
        node_state = self.norm(projected)
        if return_input:
            return node_state, node_input
        return node_state


class RelationGenerator(nn.Module):
    """
    Inputs:
        v_i:      [batch, num_segments, hidden_dim]
        p_i:      [batch, num_segments, 3]
        g:        [batch, hidden_dim]
        segmask:  [batch, num_segments]
    Outputs:
        S/A/U:    [batch, num_segments, num_segments]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        evidence_residual_scale: float = 0.1,
        temperature: float = 1.0,
        relation_confidence_residual: bool = False,
        relation_confidence_scale: float = 0.0,
    ):
        super().__init__()
        self.evidence_residual_scale = float(max(0.0, evidence_residual_scale))
        self.temperature = float(max(1e-6, temperature))
        self.relation_confidence_residual = bool(relation_confidence_residual)
        self.relation_confidence_scale = float(max(0.0, relation_confidence_scale))
        self.mlp_loc = _build_mlp(hidden_dim * 4, hidden_dim, hidden_dim, dropout)
        self.mlp_role = _build_mlp(12, hidden_dim, hidden_dim, dropout)
        self.mlp_glob = _build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.mlp_fuse = _build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.out = nn.Linear(hidden_dim, 3)
        self.evidence_residual = nn.Linear(hidden_dim * 4, 3, bias=False)

    def _apply_confidence_residual(
        self,
        relation_logits: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if not self.relation_confidence_residual or self.relation_confidence_scale <= 0.0:
            return relation_logits
        return relation_logits + self.relation_confidence_scale * confidence * torch.tanh(relation_logits)

    def forward(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        segmask: torch.Tensor,
        return_debug: bool = False,
    ):
        batch_size, num_nodes, hidden_dim = v_i.shape
        src = v_i.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        dst = v_i.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        src_role = p_i.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, 3)
        dst_role = p_i.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, 3)
        g_expand = g.unsqueeze(1).unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)

        r_loc = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)
        h_loc = self.mlp_loc(r_loc)

        r_role = torch.cat(
            [src_role, dst_role, src_role * dst_role, torch.abs(src_role - dst_role)], dim=-1
        )
        h_role = self.mlp_role(r_role)

        r_glob = torch.cat([g_expand, src * g_expand, dst * g_expand], dim=-1)
        h_glob = self.mlp_glob(r_glob)

        h_fuse = self.mlp_fuse(torch.cat([h_loc, h_role, h_glob], dim=-1))
        relation_logits = self.out(h_fuse)
        if self.evidence_residual_scale > 0.0:
            relation_logits = relation_logits + self.evidence_residual_scale * self.evidence_residual(r_loc)
        relation_confidence = h_fuse.norm(dim=-1, keepdim=True) / (hidden_dim ** 0.5)
        relation_logits = self._apply_confidence_residual(relation_logits, relation_confidence)
        relation_probs = torch.softmax(relation_logits / self.temperature, dim=-1)

        valid_pair_mask = segmask.unsqueeze(2) & segmask.unsqueeze(1)
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=v_i.device).unsqueeze(0)
        edge_mask = valid_pair_mask & diag_mask
        relation_probs = relation_probs * edge_mask.unsqueeze(-1).float()

        support, attack, undercut = relation_probs.unbind(dim=-1)
        if return_debug:
            return support, attack, undercut, {
                "relation_input": r_loc,
                "relation_input_loc": r_loc,
                "relation_input_role": r_role,
                "relation_input_global": r_glob,
                "relation_fuse": h_fuse,
                "relation_confidence": relation_confidence.squeeze(-1),
                "relation_logits": relation_logits,
                "support_logits": relation_logits[..., 0],
                "attack_logits": relation_logits[..., 1],
                "undercut_logits": relation_logits[..., 2],
                "relation_probs": relation_probs,
            }
        return support, attack, undercut


class MessageExtractor(nn.Module):
    """
    Inputs:
        v_i: [batch, num_segments, hidden_dim]
        S/A/U: [batch, num_segments, num_segments]
    Outputs:
        m_sup/m_att/m_und: [batch, num_segments, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_sup = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_att = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_und = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        v_i: torch.Tensor,
        support: torch.Tensor,
        attack: torch.Tensor,
        undercut: torch.Tensor,
        segmask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        incoming_support = torch.bmm(support.transpose(1, 2), self.w_sup(v_i))
        incoming_attack = torch.bmm(attack.transpose(1, 2), self.w_att(v_i))
        incoming_undercut = torch.bmm(undercut.transpose(1, 2), self.w_und(v_i))
        mask = segmask.float().unsqueeze(-1)
        return (
            incoming_support * mask,
            incoming_attack * mask,
            incoming_undercut * mask,
        )


class RoleAwareGatingFusion(nn.Module):
    """
    Inputs:
        v_i: [batch, num_segments, hidden_dim]
        p_i: [batch, num_segments, 3]
        g:   [batch, hidden_dim]
        m_*: [batch, num_segments, hidden_dim]
    Outputs:
        m_i: [batch, num_segments, hidden_dim]
        beta:[batch, num_segments, 3]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        gate_temperature: float = 1.0,
        undercut_gate_balance: bool = False,
        undercut_gate_scale: float = 0.0,
        relation_aware_gate: bool = False,
        attack_gate_boost: float = 0.0,
        undercut_gate_boost: float = 0.0,
    ):
        super().__init__()
        self.gate_temperature = float(max(1e-6, gate_temperature))
        self.undercut_gate_balance = bool(undercut_gate_balance)
        self.undercut_gate_scale = float(max(0.0, undercut_gate_scale))
        self.relation_aware_gate = bool(relation_aware_gate)
        self.attack_gate_boost = float(max(0.0, attack_gate_boost))
        self.undercut_gate_boost = float(max(0.0, undercut_gate_boost))
        self.gate = _build_mlp(hidden_dim * 2 + 3, hidden_dim, 3, dropout)
        self.agg = _build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)

    def _apply_undercut_balance(
        self,
        beta: torch.Tensor,
        m_und: torch.Tensor,
    ) -> torch.Tensor:
        if not self.undercut_gate_balance or self.undercut_gate_scale <= 0.0:
            return beta
        undercut_strength = m_und.norm(dim=-1, keepdim=True).detach()
        centered = undercut_strength - undercut_strength.mean(dim=1, keepdim=True)
        undercut_boost = self.undercut_gate_scale * torch.tanh(centered + undercut_strength)
        logits = torch.log(beta.clamp(min=1e-8))
        logits = logits.clone()
        logits[..., 2:3] = logits[..., 2:3] + undercut_boost
        return torch.softmax(logits, dim=-1)

    def _apply_relation_aware_gate_logits(
        self,
        logits: torch.Tensor,
        m_sup: torch.Tensor,
        m_att: torch.Tensor,
        m_und: torch.Tensor,
    ) -> torch.Tensor:
        if not self.relation_aware_gate:
            return logits
        if self.attack_gate_boost <= 0.0 and self.undercut_gate_boost <= 0.0:
            return logits
        message_norms = torch.cat(
            [
                m_sup.norm(dim=-1, keepdim=True),
                m_att.norm(dim=-1, keepdim=True),
                m_und.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        ).detach()
        centered = message_norms - message_norms.mean(dim=-1, keepdim=True)
        boosts = torch.zeros_like(logits)
        if self.attack_gate_boost > 0.0:
            boosts[..., 1:2] = self.attack_gate_boost * torch.tanh(centered[..., 1:2])
        if self.undercut_gate_boost > 0.0:
            boosts[..., 2:3] = self.undercut_gate_boost * torch.tanh(centered[..., 2:3])
        return logits + boosts

    def forward(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        m_sup: torch.Tensor,
        m_att: torch.Tensor,
        m_und: torch.Tensor,
        segmask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_expand = _expand_global(g, v_i.size(1))
        gate_input = torch.cat([v_i, p_i, g_expand], dim=-1)
        gate_logits = self.gate(gate_input)
        gate_logits = self._apply_relation_aware_gate_logits(gate_logits, m_sup, m_att, m_und)
        beta = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
        beta = self._apply_undercut_balance(beta, m_und)
        weighted_messages = torch.cat(
            [
                beta[..., 0:1] * m_sup,
                beta[..., 1:2] * m_att,
                beta[..., 2:3] * m_und,
            ],
            dim=-1,
        )
        total_message = self.agg(weighted_messages)
        total_message = total_message * segmask.float().unsqueeze(-1)
        return total_message, beta


class NodeUpdater(nn.Module):
    """
    Inputs:
        prev_state: [batch, num_segments, hidden_dim]
        message:    [batch, num_segments, hidden_dim]
    Output:
        next_state: [batch, num_segments, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        residual_keep: float = 0.1,
        anchor_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_keep = float(max(0.0, min(1.0, residual_keep)))
        self.anchor_residual_scale = float(max(0.0, anchor_residual_scale))
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        prev_state: torch.Tensor,
        message: torch.Tensor,
        segmask: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = prev_state.shape
        updated = self.gru_cell(
            message.reshape(batch_size * num_nodes, hidden_dim),
            prev_state.reshape(batch_size * num_nodes, hidden_dim),
        ).reshape(batch_size, num_nodes, hidden_dim)
        updated = self.norm(updated)
        # Keep a small sample-specific residual after the GRU update. This does
        # not change the GRU-based node-update mechanism; it prevents the
        # deterministic SIMS path from collapsing all nodes to the same fixed point.
        if self.residual_keep > 0.0:
            updated = (1.0 - self.residual_keep) * updated + self.residual_keep * prev_state
            updated = self.norm(updated)
        # If the recurrent state has already begun to collapse, retaining a small
        # residual from the initial argument-conditioned node state prevents a
        # common attractor without removing GRU reasoning.
        if self.anchor_residual_scale > 0.0 and anchor_state is not None:
            updated = updated + self.anchor_residual_scale * anchor_state
            updated = self.norm(updated)
        mask = segmask.unsqueeze(-1)
        return torch.where(mask, updated, prev_state)


class RoleUpdater(nn.Module):
    """
    Inputs:
        v_i:   [batch, num_segments, hidden_dim]
        g:     [batch, hidden_dim]
        p_old: [batch, num_segments, 3]
    Output:
        p_new: [batch, num_segments, 3]
    """

    def __init__(self, hidden_dim: int, dropout: float, lambda_role: float):
        super().__init__()
        self.lambda_role = lambda_role
        self.mlp = _build_mlp(hidden_dim * 2, hidden_dim, 3, dropout)

    def forward(
        self,
        v_i: torch.Tensor,
        g: torch.Tensor,
        p_old: torch.Tensor,
        segmask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_expand = _expand_global(g, v_i.size(1))
        p_hat = torch.softmax(self.mlp(torch.cat([v_i, g_expand], dim=-1)), dim=-1)
        p_new = (1.0 - self.lambda_role) * p_old + self.lambda_role * p_hat
        mask = segmask.float().unsqueeze(-1)
        p_new = p_new * mask + p_old * (1.0 - mask)
        p_new = p_new / p_new.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return p_new, p_hat


class AcceptabilityUpdater(nn.Module):
    """
    Inputs:
        v_i:   [batch, num_segments, hidden_dim]
        p_i:   [batch, num_segments, 3]
        g:     [batch, hidden_dim]
        q_old: [batch, num_segments, 1]
    Output:
        q_new: [batch, num_segments, 1]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        lambda_q: float,
        temperature: float = 1.0,
        state_floor: float = 0.0,
        q_logit_centering: bool = False,
        q_logit_centering_mode: str = "sample",
        q_logit_centering_momentum: float = 0.0,
        q_logit_norm: bool = False,
        q_logit_scale: float = 1.0,
        q_relation_aware: bool = False,
        q_relation_aware_v2: bool = False,
        q_relation_input: str = "message_gate_relation",
        q_relation_feature_scale: float = 0.0,
        q_preserve_delta: bool = False,
        q_preserve_delta_scale: float = 0.0,
        q_delta_temperature: float = 1.0,
        q_min_entropy_guard: float = 0.0,
        q_delta_bound: bool = False,
        q_delta_bound_max: float = 0.0,
    ):
        super().__init__()
        self.lambda_q = lambda_q
        self.temperature = float(max(1e-6, temperature))
        self.state_floor = float(max(0.0, min(1.0, state_floor)))
        self.q_logit_centering = bool(q_logit_centering)
        self.q_logit_centering_mode = str(q_logit_centering_mode).lower()
        self.q_logit_centering_momentum = float(max(0.0, min(1.0, q_logit_centering_momentum)))
        self.q_logit_norm = bool(q_logit_norm)
        self.q_logit_scale = float(q_logit_scale)
        self.q_relation_aware = bool(q_relation_aware)
        self.q_relation_aware_v2 = bool(q_relation_aware_v2)
        self.q_relation_input = str(q_relation_input).lower()
        self.q_relation_feature_scale = float(max(0.0, q_relation_feature_scale))
        self.q_preserve_delta = bool(q_preserve_delta)
        self.q_preserve_delta_scale = float(max(0.0, q_preserve_delta_scale))
        self.q_delta_temperature = float(max(1e-6, q_delta_temperature))
        self.q_min_entropy_guard = float(max(0.0, min(1.0, q_min_entropy_guard)))
        # ROUND 6 MECHANISM SWITCH (default OFF, identical to baseline when disabled).
        # q_delta_bound caps the per-iteration absolute change of q to
        # q_delta_bound_max, AFTER the existing lambda_q smoothing. This does NOT
        # remove or replace the q dynamic update (ARG-MSA core innovation #7) --
        # it only prevents the documented q-collapse failure mode where a single
        # iteration drives the whole q distribution toward zero or toward 1 and
        # erases sample-level acceptability information used by step 23
        # (h_local aggregation) and step 20 (global refresh). When
        # q_delta_bound is False OR q_delta_bound_max <= 0, the clamp is skipped
        # and code path is bit-identical to the original.
        self.q_delta_bound = bool(q_delta_bound)
        self.q_delta_bound_max = float(max(0.0, q_delta_bound_max))
        self.base_input_dim = hidden_dim * 4 + 3
        self.relation_feature_dim = 13 if self.q_relation_aware_v2 else 7
        mlp_input_dim = self.base_input_dim + (
            self.relation_feature_dim
            if (self.q_relation_aware or self.q_relation_aware_v2)
            and self.q_relation_feature_scale > 0.0
            else 0
        )
        self.mlp = _build_mlp(mlp_input_dim, hidden_dim, 1, dropout)

    def _apply_state_floor(self, q: torch.Tensor, segmask: torch.Tensor) -> torch.Tensor:
        if self.state_floor <= 0.0:
            return q * segmask.float().unsqueeze(-1)
        q = _apply_q_floor(q, self.state_floor)
        return q * segmask.float().unsqueeze(-1)

    def _prepare_q_logits(
        self, q_logits: torch.Tensor, segmask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.q_logit_centering and not self.q_logit_norm:
            return q_logits
        if segmask is None:
            valid = torch.ones_like(q_logits, dtype=torch.bool)
        else:
            valid = segmask.bool().unsqueeze(-1)
        denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (q_logits * valid.float()).sum(dim=1, keepdim=True) / denom
        centered = q_logits - mean
        if self.q_logit_norm:
            variance = ((centered * valid.float()) ** 2).sum(dim=1, keepdim=True) / denom
            std = torch.sqrt(variance.clamp(min=1e-12))
            prepared = self.q_logit_scale * centered / std.clamp(min=1e-6)
        else:
            prepared = centered
        return torch.where(valid, prepared, q_logits)

    def _relation_features_for_input(
        self,
        relation_features: Optional[torch.Tensor],
        v_i: torch.Tensor,
    ) -> torch.Tensor:
        if relation_features is None:
            return torch.zeros(
                v_i.size(0),
                v_i.size(1),
                self.relation_feature_dim,
                device=v_i.device,
                dtype=v_i.dtype,
            )
        relation_features = relation_features.to(device=v_i.device, dtype=v_i.dtype)
        feature_dim = relation_features.size(-1)
        if feature_dim == self.relation_feature_dim:
            return relation_features
        if feature_dim > self.relation_feature_dim:
            return relation_features[..., : self.relation_feature_dim]
        pad = torch.zeros(
            *relation_features.shape[:-1],
            self.relation_feature_dim - feature_dim,
            device=v_i.device,
            dtype=v_i.dtype,
        )
        return torch.cat([relation_features, pad], dim=-1)

    def _apply_q_delta_preservation(
        self,
        q_raw: torch.Tensor,
        q_logits_prepared: torch.Tensor,
        segmask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.q_preserve_delta or self.q_preserve_delta_scale <= 0.0:
            return q_raw
        if segmask is None:
            valid = torch.ones_like(q_raw, dtype=torch.bool)
        else:
            valid = segmask.bool().unsqueeze(-1)
        denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (q_logits_prepared * valid.float()).sum(dim=1, keepdim=True) / denom
        centered = (q_logits_prepared - mean) / self.q_delta_temperature
        delta = torch.tanh(centered) * self.q_preserve_delta_scale
        repaired = q_raw + delta
        repaired = repaired.clamp(min=1e-6, max=1.0 - 1e-6)
        if self.q_min_entropy_guard > 0.0 and repaired.size(1) > 1:
            probs = repaired.clamp(min=0.0) * valid.float()
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1, keepdim=True)
            entropy = entropy / torch.log(torch.as_tensor(float(repaired.size(1)), device=repaired.device))
            blend = ((self.q_min_entropy_guard - entropy) / self.q_min_entropy_guard).clamp(
                min=0.0,
                max=1.0,
            )
            repaired = repaired * (1.0 - blend) + q_raw * blend
        return torch.where(valid, repaired, q_raw)

    def candidate(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        segmask: Optional[torch.Tensor] = None,
        relation_features: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        g_expand = _expand_global(g, v_i.size(1))
        q_base_input = torch.cat(
            [v_i, p_i, g_expand, v_i * g_expand, torch.abs(v_i - g_expand)], dim=-1
        )
        q_input = q_base_input
        relation_features_for_debug = self._relation_features_for_input(relation_features, v_i)
        if (
            (self.q_relation_aware or self.q_relation_aware_v2)
            and self.q_relation_feature_scale > 0.0
        ):
            q_input = torch.cat(
                [q_input, relation_features_for_debug * self.q_relation_feature_scale],
                dim=-1,
            )
        q_logits = self.mlp(q_input)
        q_logits_prepared = self._prepare_q_logits(q_logits, segmask)
        q_raw = torch.sigmoid(q_logits_prepared / self.temperature)
        q_raw = self._apply_q_delta_preservation(q_raw, q_logits_prepared, segmask)
        if return_debug:
            return q_raw, {
                "q_base_input": q_base_input,
                "q_input": q_input,
                "q_logits": q_logits,
                "q_logits_prepared": q_logits_prepared,
                "q_relation_features": relation_features_for_debug,
            }
        return q_raw

    def initialize(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        segmask: torch.Tensor,
        relation_features: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        if return_debug:
            q_init, debug = self.candidate(
                v_i,
                p_i,
                g,
                segmask=segmask,
                relation_features=relation_features,
                return_debug=True,
            )
            debug["q_before_state_floor"] = q_init
            return self._apply_state_floor(q_init, segmask), debug
        q_init = self.candidate(v_i, p_i, g, segmask=segmask, relation_features=relation_features)
        return self._apply_state_floor(q_init, segmask)

    def forward(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        g: torch.Tensor,
        q_old: torch.Tensor,
        segmask: torch.Tensor,
        relation_features: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        # q_i is updated by smoothing the new acceptability candidate with the
        # previous iteration state.
        if return_debug:
            q_hat, debug = self.candidate(
                v_i,
                p_i,
                g,
                segmask=segmask,
                relation_features=relation_features,
                return_debug=True,
            )
        else:
            q_hat = self.candidate(
                v_i,
                p_i,
                g,
                segmask=segmask,
                relation_features=relation_features,
            )
        q_mixed = (1.0 - self.lambda_q) * q_old + self.lambda_q * q_hat
        # ROUND 6 SWITCH: bound the per-iteration q delta. Default OFF.
        if self.q_delta_bound and self.q_delta_bound_max > 0.0:
            delta = q_mixed - q_old
            delta = delta.clamp(min=-self.q_delta_bound_max, max=self.q_delta_bound_max)
            q_mixed = q_old + delta
        q_new = self._apply_state_floor(q_mixed, segmask)
        mask = segmask.float().unsqueeze(-1)
        q_new = q_new * mask + q_old * (1.0 - mask)
        if return_debug:
            debug["q_before_floor"] = q_mixed * mask + q_old * (1.0 - mask)
            debug["q_before_state_floor"] = debug["q_before_floor"]
            return q_new, q_hat, debug
        return q_new, q_hat


class GlobalStateUpdater(nn.Module):
    """
    Inputs:
        g_prev: [batch, hidden_dim]
        v_i:    [batch, num_segments, hidden_dim]
        q_i:    [batch, num_segments, 1]
        segmask:[batch, num_segments]
    Output:
        g_next: [batch, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        q_floor: float = 0.05,
        q_floor_mode: str = "absolute",
        residual_keep: float = 0.1,
        pooled_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.q_floor = float(max(0.0, q_floor))
        self.q_floor_mode = str(q_floor_mode).lower()
        self.residual_keep = float(max(0.0, min(1.0, residual_keep)))
        self.pooled_residual_scale = float(max(0.0, pooled_residual_scale))
        self.mlp = _build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        g_prev: torch.Tensor,
        v_i: torch.Tensor,
        q_i: torch.Tensor,
        segmask: torch.Tensor,
        pooled_anchor: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        pooled_nodes = _masked_mean(v_i, segmask)
        # A tiny acceptability floor avoids the degenerate case where sigmoid q
        # values become near-zero for every segment and erase sample-level
        # information in the weighted global refresh branch.
        q_effective = _apply_q_floor(q_i, self.q_floor, mode=self.q_floor_mode, mask=segmask)
        pooled_weighted_nodes = _masked_weighted_mean(v_i, q_effective, segmask)
        global_input = torch.cat([g_prev, pooled_nodes, pooled_weighted_nodes], dim=-1)
        g_next = self.mlp(global_input)
        if self.pooled_residual_scale > 0.0 and pooled_anchor is not None:
            g_next = g_next + self.pooled_residual_scale * pooled_anchor
        if self.residual_keep > 0.0:
            g_next = (1.0 - self.residual_keep) * g_next + self.residual_keep * g_prev
        g_next = self.norm(g_next)
        if return_debug:
            return g_next, {
                "global_update_input": global_input,
                "pooled_nodes": pooled_nodes,
                "pooled_weighted_nodes": pooled_weighted_nodes,
                "global_before": g_prev,
                "global_after": g_next,
            }
        return g_next


class FinalPredictor(nn.Module):
    """
    Inputs:
        v_i: [batch, num_segments, hidden_dim]
        q_i: [batch, num_segments, 1]
        g:   [batch, hidden_dim]
    Outputs:
        y_hat:   [batch, output_dim]
        h_final: [batch, hidden_dim]
        h_local: [batch, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        q_floor: float = 0.05,
        q_floor_mode: str = "absolute",
        local_residual_scale: float = 0.1,
        final_residual_scale: float = 0.0,
        final_output_residual_scale: float = 0.0,
        negative_output_residual_scale: float = 0.0,
        negative_residual_threshold: float = 0.0,
        pos_output_residual_scale: float = 0.0,
        neg_output_residual_scale: float = 0.0,
        negative_residual_warmup_epochs: int = 0,
        negative_residual_max_abs: float = 0.0,
        final_q_normalize: bool = False,
        sign_aware_final: bool = False,
        sign_negative_scale: float = 0.0,
        sign_positive_scale: float = 0.0,
        final_amplitude_gate: bool = False,
        final_amplitude_scale: float = 0.0,
        final_amplitude_temperature: float = 1.0,
        final_amplitude_warmup_epochs: int = 0,
        final_amplitude_warmup_start: float = 0.0,
        negative_evidence_gate: bool = False,
        negative_evidence_scale: float = 0.0,
        negative_evidence_threshold: float = 0.0,
        sign_aware_final_v2: bool = False,
        negative_evidence_projection: bool = False,
        positive_evidence_scale: float = 0.0,
        negative_gate_temperature: float = 1.0,
        positive_amplitude_recovery: bool = False,
        positive_amplitude_scale: float = 0.0,
        positive_threshold: float = 1.0,
        positive_amplitude_temperature: float = 1.0,
        negative_amplitude_recovery: bool = False,
        negative_amplitude_scale: float = 0.0,
        negative_threshold: float = -1.0,
        negative_amplitude_temperature: float = 1.0,
        attack_aware_negative_final: bool = False,
        attack_evidence_scale: float = 0.0,
        undercut_evidence_scale: float = 0.0,
        negative_final_temperature: float = 1.0,
        amplitude_preserving_fusion: bool = False,
        local_amplitude_residual_scale: float = 0.0,
        global_amplitude_residual_scale: float = 0.0,
        final_fusion_retention_repair: bool = False,
        local_retention_residual_scale: float = 0.0,
        global_retention_residual_scale: float = 0.0,
        retention_gate_temperature: float = 1.0,
        sign_preserve_gate: bool = False,
        local_global_balance_gate: bool = False,
        local_preserve_scale: float = 0.0,
        global_preserve_scale: float = 0.0,
        positive_final_retention: bool = False,
        negative_output_amplitude_scale: float = 0.0,
        positive_output_amplitude_scale: float = 0.0,
        retention_output_residual_scale: float = 0.0,
        positive_retention_output_scale: float = 0.0,
        negative_retention_output_scale: float = 0.0,
        retention_output_residual: bool = False,
        retention_output_mode: str = "signed",
        negative_retention_scale: float = 0.0,
        positive_retention_scale: float = 0.0,
        negative_confidence_threshold: float = 0.6,
        positive_confidence_threshold: float = 0.6,
        positive_residual_max_abs: float = 0.0,
        positive_residual_warmup_epochs: int = 0,
        negative_evidence_consistency_residual: bool = False,
        negative_evidence_consistency_residual_v2: bool = False,
        negative_evidence_consistency_residual_v3: bool = False,
        negative_evidence_consistency_residual_v4: bool = False,
        negative_evidence_residual_v6: bool = False,
        negative_v6_threshold: float = 0.50,
        negative_v6_scale: float = 0.0,
        negative_v6_max_abs: float = 0.0,
        negative_v6_warmup_epochs: int = 0,
        negative_v6_trace: bool = False,
        negative_evidence_residual_v7: bool = False,
        negative_v7_threshold: float = 0.55,
        negative_v7_scale: float = 0.0,
        negative_v7_max_abs: float = 0.0,
        negative_v7_warmup_epochs: int = 0,
        negative_v7_trace: bool = False,
        negative_residual_path_gain: bool = False,
        negative_path_gain_mode: str = "gate_confidence",
        negative_path_gain_scale: float = 1.5,
        negative_path_gain_max: float = 2.5,
        negative_path_gain_min_gate: float = 0.70,
        negative_path_gain_require_neutral_block: bool = True,
        negative_path_gain_require_weak_block: bool = True,
        negative_path_gain_trace: bool = False,
        negative_evidence_consistency_scale: float = 0.0,
        negative_neutral_suppress_threshold: float = 0.4,
        negative_consistency_warmup_epochs: int = 0,
        positive_evidence_consistency_residual: bool = False,
        positive_evidence_consistency_residual_v2: bool = False,
        positive_evidence_consistency_residual_v3: bool = False,
        positive_evidence_consistency_residual_v4: bool = False,
        positive_evidence_consistency_scale: float = 0.0,
        positive_evidence_threshold: float = 0.65,
        positive_neutral_suppress_threshold: float = 0.4,
        positive_consistency_warmup_epochs: int = 0,
        final_dispersion_preserve: bool = False,
        final_dispersion_target_std: float = 0.0,
        final_dispersion_max_expand: float = 0.0,
        final_dispersion_decay_epochs: int = 0,
        # ROUND 7d boundary-safe add-on (default OFF). When all four params
        # are at their defaults (False / 0.0 / 1.0 / 1.0) the code path is
        # bit-identical to r7b. See `_apply_final_dispersion_preserve` for
        # the per-sample down-weighting and the optional dispersed/original
        # residual mix.
        final_dispersion_boundary_safe: bool = False,
        final_dispersion_boundary_margin: float = 0.0,
        final_dispersion_boundary_scale: float = 1.0,
        final_dispersion_residual_mix: float = 1.0,
    ):
        super().__init__()
        self.q_floor = float(max(0.0, q_floor))
        self.q_floor_mode = str(q_floor_mode).lower()
        self.local_residual_scale = float(max(0.0, local_residual_scale))
        self.final_residual_scale = float(max(0.0, final_residual_scale))
        self.final_output_residual_scale = float(max(0.0, final_output_residual_scale))
        self.negative_output_residual_scale = float(max(0.0, negative_output_residual_scale))
        self.negative_residual_threshold = float(negative_residual_threshold)
        self.pos_output_residual_scale = float(max(0.0, pos_output_residual_scale))
        self.neg_output_residual_scale = float(max(0.0, neg_output_residual_scale))
        self.negative_residual_warmup_epochs = int(max(0, negative_residual_warmup_epochs))
        self.negative_residual_max_abs = float(max(0.0, negative_residual_max_abs))
        self.final_q_normalize = bool(final_q_normalize)
        self.sign_aware_final = bool(sign_aware_final)
        self.sign_negative_scale = float(max(0.0, sign_negative_scale))
        self.sign_positive_scale = float(max(0.0, sign_positive_scale))
        self.final_amplitude_gate = bool(final_amplitude_gate)
        self.final_amplitude_scale = float(max(0.0, final_amplitude_scale))
        self.final_amplitude_temperature = float(max(1e-6, final_amplitude_temperature))
        self.final_amplitude_warmup_epochs = int(max(0, final_amplitude_warmup_epochs))
        self.final_amplitude_warmup_start = float(
            max(0.0, min(1.0, final_amplitude_warmup_start))
        )
        self.negative_evidence_gate = bool(negative_evidence_gate)
        self.negative_evidence_scale = float(max(0.0, negative_evidence_scale))
        self.negative_evidence_threshold = float(negative_evidence_threshold)
        self.sign_aware_final_v2 = bool(sign_aware_final_v2)
        self.negative_evidence_projection = bool(negative_evidence_projection)
        self.positive_evidence_scale = float(max(0.0, positive_evidence_scale))
        self.negative_gate_temperature = float(max(1e-6, negative_gate_temperature))
        self.positive_amplitude_recovery = bool(positive_amplitude_recovery)
        self.positive_amplitude_scale = float(max(0.0, positive_amplitude_scale))
        self.positive_threshold = float(positive_threshold)
        self.positive_amplitude_temperature = float(max(1e-6, positive_amplitude_temperature))
        self.negative_amplitude_recovery = bool(negative_amplitude_recovery)
        self.negative_amplitude_scale = float(max(0.0, negative_amplitude_scale))
        self.negative_threshold = float(negative_threshold)
        self.negative_amplitude_temperature = float(max(1e-6, negative_amplitude_temperature))
        self.attack_aware_negative_final = bool(attack_aware_negative_final)
        self.attack_evidence_scale = float(max(0.0, attack_evidence_scale))
        self.undercut_evidence_scale = float(max(0.0, undercut_evidence_scale))
        self.negative_final_temperature = float(max(1e-6, negative_final_temperature))
        self.amplitude_preserving_fusion = bool(amplitude_preserving_fusion)
        self.local_amplitude_residual_scale = float(max(0.0, local_amplitude_residual_scale))
        self.global_amplitude_residual_scale = float(max(0.0, global_amplitude_residual_scale))
        self.final_fusion_retention_repair = bool(final_fusion_retention_repair)
        self.local_retention_residual_scale = float(max(0.0, local_retention_residual_scale))
        self.global_retention_residual_scale = float(max(0.0, global_retention_residual_scale))
        self.retention_gate_temperature = float(max(1e-6, retention_gate_temperature))
        self.sign_preserve_gate = bool(sign_preserve_gate)
        self.local_global_balance_gate = bool(local_global_balance_gate)
        self.local_preserve_scale = float(max(0.0, local_preserve_scale))
        self.global_preserve_scale = float(max(0.0, global_preserve_scale))
        self.positive_final_retention = bool(positive_final_retention)
        self.negative_output_amplitude_scale = float(max(0.0, negative_output_amplitude_scale))
        self.positive_output_amplitude_scale = float(max(0.0, positive_output_amplitude_scale))
        self.retention_output_residual_scale = float(max(0.0, retention_output_residual_scale))
        self.positive_retention_output_scale = float(max(0.0, positive_retention_output_scale))
        self.negative_retention_output_scale = float(max(0.0, negative_retention_output_scale))
        self.retention_output_residual = bool(retention_output_residual)
        self.retention_output_mode = str(retention_output_mode).lower()
        self.negative_retention_scale = float(max(0.0, negative_retention_scale))
        self.positive_retention_scale = float(max(0.0, positive_retention_scale))
        self.negative_confidence_threshold = float(negative_confidence_threshold)
        self.positive_confidence_threshold = float(positive_confidence_threshold)
        self.positive_residual_max_abs = float(max(0.0, positive_residual_max_abs))
        self.positive_residual_warmup_epochs = int(max(0, positive_residual_warmup_epochs))
        self.negative_evidence_consistency_residual = bool(negative_evidence_consistency_residual)
        self.negative_evidence_consistency_residual_v2 = bool(
            negative_evidence_consistency_residual_v2
        )
        self.negative_evidence_consistency_residual_v3 = bool(
            negative_evidence_consistency_residual_v3
        )
        self.negative_evidence_consistency_residual_v4 = bool(
            negative_evidence_consistency_residual_v4
        )
        self.negative_evidence_residual_v6 = bool(negative_evidence_residual_v6)
        self.negative_v6_threshold = float(negative_v6_threshold)
        self.negative_v6_scale = float(max(0.0, negative_v6_scale))
        self.negative_v6_max_abs = float(max(0.0, negative_v6_max_abs))
        self.negative_v6_warmup_epochs = int(max(0, negative_v6_warmup_epochs))
        self.negative_v6_trace = bool(negative_v6_trace)
        self.negative_evidence_residual_v7 = bool(negative_evidence_residual_v7)
        self.negative_v7_threshold = float(negative_v7_threshold)
        self.negative_v7_scale = float(max(0.0, negative_v7_scale))
        self.negative_v7_max_abs = float(max(0.0, negative_v7_max_abs))
        self.negative_v7_warmup_epochs = int(max(0, negative_v7_warmup_epochs))
        self.negative_v7_trace = bool(negative_v7_trace)
        self.negative_residual_path_gain = bool(negative_residual_path_gain)
        self.negative_path_gain_mode = str(negative_path_gain_mode).lower()
        self.negative_path_gain_scale = float(max(0.0, negative_path_gain_scale))
        self.negative_path_gain_max = float(max(1.0, negative_path_gain_max))
        self.negative_path_gain_min_gate = float(
            max(0.0, min(1.0, negative_path_gain_min_gate))
        )
        self.negative_path_gain_require_neutral_block = bool(
            negative_path_gain_require_neutral_block
        )
        self.negative_path_gain_require_weak_block = bool(negative_path_gain_require_weak_block)
        self.negative_path_gain_trace = bool(negative_path_gain_trace)
        self.negative_evidence_consistency_scale = float(
            max(0.0, negative_evidence_consistency_scale)
        )
        self.negative_neutral_suppress_threshold = float(
            max(0.0, negative_neutral_suppress_threshold)
        )
        self.negative_consistency_warmup_epochs = int(max(0, negative_consistency_warmup_epochs))
        self.positive_evidence_consistency_residual = bool(positive_evidence_consistency_residual)
        self.positive_evidence_consistency_residual_v2 = bool(
            positive_evidence_consistency_residual_v2
        )
        self.positive_evidence_consistency_residual_v3 = bool(
            positive_evidence_consistency_residual_v3
        )
        self.positive_evidence_consistency_residual_v4 = bool(
            positive_evidence_consistency_residual_v4
        )
        self.positive_evidence_consistency_scale = float(
            max(0.0, positive_evidence_consistency_scale)
        )
        self.positive_evidence_threshold = float(positive_evidence_threshold)
        self.positive_neutral_suppress_threshold = float(
            max(0.0, positive_neutral_suppress_threshold)
        )
        self.positive_consistency_warmup_epochs = int(max(0, positive_consistency_warmup_epochs))
        # ROUND 6 MECHANISM SWITCH (default OFF, identical to baseline when disabled).
        # final_dispersion_preserve rescales y_hat at the very END of the
        # predictor by a sign-preserving multiplicative ratio that restores per-
        # batch dispersion when the predicted std has collapsed below
        # final_dispersion_target_std. It does NOT touch the ARG-MSA core
        # innovation #11 (h_local + g fusion) -- it only repairs the documented
        # final-amplitude collapse failure mode in which strong-sentiment
        # samples are squeezed toward 0 by the predictor MLP. When the switch is
        # False OR final_dispersion_target_std <= 0 OR final_dispersion_max_expand <= 0
        # the method is a no-op and code path is bit-identical to the original.
        # This is DIFFERENT from final_amplitude_gate (which is a learned
        # per-sample sigmoid bump on y_hat) -- this switch operates on the
        # whole batch's std without any new learnable parameter.
        self.final_dispersion_preserve = bool(final_dispersion_preserve)
        self.final_dispersion_target_std = float(max(0.0, final_dispersion_target_std))
        self.final_dispersion_max_expand = float(max(0.0, final_dispersion_max_expand))
        # ROUND 7b add-on (default OFF). When final_dispersion_decay_epochs > 0,
        # the effective max_expand decays LINEARLY from full value at epoch 1
        # down to 0 at epoch == final_dispersion_decay_epochs, fading dispersion
        # out as the model matures and validation predictions become tight.
        # This addresses the round7 e9-strong / e10-collapse failure mode where
        # a 5% multiplicative kick at late epochs flips boundary samples and
        # spikes Val_Loss. When the value is 0 (default), the behaviour is
        # bit-identical to round 7: dispersion stays at full strength every
        # epoch.
        self.final_dispersion_decay_epochs = int(max(0, final_dispersion_decay_epochs))
        # ROUND 7d boundary-safe add-on (default OFF; identical to r7b when
        # left at defaults). final_dispersion_boundary_safe gates the
        # per-sample down-weighting branch; final_dispersion_boundary_margin
        # defines the |y_hat| region considered "near the A_2 boundary";
        # final_dispersion_boundary_scale is the per-sample kick fraction at
        # |y_hat|=0 (1.0 -> full kick everywhere -> no-op);
        # final_dispersion_residual_mix mixes the dispersed and original
        # tensors (1.0 -> all dispersed -> no-op).
        self.final_dispersion_boundary_safe = bool(final_dispersion_boundary_safe)
        self.final_dispersion_boundary_margin = float(max(0.0, final_dispersion_boundary_margin))
        self.final_dispersion_boundary_scale = float(
            max(0.0, min(1.0, final_dispersion_boundary_scale))
        )
        self.final_dispersion_residual_mix = float(
            max(0.0, min(1.0, final_dispersion_residual_mix))
        )
        self.current_epoch = 0
        self.pred = _build_mlp(hidden_dim * 4, hidden_dim, hidden_dim, dropout)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.input_residual = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.output_residual = nn.Linear(hidden_dim * 4, output_dim, bias=False)
        self.amplitude_gate = nn.Linear(hidden_dim * 4, output_dim)
        self.negative_evidence = nn.Linear(hidden_dim * 4, output_dim)
        self.positive_evidence = nn.Linear(hidden_dim * 4, output_dim)
        self.local_preserve_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_preserve_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.local_global_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.local_retention_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_retention_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.positive_local_retention = nn.Linear(hidden_dim, output_dim, bias=False)

    def set_training_epoch(self, epoch: int) -> None:
        self.current_epoch = int(max(0, epoch))

    def _effective_final_amplitude_scale(self) -> float:
        if self.final_amplitude_scale <= 0.0:
            return 0.0
        if self.final_amplitude_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.final_amplitude_scale
        progress = min(
            1.0,
            max(
                0.0,
                (self.current_epoch - 1) / float(max(1, self.final_amplitude_warmup_epochs - 1)),
            ),
        )
        factor = self.final_amplitude_warmup_start + (
            1.0 - self.final_amplitude_warmup_start
        ) * progress
        return self.final_amplitude_scale * factor

    def _effective_negative_residual_scale(self) -> float:
        if self.neg_output_residual_scale <= 0.0:
            return 0.0
        if self.negative_residual_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.neg_output_residual_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.negative_residual_warmup_epochs))),
        )
        return self.neg_output_residual_scale * progress

    def _effective_confidence_negative_scale(self) -> float:
        if self.negative_retention_scale <= 0.0:
            return 0.0
        if self.negative_residual_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.negative_retention_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.negative_residual_warmup_epochs))),
        )
        return self.negative_retention_scale * progress

    def _effective_confidence_positive_scale(self) -> float:
        if self.positive_retention_scale <= 0.0:
            return 0.0
        if self.positive_residual_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.positive_retention_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.positive_residual_warmup_epochs))),
        )
        return self.positive_retention_scale * progress

    def _effective_negative_consistency_scale(self) -> float:
        if self.negative_evidence_consistency_scale <= 0.0:
            return 0.0
        if self.negative_consistency_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.negative_evidence_consistency_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.negative_consistency_warmup_epochs))),
        )
        return self.negative_evidence_consistency_scale * progress

    def _effective_negative_v6_scale(self) -> float:
        if not self.negative_evidence_residual_v6 or self.negative_v6_scale <= 0.0:
            return 0.0
        if self.negative_v6_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.negative_v6_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.negative_v6_warmup_epochs))),
        )
        return self.negative_v6_scale * progress

    def _effective_negative_v7_scale(self) -> float:
        if not self.negative_evidence_residual_v7 or self.negative_v7_scale <= 0.0:
            return 0.0
        if self.negative_v7_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.negative_v7_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.negative_v7_warmup_epochs))),
        )
        return self.negative_v7_scale * progress

    def _effective_negative_path_gain_scale(self) -> float:
        if not self.negative_residual_path_gain or self.negative_path_gain_scale <= 0.0:
            return 0.0
        return self.negative_path_gain_scale

    def _effective_positive_consistency_scale(self) -> float:
        if self.positive_evidence_consistency_scale <= 0.0:
            return 0.0
        if self.positive_consistency_warmup_epochs <= 1 or self.current_epoch <= 0:
            return self.positive_evidence_consistency_scale
        progress = min(
            1.0,
            max(0.0, self.current_epoch / float(max(1, self.positive_consistency_warmup_epochs))),
        )
        return self.positive_evidence_consistency_scale * progress

    def _bounded_negative_residual(self, y_residual: torch.Tensor) -> torch.Tensor:
        if self.negative_residual_max_abs <= 0.0:
            return y_residual
        return y_residual.clamp(
            min=-self.negative_residual_max_abs,
            max=self.negative_residual_max_abs,
        )

    def _bounded_positive_residual(self, y_residual: torch.Tensor) -> torch.Tensor:
        if self.positive_residual_max_abs <= 0.0:
            return y_residual
        return y_residual.clamp(
            min=-self.positive_residual_max_abs,
            max=self.positive_residual_max_abs,
        )

    def _bounded_negative_v6_residual(self, y_residual: torch.Tensor) -> torch.Tensor:
        if self.negative_v6_max_abs <= 0.0:
            return y_residual
        return y_residual.clamp(
            min=-self.negative_v6_max_abs,
            max=self.negative_v6_max_abs,
        )

    def _bounded_negative_v7_residual(self, y_residual: torch.Tensor) -> torch.Tensor:
        if self.negative_v7_max_abs <= 0.0:
            return y_residual
        return y_residual.clamp(
            min=-self.negative_v7_max_abs,
            max=self.negative_v7_max_abs,
        )

    def _combine_output_residual(
        self,
        y_base: torch.Tensor,
        y_residual: torch.Tensor,
    ) -> torch.Tensor:
        signed_override = self.pos_output_residual_scale > 0.0 or self.neg_output_residual_scale > 0.0
        if not signed_override:
            y_hat = y_base
            if self.final_output_residual_scale > 0.0:
                y_hat = y_hat + self.final_output_residual_scale * y_residual
        else:
            scale = torch.full_like(y_residual, self.final_output_residual_scale)
            residual = y_residual
            if self.pos_output_residual_scale > 0.0:
                scale = torch.where(
                    y_residual >= 0.0,
                    torch.full_like(y_residual, self.pos_output_residual_scale),
                    scale,
                )
            if self.neg_output_residual_scale > 0.0:
                residual = torch.where(
                    y_residual < 0.0,
                    self._bounded_negative_residual(y_residual),
                    y_residual,
                )
                scale = torch.where(
                    y_residual < 0.0,
                    torch.full_like(y_residual, self._effective_negative_residual_scale()),
                    scale,
                )
            y_hat = y_base + scale * residual
        if self.negative_output_residual_scale > 0.0:
            negative_mask = (y_residual < self.negative_residual_threshold).to(y_residual.dtype)
            residual = torch.where(
                negative_mask.bool(),
                self._bounded_negative_residual(y_residual),
                y_residual,
            )
            y_hat = y_hat + self.negative_output_residual_scale * residual * negative_mask
        return y_hat

    def _apply_sign_aware_final(self, y_hat: torch.Tensor) -> torch.Tensor:
        if not self.sign_aware_final:
            return y_hat
        if self.sign_negative_scale <= 0.0 and self.sign_positive_scale <= 0.0:
            return y_hat
        scale = torch.zeros_like(y_hat)
        if self.sign_negative_scale > 0.0:
            scale = torch.where(
                y_hat < 0.0,
                torch.full_like(y_hat, self.sign_negative_scale),
                scale,
            )
        if self.sign_positive_scale > 0.0:
            scale = torch.where(
                y_hat > 0.0,
                torch.full_like(y_hat, self.sign_positive_scale),
                scale,
            )
        return y_hat * (1.0 + scale)

    def _apply_negative_evidence(
        self,
        y_hat: torch.Tensor,
        pred_input: torch.Tensor,
    ) -> torch.Tensor:
        if not self.negative_evidence_gate or self.negative_evidence_scale <= 0.0:
            return y_hat
        negative_mask = (y_hat < self.negative_evidence_threshold).to(y_hat.dtype)
        evidence = F.softplus(self.negative_evidence(pred_input))
        return y_hat - self.negative_evidence_scale * evidence * negative_mask

    def _apply_sign_aware_final_v2(
        self,
        y_hat: torch.Tensor,
        pred_input: torch.Tensor,
    ) -> torch.Tensor:
        if not self.sign_aware_final_v2 or not self.negative_evidence_projection:
            return y_hat
        if self.negative_evidence_scale <= 0.0 and self.positive_evidence_scale <= 0.0:
            return y_hat

        corrected = y_hat
        if self.negative_evidence_scale > 0.0:
            negative_mask = (y_hat < 0.0).to(y_hat.dtype)
            negative_gate = torch.sigmoid(-y_hat / self.negative_gate_temperature) * negative_mask
            negative_evidence = F.softplus(self.negative_evidence(pred_input))
            corrected = corrected - self.negative_evidence_scale * negative_evidence * negative_gate
        if self.positive_evidence_scale > 0.0:
            positive_mask = (y_hat > 0.0).to(y_hat.dtype)
            positive_gate = torch.sigmoid(y_hat / self.negative_gate_temperature) * positive_mask
            positive_evidence = F.softplus(self.positive_evidence(pred_input))
            corrected = corrected + self.positive_evidence_scale * positive_evidence * positive_gate
        return corrected

    def _apply_positive_amplitude_recovery(self, y_hat: torch.Tensor) -> torch.Tensor:
        if not self.positive_amplitude_recovery or self.positive_amplitude_scale <= 0.0:
            return y_hat
        positive_base = y_hat.clamp(min=0.0)
        if not torch.is_tensor(positive_base):
            return y_hat
        gate = torch.sigmoid(
            (y_hat - self.positive_threshold) / self.positive_amplitude_temperature
        )
        return y_hat + self.positive_amplitude_scale * positive_base * gate

    def _apply_negative_amplitude_recovery(self, y_hat: torch.Tensor) -> torch.Tensor:
        if not self.negative_amplitude_recovery or self.negative_amplitude_scale <= 0.0:
            return y_hat
        negative_base = y_hat.clamp(max=0.0)
        gate = torch.sigmoid(
            (self.negative_threshold - y_hat) / self.negative_amplitude_temperature
        )
        return y_hat + self.negative_amplitude_scale * negative_base * gate

    def _apply_attack_aware_negative_final(
        self,
        y_hat: torch.Tensor,
        attack_evidence: Optional[torch.Tensor] = None,
        undercut_evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.attack_aware_negative_final:
            return y_hat
        if self.attack_evidence_scale <= 0.0 and self.undercut_evidence_scale <= 0.0:
            return y_hat
        if attack_evidence is None:
            attack_evidence = torch.zeros_like(y_hat)
        if undercut_evidence is None:
            undercut_evidence = torch.zeros_like(y_hat)
        evidence = (
            self.attack_evidence_scale * attack_evidence.to(y_hat)
            + self.undercut_evidence_scale * undercut_evidence.to(y_hat)
        )
        negative_gate = torch.sigmoid(-y_hat / self.negative_final_temperature)
        return y_hat - negative_gate * evidence

    def _apply_amplitude_preserving_fusion(
        self,
        h_final: torch.Tensor,
        h_local: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        if not self.amplitude_preserving_fusion:
            return h_final
        if self.local_amplitude_residual_scale <= 0.0 and self.global_amplitude_residual_scale <= 0.0:
            return h_final
        local_residual = self.local_amplitude_residual_scale * h_local
        global_residual = self.global_amplitude_residual_scale * g
        residual = local_residual + global_residual
        if self.sign_preserve_gate:
            gate = torch.sigmoid((h_local * g).mean(dim=-1, keepdim=True))
            residual = residual * gate
        return h_final + residual

    def _apply_local_global_balance_gate(
        self,
        h_final: torch.Tensor,
        h_local: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        if not self.local_global_balance_gate:
            return h_final
        if self.local_preserve_scale <= 0.0 and self.global_preserve_scale <= 0.0:
            return h_final
        gate = torch.sigmoid(self.local_global_gate(torch.cat([h_local, g], dim=-1)))
        local_residual = self.local_preserve_scale * self.local_preserve_proj(h_local)
        global_residual = self.global_preserve_scale * self.global_preserve_proj(g)
        return h_final + gate * (local_residual + global_residual)

    def _effective_final_dispersion_max_expand(self) -> float:
        # ROUND 7b decay helper. When final_dispersion_decay_epochs == 0
        # (default), returns the configured max_expand untouched -> behaviour
        # identical to round 7. When > 0, linearly fades the max_expand from
        # full value at epoch 1 down to 0 at epoch == final_dispersion_decay_epochs.
        # set_training_epoch is called by trains/ARG_MSA.py per epoch, so
        # self.current_epoch is always up to date during training and validation.
        if self.final_dispersion_max_expand <= 0.0:
            return 0.0
        if self.final_dispersion_decay_epochs <= 0 or self.current_epoch <= 0:
            return self.final_dispersion_max_expand
        progress = min(
            1.0,
            max(
                0.0,
                (self.current_epoch - 1)
                / float(max(1, self.final_dispersion_decay_epochs - 1)),
            ),
        )
        factor = 1.0 - progress
        if factor <= 0.0:
            return 0.0
        return self.final_dispersion_max_expand * factor

    def _apply_final_dispersion_preserve(self, y_hat: torch.Tensor) -> torch.Tensor:
        # Default OFF. When enabled (final_dispersion_preserve=True and BOTH
        # target_std > 0 and max_expand > 0), scale the whole-batch prediction
        # by a sign-preserving multiplicative ratio so that when the batch's
        # std drops well below the configured target the dispersion gets
        # bounded-expanded back. The ratio is clamped to [1.0, 1 + max_expand]
        # so this can ONLY expand (never compress) and can NEVER flip a sign.
        # Uses no new learnable parameter, no batch-normalization statistics,
        # and no train/eval-mode dependency: identical at training and inference.
        if not self.final_dispersion_preserve:
            return y_hat
        if self.final_dispersion_target_std <= 0.0 or self.final_dispersion_max_expand <= 0.0:
            return y_hat
        # ROUND 7b: pull through the decayed max_expand so a finished decay
        # window cleanly skips dispersion in late epochs.
        effective_max_expand = self._effective_final_dispersion_max_expand()
        if effective_max_expand <= 0.0:
            return y_hat
        # Compute batch-wide std along the batch dimension (dim=0) so it is well
        # defined for both [B] and [B, output_dim] shapes.
        if y_hat.dim() == 0:
            return y_hat
        current_std = y_hat.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
        target_std = torch.full_like(current_std, float(self.final_dispersion_target_std))
        ratio = (target_std / current_std).clamp(
            min=1.0,
            max=1.0 + float(effective_max_expand),
        )
        # ROUND 7d boundary-safe branch (default OFF). When boundary_safe is
        # True AND boundary_margin > 0 AND boundary_scale < 1.0, attenuate the
        # per-sample multiplicative kick for samples whose |y_hat| is small
        # (i.e. close to the A_2 decision boundary at 0). The attenuation
        # ramps linearly from `boundary_scale` at |y_hat|=0 up to 1.0 (full
        # kick) at |y_hat| >= boundary_margin. This preserves the strong-
        # sentiment expansion (which drives the Acc / MAE gains) while
        # protecting near-zero samples from being pushed across the boundary
        # (which drives the F1 loss). The detach on |y_hat| keeps the weight
        # itself constant w.r.t. gradient flow, matching how `ratio` already
        # behaves as a fixed multiplicative scale.
        if (
            self.final_dispersion_boundary_safe
            and self.final_dispersion_boundary_margin > 0.0
            and self.final_dispersion_boundary_scale < 1.0
        ):
            amp = y_hat.detach().abs()
            ramp = (amp / float(self.final_dispersion_boundary_margin)).clamp(max=1.0)
            weight = float(self.final_dispersion_boundary_scale) + (
                1.0 - float(self.final_dispersion_boundary_scale)
            ) * ramp
            per_sample_ratio = 1.0 + (ratio - 1.0) * weight
            y_disp = y_hat * per_sample_ratio
        else:
            y_disp = y_hat * ratio
        # ROUND 7d residual_mix (default 1.0 -> y_disp -> no-op). When set to
        # a value in [0.0, 1.0), blend the dispersed output with the
        # original: y_new = mix * y_disp + (1 - mix) * y_hat. Used as an
        # orthogonal safety lever; r7d phaseA leaves it at default 1.0 and
        # only exercises boundary_safe.
        mix = float(self.final_dispersion_residual_mix)
        if 0.0 <= mix < 1.0:
            return mix * y_disp + (1.0 - mix) * y_hat
        return y_disp

    def _apply_final_fusion_retention_repair(
        self,
        h_final: torch.Tensor,
        h_local: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        if not self.final_fusion_retention_repair:
            return h_final
        if self.local_retention_residual_scale <= 0.0 and self.global_retention_residual_scale <= 0.0:
            return h_final
        residual = (
            self.local_retention_residual_scale * self.local_retention_proj(h_local)
            + self.global_retention_residual_scale * self.global_retention_proj(g)
        )
        if self.sign_preserve_gate:
            gate = torch.sigmoid(
                (h_local * g).mean(dim=-1, keepdim=True) / self.retention_gate_temperature
            )
            residual = residual * gate
        return h_final + residual

    def _apply_output_amplitude_recovery(self, y_hat: torch.Tensor) -> torch.Tensor:
        corrected = y_hat
        if self.negative_output_amplitude_scale > 0.0:
            negative_gate = torch.sigmoid(
                (self.negative_threshold - y_hat) / self.negative_amplitude_temperature
            )
            corrected = corrected + (
                self.negative_output_amplitude_scale * y_hat.clamp(max=0.0) * negative_gate
            )
        if self.positive_output_amplitude_scale > 0.0:
            positive_gate = torch.sigmoid(
                (y_hat - self.positive_threshold) / self.positive_amplitude_temperature
            )
            corrected = corrected + (
                self.positive_output_amplitude_scale * y_hat.clamp(min=0.0) * positive_gate
            )
        return corrected

    def _apply_positive_final_retention(
        self,
        y_hat: torch.Tensor,
        h_local: torch.Tensor,
    ) -> torch.Tensor:
        if not self.positive_final_retention or self.positive_amplitude_scale <= 0.0:
            return y_hat
        positive_gate = torch.sigmoid(
            (y_hat - self.positive_threshold) / self.positive_amplitude_temperature
        )
        positive_mask = (y_hat > 0.0).to(y_hat.dtype)
        local_evidence = F.softplus(self.positive_local_retention(h_local))
        return y_hat + self.positive_amplitude_scale * local_evidence * positive_gate * positive_mask

    def _apply_retention_output_residual(
        self,
        y_hat: torch.Tensor,
        y_residual: torch.Tensor,
        negative_confidence: Optional[torch.Tensor] = None,
        positive_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.zeros_like(y_hat)
        self._last_negative_v7_residual_delta = torch.zeros_like(y_hat)
        self._last_negative_v7_residual_gate = torch.zeros_like(y_hat)
        self._last_negative_path_gain_gate = torch.zeros_like(y_hat)
        self._last_negative_path_gain_value = torch.ones_like(y_hat)
        self._last_negative_path_gain_before_delta = torch.zeros_like(y_hat)
        self._last_negative_path_gain_after_delta = torch.zeros_like(y_hat)
        self._last_negative_path_gain_extra_delta = torch.zeros_like(y_hat)
        self._last_negative_path_gain_final_total_delta = torch.zeros_like(y_hat)
        negative_v6_scale = self._effective_negative_v6_scale()
        negative_v7_scale = self._effective_negative_v7_scale()
        path_gain_scale = self._effective_negative_path_gain_scale()
        if (
            negative_v6_scale > 0.0
            or negative_v7_scale > 0.0
            or path_gain_scale > 0.0
            or self.negative_evidence_consistency_residual
            or self.positive_evidence_consistency_residual
        ):
            delta = torch.zeros_like(y_hat)
            negative_v7_gate = torch.zeros_like(y_hat)
            if negative_v7_scale > 0.0:
                if negative_confidence is None:
                    negative_confidence = torch.zeros_like(y_hat)
                negative_residual = self._bounded_negative_v7_residual(y_residual).clamp(max=0.0)
                negative_v7_gate = (
                    (negative_confidence.to(y_hat) >= self.negative_v7_threshold)
                    & (y_residual < 0.0)
                ).to(y_hat.dtype)
                self._last_negative_v7_residual_gate = negative_v7_gate
                if negative_v7_scale > 0.0:
                    negative_delta = negative_v7_scale * negative_residual * negative_v7_gate
                    before_gain_delta = negative_delta
                    if path_gain_scale > 0.0:
                        adapter_confidence = negative_confidence.to(y_hat).clamp(0.0, 1.0)
                        min_gate = self.negative_path_gain_min_gate
                        gain_gate = negative_v7_gate * (
                            adapter_confidence >= min_gate
                        ).to(y_hat.dtype)
                        if self.negative_path_gain_mode == "constant":
                            gain_value = torch.full_like(
                                adapter_confidence,
                                min(self.negative_path_gain_max, 1.0 + path_gain_scale),
                            )
                        else:
                            denom = max(1e-6, 1.0 - min_gate)
                            confidence_excess = (
                                (adapter_confidence - min_gate) / denom
                            ).clamp(0.0, 1.0)
                            gain_value = (
                                1.0 + path_gain_scale * confidence_excess
                            ).clamp(max=self.negative_path_gain_max)
                        after_gain_delta = torch.where(
                            gain_gate.bool(),
                            negative_delta * gain_value,
                            negative_delta,
                        )
                        self._last_negative_path_gain_gate = gain_gate
                        self._last_negative_path_gain_value = torch.where(
                            gain_gate.bool(),
                            gain_value,
                            torch.ones_like(gain_value),
                        )
                        self._last_negative_path_gain_before_delta = before_gain_delta
                        self._last_negative_path_gain_after_delta = after_gain_delta
                        self._last_negative_path_gain_extra_delta = (
                            after_gain_delta - before_gain_delta
                        )
                        negative_delta = after_gain_delta
                    self._last_negative_v7_residual_delta = before_gain_delta
                    delta = delta + negative_delta
                gate = torch.maximum(gate, negative_v7_gate)
            if negative_v6_scale > 0.0:
                if negative_confidence is None:
                    negative_confidence = torch.zeros_like(y_hat)
                negative_residual = self._bounded_negative_v6_residual(y_residual).clamp(max=0.0)
                negative_gate = (
                    (negative_confidence.to(y_hat) >= self.negative_v6_threshold)
                    & (y_residual < 0.0)
                ).to(y_hat.dtype)
                negative_delta = negative_v6_scale * negative_residual * negative_gate
                delta = delta + negative_delta
                gate = torch.maximum(gate, negative_gate)
            if self.negative_evidence_consistency_residual:
                scale = self._effective_negative_consistency_scale()
                if scale > 0.0:
                    if negative_confidence is None:
                        negative_confidence = torch.zeros_like(y_hat)
                    negative_residual = self._bounded_negative_residual(y_residual).clamp(max=0.0)
                    if self.negative_evidence_consistency_residual_v4:
                        residual_direction_conf = torch.sigmoid(
                            (negative_residual.abs() - 0.05) / 0.10
                        )
                        negative_gate_confidence = (
                            0.55 * negative_confidence.to(y_hat)
                            + 0.45 * residual_direction_conf
                        ).clamp(0.0, 1.0)
                        negative_gate = (
                            (negative_gate_confidence >= self.negative_evidence_threshold)
                            & (y_residual < 0.0)
                        ).to(y_hat.dtype)
                    elif self.negative_evidence_consistency_residual_v3:
                        residual_direction_conf = torch.sigmoid(
                            (negative_residual.abs() - 0.05) / 0.10
                        )
                        negative_gate_confidence = (
                            0.30 * negative_confidence.to(y_hat)
                            + 0.70 * residual_direction_conf
                        ).clamp(0.0, 1.0)
                        negative_gate = (
                            (negative_gate_confidence >= self.negative_evidence_threshold)
                            & (y_residual < 0.0)
                        ).to(y_hat.dtype)
                    elif self.negative_evidence_consistency_residual_v2:
                        residual_direction_conf = torch.sigmoid(
                            (negative_residual.abs() - 0.05) / 0.10
                        )
                        negative_confidence = torch.maximum(
                            negative_confidence.to(y_hat),
                            residual_direction_conf,
                        )
                        negative_gate = (
                            (negative_confidence >= self.negative_evidence_threshold)
                            & (y_residual < 0.0)
                            & (y_hat < 0.0)
                        ).to(y_hat.dtype)
                    else:
                        negative_gate = (
                            (negative_confidence.to(y_hat) >= self.negative_evidence_threshold)
                            & (y_hat < -self.negative_neutral_suppress_threshold)
                            & (y_residual < 0.0)
                        ).to(y_hat.dtype)
                    negative_delta = scale * negative_residual * negative_gate
                    delta = delta + negative_delta
                    gate = torch.maximum(gate, negative_gate)
            if self.positive_evidence_consistency_residual:
                scale = self._effective_positive_consistency_scale()
                if scale > 0.0:
                    if positive_confidence is None:
                        positive_confidence = torch.zeros_like(y_hat)
                    positive_residual = self._bounded_positive_residual(y_residual).clamp(min=0.0)
                    if self.positive_evidence_consistency_residual_v4:
                        residual_direction_conf = torch.sigmoid(
                            (positive_residual.abs() - 0.05) / 0.10
                        )
                        positive_gate_confidence = (
                            0.55 * positive_confidence.to(y_hat)
                            + 0.45 * residual_direction_conf
                        ).clamp(0.0, 1.0)
                        positive_gate = (
                            (positive_gate_confidence >= self.positive_evidence_threshold)
                            & (y_residual > 0.0)
                        ).to(y_hat.dtype)
                    elif self.positive_evidence_consistency_residual_v3:
                        residual_direction_conf = torch.sigmoid(
                            (positive_residual.abs() - 0.05) / 0.10
                        )
                        positive_gate_confidence = (
                            0.30 * positive_confidence.to(y_hat)
                            + 0.70 * residual_direction_conf
                        ).clamp(0.0, 1.0)
                        positive_gate = (
                            (positive_gate_confidence >= self.positive_evidence_threshold)
                            & (y_residual > 0.0)
                        ).to(y_hat.dtype)
                    elif self.positive_evidence_consistency_residual_v2:
                        residual_direction_conf = torch.sigmoid(
                            (positive_residual.abs() - 0.05) / 0.10
                        )
                        positive_confidence = torch.maximum(
                            positive_confidence.to(y_hat),
                            residual_direction_conf,
                        )
                        positive_gate = (
                            (positive_confidence >= self.positive_evidence_threshold)
                            & (y_residual > 0.0)
                            & (y_hat > 0.0)
                        ).to(y_hat.dtype)
                    else:
                        positive_gate = (
                            (positive_confidence.to(y_hat) >= self.positive_evidence_threshold)
                            & (y_hat > self.positive_neutral_suppress_threshold)
                            & (y_residual > 0.0)
                        ).to(y_hat.dtype)
                    positive_delta = scale * positive_residual * positive_gate
                    delta = delta + positive_delta
                    gate = torch.maximum(gate, positive_gate)
            self._last_negative_path_gain_final_total_delta = delta
            return y_hat + delta, delta, gate
        if self.retention_output_residual:
            mode = self.retention_output_mode
            if mode == "negative_confidence":
                scale = self._effective_confidence_negative_scale()
                if scale <= 0.0:
                    return y_hat, torch.zeros_like(y_hat), gate
                if negative_confidence is None:
                    negative_confidence = torch.zeros_like(y_hat)
                gate = (
                    (negative_confidence.to(y_hat) >= self.negative_confidence_threshold)
                    & (y_hat < 0.0)
                    & (y_residual < 0.0)
                ).to(y_hat.dtype)
                delta = scale * self._bounded_negative_residual(y_residual).clamp(max=0.0) * gate
                return y_hat + delta, delta, gate
            if mode == "positive_confidence":
                scale = self._effective_confidence_positive_scale()
                if scale <= 0.0:
                    return y_hat, torch.zeros_like(y_hat), gate
                if positive_confidence is None:
                    positive_confidence = torch.zeros_like(y_hat)
                gate = (
                    (positive_confidence.to(y_hat) >= self.positive_confidence_threshold)
                    & (y_hat > 0.0)
                    & (y_residual > 0.0)
                ).to(y_hat.dtype)
                delta = scale * self._bounded_positive_residual(y_residual).clamp(min=0.0) * gate
                return y_hat + delta, delta, gate
        if (
            self.retention_output_residual_scale <= 0.0
            and self.positive_retention_output_scale <= 0.0
            and self.negative_retention_output_scale <= 0.0
        ):
            return y_hat, torch.zeros_like(y_hat), gate
        delta = torch.zeros_like(y_hat)
        if self.retention_output_residual_scale > 0.0:
            delta = delta + self.retention_output_residual_scale * y_residual
        if self.positive_retention_output_scale > 0.0:
            positive_residual = y_residual.clamp(min=0.0)
            delta = delta + self.positive_retention_output_scale * positive_residual
        if self.negative_retention_output_scale > 0.0:
            negative_residual = self._bounded_negative_residual(y_residual).clamp(max=0.0)
            delta = delta + self.negative_retention_output_scale * negative_residual
        gate = (delta.abs() > 0.0).to(y_hat.dtype)
        return y_hat + delta, delta, gate

    def forward(
        self,
        v_i: torch.Tensor,
        q_i: torch.Tensor,
        g: torch.Tensor,
        segmask: torch.Tensor,
        attack_final_evidence: Optional[torch.Tensor] = None,
        undercut_final_evidence: Optional[torch.Tensor] = None,
        negative_retention_confidence: Optional[torch.Tensor] = None,
        positive_retention_confidence: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Final local aggregation follows the acceptability-weighted node sum.
        # q_effective is only a numerical safeguard for the all-near-zero q case;
        # q itself is still dynamically estimated and logged unchanged.
        mask = segmask.float().unsqueeze(-1)
        q_effective = _apply_q_floor(q_i, self.q_floor, mode=self.q_floor_mode, mask=segmask) * mask
        h_local_weighted = (v_i * q_effective).sum(dim=1)
        if self.final_q_normalize:
            h_local_weighted = h_local_weighted / q_effective.sum(dim=1).clamp(min=1e-6)
        h_local_residual = _masked_mean(v_i, segmask)
        h_local = h_local_weighted + self.local_residual_scale * h_local_residual
        pred_input = torch.cat([h_local, g, h_local * g, torch.abs(h_local - g)], dim=-1)
        trace = {"final_predictor_input": pred_input}
        hidden = pred_input
        for layer_idx, layer in enumerate(self.pred):
            hidden = layer(hidden)
            trace[f"final_predictor_layer_{layer_idx}_{layer.__class__.__name__}"] = hidden
        h_mlp = hidden
        h_residual = self.input_residual(pred_input)
        h_final = h_mlp
        if self.final_residual_scale > 0.0:
            h_final = h_final + self.final_residual_scale * h_residual
        h_final = self._apply_amplitude_preserving_fusion(h_final, h_local, g)
        h_final = self._apply_local_global_balance_gate(h_final, h_local, g)
        h_final = self._apply_final_fusion_retention_repair(h_final, h_local, g)
        y_base = self.out(h_final)
        y_residual = self.output_residual(pred_input)
        y_hat = self._combine_output_residual(y_base, y_residual)
        amp_gate = None
        effective_amplitude_scale = self._effective_final_amplitude_scale()
        if self.final_amplitude_gate and effective_amplitude_scale > 0.0:
            amp_gate = torch.sigmoid(self.amplitude_gate(pred_input) / self.final_amplitude_temperature)
            y_hat = y_hat * (1.0 + effective_amplitude_scale * amp_gate)
        y_hat = self._apply_sign_aware_final(y_hat)
        y_hat = self._apply_sign_aware_final_v2(y_hat, pred_input)
        y_hat = self._apply_output_amplitude_recovery(y_hat)
        y_before_positive_final_retention = y_hat
        y_hat = self._apply_positive_final_retention(y_hat, h_local)
        positive_final_retention_delta = y_hat - y_before_positive_final_retention
        y_before_positive_recovery = y_hat
        y_hat = self._apply_positive_amplitude_recovery(y_hat)
        positive_amplitude_delta = y_hat - y_before_positive_recovery
        y_before_negative_recovery = y_hat
        y_hat = self._apply_negative_amplitude_recovery(y_hat)
        negative_amplitude_delta = y_hat - y_before_negative_recovery
        y_before_retention_output = y_hat
        y_hat, retention_output_delta, retention_output_gate = self._apply_retention_output_residual(
            y_hat,
            y_residual,
            negative_confidence=negative_retention_confidence,
            positive_confidence=positive_retention_confidence,
        )
        attack_evidence = (
            attack_final_evidence.to(y_hat)
            if attack_final_evidence is not None
            else torch.zeros_like(y_hat)
        )
        undercut_evidence = (
            undercut_final_evidence.to(y_hat)
            if undercut_final_evidence is not None
            else torch.zeros_like(y_hat)
        )
        y_hat = self._apply_negative_evidence(y_hat, pred_input)
        y_before_attack_aware = y_hat
        y_hat = self._apply_attack_aware_negative_final(y_hat, attack_evidence, undercut_evidence)
        attack_aware_delta = y_hat - y_before_attack_aware
        # ROUND 6 SWITCH: bounded sign-preserving dispersion repair. Default OFF.
        y_hat = self._apply_final_dispersion_preserve(y_hat)
        if return_trace:
            trace.update(
                {
                    "final_predictor_mlp_output": h_mlp,
                    "final_predictor_input_residual": h_residual,
                    "final_output_base": y_base,
                    "final_output_residual": y_residual,
                    "final_amplitude_gate": amp_gate if amp_gate is not None else torch.zeros_like(y_hat),
                    "negative_evidence": F.softplus(self.negative_evidence(pred_input))
                    if self.negative_evidence_gate and self.negative_evidence_scale > 0.0
                    else torch.zeros_like(y_hat),
                    "negative_evidence_v2": F.softplus(self.negative_evidence(pred_input))
                    if self.sign_aware_final_v2 and self.negative_evidence_projection
                    else torch.zeros_like(y_hat),
                    "positive_evidence_v2": F.softplus(self.positive_evidence(pred_input))
                    if self.sign_aware_final_v2 and self.negative_evidence_projection
                    else torch.zeros_like(y_hat),
                    "positive_amplitude_recovery": positive_amplitude_delta
                    if self.positive_amplitude_recovery and self.positive_amplitude_scale > 0.0
                    else torch.zeros_like(y_hat),
                    "positive_final_retention": positive_final_retention_delta
                    if self.positive_final_retention and self.positive_amplitude_scale > 0.0
                    else torch.zeros_like(y_hat),
                    "negative_amplitude_recovery": negative_amplitude_delta
                    if self.negative_amplitude_recovery and self.negative_amplitude_scale > 0.0
                    else torch.zeros_like(y_hat),
                    "retention_output_raw_pred": y_before_retention_output,
                    "retention_output_residual": retention_output_delta
                    if (
                        self.retention_output_residual
                        or
                        self.negative_evidence_residual_v7
                        or
                        self.negative_evidence_residual_v6
                        or
                        self.negative_evidence_consistency_residual
                        or self.positive_evidence_consistency_residual
                        or
                        self.retention_output_residual_scale > 0.0
                        or self.positive_retention_output_scale > 0.0
                        or self.negative_retention_output_scale > 0.0
                    )
                    else torch.zeros_like(y_hat),
                    "retention_output_residual_gate": retention_output_gate,
                    "negative_confidence_component_negative_residual_direction": torch.sigmoid(
                        (self._bounded_negative_residual(y_residual).clamp(max=0.0).abs() - 0.05)
                        / 0.10
                    )
                    if (
                        self.negative_evidence_consistency_residual_v2
                        or self.negative_evidence_consistency_residual_v3
                        or self.negative_evidence_consistency_residual_v4
                        or self.negative_evidence_residual_v6
                        or self.negative_evidence_residual_v7
                    )
                    else torch.zeros_like(y_hat),
                    "positive_confidence_component_positive_residual_direction": torch.sigmoid(
                        (self._bounded_positive_residual(y_residual).clamp(min=0.0).abs() - 0.05)
                        / 0.10
                    )
                    if (
                        self.positive_evidence_consistency_residual_v2
                        or self.positive_evidence_consistency_residual_v3
                        or self.positive_evidence_consistency_residual_v4
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_negative_gate_confidence_v3": (
                        0.30 * negative_retention_confidence.to(y_hat)
                        + 0.70
                        * torch.sigmoid(
                            (
                                self._bounded_negative_residual(y_residual)
                                .clamp(max=0.0)
                                .abs()
                                - 0.05
                            )
                            / 0.10
                        )
                    ).clamp(0.0, 1.0)
                    if (
                        self.negative_evidence_consistency_residual_v3
                        and negative_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_negative_gate_confidence_v4": (
                        0.55 * negative_retention_confidence.to(y_hat)
                        + 0.45
                        * torch.sigmoid(
                            (
                                self._bounded_negative_residual(y_residual)
                                .clamp(max=0.0)
                                .abs()
                                - 0.05
                            )
                            / 0.10
                        )
                    ).clamp(0.0, 1.0)
                    if (
                        self.negative_evidence_consistency_residual_v4
                        and negative_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_negative_gate_confidence_v6": negative_retention_confidence.to(y_hat)
                    if (
                        self.negative_evidence_residual_v6
                        and negative_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_negative_gate_confidence_v7": negative_retention_confidence.to(y_hat)
                    if (
                        self.negative_evidence_residual_v7
                        and negative_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_v7_delta": self._last_negative_v7_residual_delta
                    if self.negative_v7_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_v7_delta_abs": self._last_negative_v7_residual_delta.abs()
                    if self.negative_v7_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_v7_gate": self._last_negative_v7_residual_gate
                    if (
                        self.negative_v7_trace
                        or self.negative_path_gain_trace
                    )
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_gate": self._last_negative_path_gain_gate
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_value": self._last_negative_path_gain_value
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_before_delta": self._last_negative_path_gain_before_delta
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_before_delta_abs": self._last_negative_path_gain_before_delta.abs()
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_after_delta": self._last_negative_path_gain_after_delta
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_after_delta_abs": self._last_negative_path_gain_after_delta.abs()
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_extra_delta": self._last_negative_path_gain_extra_delta
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_extra_delta_abs": self._last_negative_path_gain_extra_delta.abs()
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_final_total_delta": self._last_negative_path_gain_final_total_delta
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "negative_confidence_component_path_gain_final_total_delta_abs": self._last_negative_path_gain_final_total_delta.abs()
                    if self.negative_path_gain_trace
                    else torch.zeros_like(y_hat),
                    "positive_confidence_component_positive_gate_confidence_v3": (
                        0.30 * positive_retention_confidence.to(y_hat)
                        + 0.70
                        * torch.sigmoid(
                            (
                                self._bounded_positive_residual(y_residual)
                                .clamp(min=0.0)
                                .abs()
                                - 0.05
                            )
                            / 0.10
                        )
                    ).clamp(0.0, 1.0)
                    if (
                        self.positive_evidence_consistency_residual_v3
                        and positive_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "positive_confidence_component_positive_gate_confidence_v4": (
                        0.55 * positive_retention_confidence.to(y_hat)
                        + 0.45
                        * torch.sigmoid(
                            (
                                self._bounded_positive_residual(y_residual)
                                .clamp(min=0.0)
                                .abs()
                                - 0.05
                            )
                            / 0.10
                        )
                    ).clamp(0.0, 1.0)
                    if (
                        self.positive_evidence_consistency_residual_v4
                        and positive_retention_confidence is not None
                    )
                    else torch.zeros_like(y_hat),
                    "negative_retention_confidence": negative_retention_confidence.to(y_hat)
                    if negative_retention_confidence is not None
                    else torch.zeros_like(y_hat),
                    "positive_retention_confidence": positive_retention_confidence.to(y_hat)
                    if positive_retention_confidence is not None
                    else torch.zeros_like(y_hat),
                    "attack_final_evidence": attack_evidence,
                    "undercut_final_evidence": undercut_evidence,
                    "attack_aware_negative_final": attack_aware_delta
                    if self.attack_aware_negative_final
                    else torch.zeros_like(y_hat),
                    "final_predictor_output": y_hat,
                }
            )
            return (
                y_hat,
                h_final,
                h_local,
                pred_input,
                h_local_weighted,
                h_local_residual,
                q_effective,
                trace,
            )
        return y_hat, h_final, h_local, pred_input, h_local_weighted, h_local_residual, q_effective


class IterativeReasoningBlock(nn.Module):
    """
    Inputs:
        v_i: [batch, num_segments, hidden_dim]
        p_i: [batch, num_segments, 3]
        q_i: [batch, num_segments, 1]
        g:   [batch, hidden_dim]
    Outputs:
        updated states with same leading shapes
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        lambda_role: float,
        lambda_q: float,
        q_floor: float = 0.05,
        q_floor_mode: str = "absolute",
        node_residual_keep: float = 0.1,
        global_residual_keep: float = 0.1,
        node_input_residual_scale: float = 0.1,
        global_pooled_residual_scale: float = 0.1,
        relation_evidence_residual_scale: float = 0.1,
        relation_temperature: float = 1.0,
        relation_confidence_residual: bool = False,
        relation_confidence_scale: float = 0.0,
        gate_temperature: float = 1.0,
        undercut_gate_balance: bool = False,
        undercut_gate_scale: float = 0.0,
        relation_aware_gate: bool = False,
        attack_gate_boost: float = 0.0,
        undercut_gate_boost: float = 0.0,
        q_temperature: float = 1.0,
        q_state_floor: float = 0.0,
        q_logit_centering: bool = False,
        q_logit_centering_mode: str = "sample",
        q_logit_centering_momentum: float = 0.0,
        q_logit_norm: bool = False,
        q_logit_scale: float = 1.0,
        q_relation_aware: bool = False,
        q_relation_aware_v2: bool = False,
        q_relation_input: str = "message_gate_relation",
        q_relation_feature_scale: float = 0.0,
        q_preserve_delta: bool = False,
        q_preserve_delta_scale: float = 0.0,
        q_delta_temperature: float = 1.0,
        q_min_entropy_guard: float = 0.0,
        q_delta_bound: bool = False,
        q_delta_bound_max: float = 0.0,
    ):
        super().__init__()
        self.relation_generator = RelationGenerator(
            hidden_dim,
            dropout,
            evidence_residual_scale=relation_evidence_residual_scale,
            temperature=relation_temperature,
            relation_confidence_residual=relation_confidence_residual,
            relation_confidence_scale=relation_confidence_scale,
        )
        self.message_extractor = MessageExtractor(hidden_dim)
        self.gating_fusion = RoleAwareGatingFusion(
            hidden_dim,
            dropout,
            gate_temperature=gate_temperature,
            undercut_gate_balance=undercut_gate_balance,
            undercut_gate_scale=undercut_gate_scale,
            relation_aware_gate=relation_aware_gate,
            attack_gate_boost=attack_gate_boost,
            undercut_gate_boost=undercut_gate_boost,
        )
        self.node_updater = NodeUpdater(
            hidden_dim,
            residual_keep=node_residual_keep,
            anchor_residual_scale=node_input_residual_scale,
        )
        self.role_updater = RoleUpdater(hidden_dim, dropout, lambda_role)
        self.acceptability_updater = AcceptabilityUpdater(
            hidden_dim,
            dropout,
            lambda_q,
            temperature=q_temperature,
            state_floor=q_state_floor,
            q_logit_centering=q_logit_centering,
            q_logit_centering_mode=q_logit_centering_mode,
            q_logit_centering_momentum=q_logit_centering_momentum,
            q_logit_norm=q_logit_norm,
            q_logit_scale=q_logit_scale,
            q_relation_aware=q_relation_aware,
            q_relation_aware_v2=q_relation_aware_v2,
            q_relation_input=q_relation_input,
            q_relation_feature_scale=q_relation_feature_scale,
            q_preserve_delta=q_preserve_delta,
            q_preserve_delta_scale=q_preserve_delta_scale,
            q_delta_temperature=q_delta_temperature,
            q_min_entropy_guard=q_min_entropy_guard,
            q_delta_bound=q_delta_bound,
            q_delta_bound_max=q_delta_bound_max,
        )
        self.global_updater = GlobalStateUpdater(
            hidden_dim,
            dropout,
            q_floor=q_floor,
            q_floor_mode=q_floor_mode,
            residual_keep=global_residual_keep,
            pooled_residual_scale=global_pooled_residual_scale,
        )

    def forward(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        q_i: torch.Tensor,
        g: torch.Tensor,
        segmask: torch.Tensor,
        node_anchor: Optional[torch.Tensor] = None,
        pooled_anchor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # S/A/U are regenerated at every iteration by a learnable relation network.
        support, attack, undercut, relation_debug = self.relation_generator(
            v_i, p_i, g, segmask, return_debug=True
        )
        m_sup, m_att, m_und = self.message_extractor(v_i, support, attack, undercut, segmask)
        total_message, beta = self.gating_fusion(v_i, p_i, g, m_sup, m_att, m_und, segmask)
        v_next = self.node_updater(v_i, total_message, segmask, anchor_state=node_anchor)
        relation_confidence = relation_debug.get("relation_confidence")
        if relation_confidence is None:
            relation_confidence = (support + attack + undercut).amax(dim=1)
        else:
            relation_confidence = relation_confidence.mean(dim=1)
        relation_features = torch.cat(
            [
                m_sup.norm(dim=-1, keepdim=True),
                m_att.norm(dim=-1, keepdim=True),
                m_und.norm(dim=-1, keepdim=True),
                beta,
                relation_confidence.unsqueeze(-1),
                support.mean(dim=1).unsqueeze(-1),
                attack.mean(dim=1).unsqueeze(-1),
                undercut.mean(dim=1).unsqueeze(-1),
                support.max(dim=1).values.unsqueeze(-1),
                attack.max(dim=1).values.unsqueeze(-1),
                undercut.max(dim=1).values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # This baseline uses a provisional global state from q^(t) to break the
        # circular dependency between g^(t+1), p^(t+1) and q^(t+1), then refines g.
        g_provisional, global_prov_debug = self.global_updater(
            g,
            v_next,
            q_i,
            segmask,
            pooled_anchor=pooled_anchor,
            return_debug=True,
        )
        p_next, p_hat = self.role_updater(v_next, g_provisional, p_i, segmask)
        q_next, q_hat, q_debug = self.acceptability_updater(
            v_next,
            p_next,
            g_provisional,
            q_i,
            segmask,
            relation_features=relation_features,
            return_debug=True,
        )
        g_next, global_final_debug = self.global_updater(
            g,
            v_next,
            q_next,
            segmask,
            pooled_anchor=pooled_anchor,
            return_debug=True,
        )

        debug = {
            "support": support,
            "attack": attack,
            "undercut": undercut,
            **relation_debug,
            "gate_beta": beta,
            "node_states_before": v_i,
            "support_message": m_sup,
            "attack_message": m_att,
            "undercut_message": m_und,
            "fused_message": total_message,
            "p_hat": p_hat,
            "q_hat": q_hat,
            "q_input": q_debug["q_input"],
            "q_base_input": q_debug.get("q_base_input"),
            "q_relation_features": q_debug.get("q_relation_features"),
            "q_logits": q_debug["q_logits"],
            "q_before_floor": q_debug["q_before_floor"],
            "q_after_floor": _apply_q_floor(
                q_next,
                self.global_updater.q_floor,
                mode=self.global_updater.q_floor_mode,
                mask=segmask,
            ),
            "global_update_input": global_final_debug["global_update_input"],
            "global_before": global_final_debug["global_before"],
            "global_after": global_final_debug["global_after"],
            "global_provisional_input": global_prov_debug["global_update_input"],
        }
        return v_next, p_next, q_next, g_next, debug


class ARG_MSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(_get_arg(args, "hidden_dim", 128))
        self.dropout = float(_get_arg(args, "dropout", 0.1))
        self.num_iters = int(_get_arg(args, "num_iters", 3))
        self.lambda_role = float(_get_arg(args, "lambda_role", 0.2))
        self.lambda_q = float(_get_arg(args, "lambda_q", 0.2))
        self.q_floor = float(_get_arg(args, "q_floor", 0.05))
        self.q_floor_mode = str(_get_arg(args, "q_floor_mode", "absolute")).lower()
        self.q_logit_centering = bool(_get_arg(args, "q_logit_centering", False))
        self.q_logit_centering_mode = str(_get_arg(args, "q_logit_centering_mode", "sample"))
        self.q_logit_centering_momentum = float(_get_arg(args, "q_logit_centering_momentum", 0.0))
        self.q_logit_norm = bool(_get_arg(args, "q_logit_norm", False))
        self.q_logit_scale = float(_get_arg(args, "q_logit_scale", 1.0))
        self.local_residual_scale = float(_get_arg(args, "local_residual_scale", 0.1))
        self.node_residual_keep = float(_get_arg(args, "node_residual_keep", 0.1))
        self.global_residual_keep = float(_get_arg(args, "global_residual_keep", 0.1))
        self.node_init_residual_scale = float(_get_arg(args, "node_init_residual_scale", 0.1))
        self.global_init_residual_scale = float(_get_arg(args, "global_init_residual_scale", 0.1))
        self.relation_evidence_residual_scale = float(
            _get_arg(args, "relation_evidence_residual_scale", 0.1)
        )
        self.relation_temperature = float(_get_arg(args, "relation_temperature", 1.0))
        self.relation_confidence_residual = bool(
            _get_arg(args, "relation_confidence_residual", False)
        )
        self.relation_confidence_scale = float(_get_arg(args, "relation_confidence_scale", 0.0))
        self.gate_temperature = float(_get_arg(args, "gate_temperature", 1.0))
        self.undercut_gate_balance = bool(_get_arg(args, "undercut_gate_balance", False))
        self.undercut_gate_scale = float(_get_arg(args, "undercut_gate_scale", 0.0))
        self.relation_aware_gate = bool(_get_arg(args, "relation_aware_gate", False))
        self.attack_gate_boost = float(_get_arg(args, "attack_gate_boost", 0.0))
        self.undercut_gate_boost = float(_get_arg(args, "undercut_gate_boost", 0.0))
        self.q_temperature = float(_get_arg(args, "q_temperature", 1.0))
        self.q_state_floor = float(_get_arg(args, "q_state_floor", 0.0))
        self.q_relation_aware = bool(_get_arg(args, "q_relation_aware", False))
        self.q_relation_aware_v2 = bool(_get_arg(args, "q_relation_aware_v2", False))
        self.q_relation_input = str(_get_arg(args, "q_relation_input", "message_gate_relation"))
        self.q_relation_feature_scale = float(_get_arg(args, "q_relation_feature_scale", 0.0))
        self.q_preserve_delta = bool(_get_arg(args, "q_preserve_delta", False))
        self.q_preserve_delta_scale = float(_get_arg(args, "q_preserve_delta_scale", 0.0))
        self.q_delta_temperature = float(_get_arg(args, "q_delta_temperature", 1.0))
        self.q_min_entropy_guard = float(_get_arg(args, "q_min_entropy_guard", 0.0))
        # ROUND 6 mechanism switches -- all default OFF.
        self.q_delta_bound = bool(_get_arg(args, "q_delta_bound", False))
        self.q_delta_bound_max = float(_get_arg(args, "q_delta_bound_max", 0.0))
        self.final_dispersion_preserve = bool(_get_arg(args, "final_dispersion_preserve", False))
        self.final_dispersion_target_std = float(_get_arg(args, "final_dispersion_target_std", 0.0))
        self.final_dispersion_max_expand = float(_get_arg(args, "final_dispersion_max_expand", 0.0))
        # ROUND 7b decay add-on. Default 0 -> no decay -> bit-identical to r7.
        self.final_dispersion_decay_epochs = int(_get_arg(args, "final_dispersion_decay_epochs", 0))
        # ROUND 7d boundary-safe add-on. All four defaults reproduce r7b
        # behaviour exactly: boundary_safe=False -> branch skipped;
        # boundary_margin=0.0 -> branch skipped even if accidentally enabled;
        # boundary_scale=1.0 -> per-sample weight is 1.0 everywhere -> no-op;
        # residual_mix=1.0 -> dispersed output returned unchanged.
        self.final_dispersion_boundary_safe = bool(
            _get_arg(args, "final_dispersion_boundary_safe", False)
        )
        self.final_dispersion_boundary_margin = float(
            _get_arg(args, "final_dispersion_boundary_margin", 0.0)
        )
        self.final_dispersion_boundary_scale = float(
            _get_arg(args, "final_dispersion_boundary_scale", 1.0)
        )
        self.final_dispersion_residual_mix = float(
            _get_arg(args, "final_dispersion_residual_mix", 1.0)
        )
        self.node_input_residual_scale = float(_get_arg(args, "node_input_residual_scale", 0.1))
        self.global_pooled_residual_scale = float(
            _get_arg(args, "global_pooled_residual_scale", 0.1)
        )
        self.final_residual_scale = float(_get_arg(args, "final_residual_scale", 0.0))
        self.final_output_residual_scale = float(
            _get_arg(args, "final_output_residual_scale", 0.0)
        )
        self.negative_output_residual_scale = float(
            _get_arg(args, "negative_output_residual_scale", 0.0)
        )
        self.negative_residual_threshold = float(_get_arg(args, "negative_residual_threshold", 0.0))
        self.pos_output_residual_scale = float(_get_arg(args, "pos_output_residual_scale", 0.0))
        self.neg_output_residual_scale = float(_get_arg(args, "neg_output_residual_scale", 0.0))
        self.negative_residual_warmup_epochs = int(
            _get_arg(args, "negative_residual_warmup_epochs", 0)
        )
        self.negative_residual_max_abs = float(_get_arg(args, "negative_residual_max_abs", 0.0))
        self.final_q_normalize = bool(_get_arg(args, "final_q_normalize", False))
        self.sign_aware_final = bool(_get_arg(args, "sign_aware_final", False))
        self.sign_negative_scale = float(_get_arg(args, "sign_negative_scale", 0.0))
        self.sign_positive_scale = float(_get_arg(args, "sign_positive_scale", 0.0))
        self.sign_aware_final_v2 = bool(_get_arg(args, "sign_aware_final_v2", False))
        self.negative_evidence_projection = bool(
            _get_arg(args, "negative_evidence_projection", False)
        )
        self.positive_evidence_scale = float(_get_arg(args, "positive_evidence_scale", 0.0))
        self.negative_gate_temperature = float(_get_arg(args, "negative_gate_temperature", 1.0))
        self.final_amplitude_gate = bool(_get_arg(args, "final_amplitude_gate", False))
        self.final_amplitude_scale = float(_get_arg(args, "final_amplitude_scale", 0.0))
        self.final_amplitude_temperature = float(_get_arg(args, "final_amplitude_temperature", 1.0))
        self.final_amplitude_warmup_epochs = int(
            _get_arg(args, "final_amplitude_warmup_epochs", 0)
        )
        self.final_amplitude_warmup_start = float(
            _get_arg(args, "final_amplitude_warmup_start", 0.0)
        )
        self.negative_evidence_gate = bool(_get_arg(args, "negative_evidence_gate", False))
        self.negative_evidence_scale = float(_get_arg(args, "negative_evidence_scale", 0.0))
        self.negative_evidence_threshold = float(_get_arg(args, "negative_evidence_threshold", 0.0))
        self.positive_amplitude_recovery = bool(
            _get_arg(args, "positive_amplitude_recovery", False)
        )
        self.positive_amplitude_scale = float(_get_arg(args, "positive_amplitude_scale", 0.0))
        self.positive_threshold = float(_get_arg(args, "positive_threshold", 1.0))
        self.positive_amplitude_temperature = float(
            _get_arg(args, "positive_amplitude_temperature", 1.0)
        )
        self.negative_amplitude_recovery = bool(
            _get_arg(args, "negative_amplitude_recovery", False)
        )
        self.negative_amplitude_scale = float(_get_arg(args, "negative_amplitude_scale", 0.0))
        self.negative_threshold = float(_get_arg(args, "negative_threshold", -1.0))
        self.negative_amplitude_temperature = float(
            _get_arg(args, "negative_amplitude_temperature", 1.0)
        )
        self.attack_aware_negative_final = bool(_get_arg(args, "attack_aware_negative_final", False))
        self.attack_evidence_scale = float(_get_arg(args, "attack_evidence_scale", 0.0))
        self.undercut_evidence_scale = float(_get_arg(args, "undercut_evidence_scale", 0.0))
        self.negative_final_temperature = float(_get_arg(args, "negative_final_temperature", 1.0))
        self.amplitude_preserving_fusion = bool(_get_arg(args, "amplitude_preserving_fusion", False))
        self.local_amplitude_residual_scale = float(
            _get_arg(args, "local_amplitude_residual_scale", 0.0)
        )
        self.global_amplitude_residual_scale = float(
            _get_arg(args, "global_amplitude_residual_scale", 0.0)
        )
        self.final_fusion_retention_repair = bool(
            _get_arg(args, "final_fusion_retention_repair", False)
        )
        self.local_retention_residual_scale = float(
            _get_arg(args, "local_retention_residual_scale", 0.0)
        )
        self.global_retention_residual_scale = float(
            _get_arg(args, "global_retention_residual_scale", 0.0)
        )
        self.retention_gate_temperature = float(_get_arg(args, "retention_gate_temperature", 1.0))
        self.sign_preserve_gate = bool(_get_arg(args, "sign_preserve_gate", False))
        self.local_global_balance_gate = bool(_get_arg(args, "local_global_balance_gate", False))
        self.local_preserve_scale = float(_get_arg(args, "local_preserve_scale", 0.0))
        self.global_preserve_scale = float(_get_arg(args, "global_preserve_scale", 0.0))
        self.positive_final_retention = bool(_get_arg(args, "positive_final_retention", False))
        self.negative_output_amplitude_scale = float(
            _get_arg(args, "negative_output_amplitude_scale", 0.0)
        )
        self.positive_output_amplitude_scale = float(
            _get_arg(args, "positive_output_amplitude_scale", 0.0)
        )
        self.retention_output_residual_scale = float(
            _get_arg(args, "retention_output_residual_scale", 0.0)
        )
        self.positive_retention_output_scale = float(
            _get_arg(args, "positive_retention_output_scale", 0.0)
        )
        self.negative_retention_output_scale = float(
            _get_arg(args, "negative_retention_output_scale", 0.0)
        )
        self.retention_output_residual = bool(_get_arg(args, "retention_output_residual", False))
        self.retention_output_mode = str(_get_arg(args, "retention_output_mode", "signed")).lower()
        self.negative_retention_scale = float(_get_arg(args, "negative_retention_scale", 0.0))
        self.positive_retention_scale = float(_get_arg(args, "positive_retention_scale", 0.0))
        self.negative_confidence_threshold = float(
            _get_arg(args, "negative_confidence_threshold", 0.6)
        )
        self.positive_confidence_threshold = float(
            _get_arg(args, "positive_confidence_threshold", 0.6)
        )
        self.positive_residual_max_abs = float(_get_arg(args, "positive_residual_max_abs", 0.0))
        self.positive_residual_warmup_epochs = int(
            _get_arg(args, "positive_residual_warmup_epochs", 0)
        )
        self.negative_evidence_consistency_residual = bool(
            _get_arg(args, "negative_evidence_consistency_residual", False)
        )
        self.negative_evidence_consistency_residual_v2 = bool(
            _get_arg(args, "negative_evidence_consistency_residual_v2", False)
        )
        self.negative_evidence_consistency_residual_v3 = bool(
            _get_arg(args, "negative_evidence_consistency_residual_v3", False)
        )
        self.negative_evidence_consistency_residual_v4 = bool(
            _get_arg(args, "negative_evidence_consistency_residual_v4", False)
        )
        self.negative_evidence_residual_v6 = bool(
            _get_arg(args, "negative_evidence_residual_v6", False)
        )
        self.negative_v6_use_hlocal = bool(_get_arg(args, "negative_v6_use_hlocal", True))
        self.negative_v6_use_q_node = bool(_get_arg(args, "negative_v6_use_q_node", True))
        self.negative_v6_use_residual_direction = bool(
            _get_arg(args, "negative_v6_use_residual_direction", True)
        )
        self.negative_v6_use_attack_role = bool(
            _get_arg(args, "negative_v6_use_attack_role", True)
        )
        self.negative_v6_use_positive_suppression = bool(
            _get_arg(args, "negative_v6_use_positive_suppression", True)
        )
        self.negative_v6_use_neutral_suppression = bool(
            _get_arg(args, "negative_v6_use_neutral_suppression", True)
        )
        self.negative_v6_score_norm = str(
            _get_arg(args, "negative_v6_score_norm", "batch_zscore")
        ).lower()
        self.negative_v6_threshold = float(_get_arg(args, "negative_v6_threshold", 0.50))
        self.negative_v6_scale = float(_get_arg(args, "negative_v6_scale", 0.0))
        self.negative_v6_max_abs = float(_get_arg(args, "negative_v6_max_abs", 0.0))
        self.negative_v6_warmup_epochs = int(_get_arg(args, "negative_v6_warmup_epochs", 0))
        self.negative_v6_trace = bool(_get_arg(args, "negative_v6_trace", False))
        self.negative_evidence_residual_v7 = bool(
            _get_arg(args, "negative_evidence_residual_v7", False)
        )
        self.negative_v7_use_hlocal = bool(_get_arg(args, "negative_v7_use_hlocal", True))
        self.negative_v7_use_q_node = bool(_get_arg(args, "negative_v7_use_q_node", True))
        self.negative_v7_use_residual_direction = bool(
            _get_arg(args, "negative_v7_use_residual_direction", True)
        )
        self.negative_v7_use_attack_role = bool(
            _get_arg(args, "negative_v7_use_attack_role", True)
        )
        self.negative_v7_use_positive_suppression = bool(
            _get_arg(args, "negative_v7_use_positive_suppression", True)
        )
        self.negative_v7_use_neutral_suppression = bool(
            _get_arg(args, "negative_v7_use_neutral_suppression", True)
        )
        self.negative_v7_use_weak_negative_suppression = bool(
            _get_arg(args, "negative_v7_use_weak_negative_suppression", True)
        )
        self.negative_v7_score_norm = str(
            _get_arg(args, "negative_v7_score_norm", "batch_zscore")
        ).lower()
        self.negative_v7_threshold = float(_get_arg(args, "negative_v7_threshold", 0.55))
        self.negative_v7_strong_threshold = float(
            _get_arg(args, "negative_v7_strong_threshold", 0.65)
        )
        self.negative_v7_neutral_suppress_threshold = float(
            _get_arg(args, "negative_v7_neutral_suppress_threshold", 0.45)
        )
        self.negative_v7_weak_suppress_threshold = float(
            _get_arg(args, "negative_v7_weak_suppress_threshold", 0.40)
        )
        self.negative_v7_scale = float(_get_arg(args, "negative_v7_scale", 0.0))
        self.negative_v7_max_abs = float(_get_arg(args, "negative_v7_max_abs", 0.0))
        self.negative_v7_warmup_epochs = int(_get_arg(args, "negative_v7_warmup_epochs", 0))
        self.negative_v7_trace = bool(_get_arg(args, "negative_v7_trace", False))
        self.negative_residual_path_gain = bool(
            _get_arg(args, "negative_residual_path_gain", False)
        )
        self.negative_path_gain_mode = str(
            _get_arg(args, "negative_path_gain_mode", "gate_confidence")
        ).lower()
        self.negative_path_gain_scale = float(_get_arg(args, "negative_path_gain_scale", 1.5))
        self.negative_path_gain_max = float(_get_arg(args, "negative_path_gain_max", 2.5))
        self.negative_path_gain_min_gate = float(
            _get_arg(args, "negative_path_gain_min_gate", 0.70)
        )
        self.negative_path_gain_require_neutral_block = bool(
            _get_arg(args, "negative_path_gain_require_neutral_block", True)
        )
        self.negative_path_gain_require_weak_block = bool(
            _get_arg(args, "negative_path_gain_require_weak_block", True)
        )
        self.negative_path_gain_trace = bool(_get_arg(args, "negative_path_gain_trace", False))
        self.negative_confidence_source = str(
            _get_arg(args, "negative_confidence_source", "legacy")
        ).lower()
        self.negative_confidence_use_attack = _config_bool_or_mode(
            _get_arg(args, "negative_confidence_use_attack", True)
        )
        self.negative_confidence_use_hlocal = bool(
            _get_arg(args, "negative_confidence_use_hlocal", True)
        )
        self.negative_confidence_use_q_node = bool(
            _get_arg(args, "negative_confidence_use_q_node", True)
        )
        self.negative_confidence_use_residual_direction = bool(
            _get_arg(args, "negative_confidence_use_residual_direction", False)
        )
        self.negative_confidence_use_role = _config_bool_or_mode(
            _get_arg(args, "negative_confidence_use_role", True)
        )
        self.negative_confidence_use_undercut = bool(
            _get_arg(args, "negative_confidence_use_undercut", True)
        )
        self.negative_confidence_neutral_suppress = bool(
            _get_arg(args, "negative_confidence_neutral_suppress", False)
        )
        self.negative_evidence_consistency_scale = float(
            _get_arg(args, "negative_evidence_consistency_scale", 0.0)
        )
        self.negative_neutral_suppress_threshold = float(
            _get_arg(args, "negative_neutral_suppress_threshold", 0.4)
        )
        self.negative_consistency_warmup_epochs = int(
            _get_arg(args, "negative_consistency_warmup_epochs", 0)
        )
        self.positive_evidence_consistency_residual = bool(
            _get_arg(args, "positive_evidence_consistency_residual", False)
        )
        self.positive_evidence_consistency_residual_v2 = bool(
            _get_arg(args, "positive_evidence_consistency_residual_v2", False)
        )
        self.positive_evidence_consistency_residual_v3 = bool(
            _get_arg(args, "positive_evidence_consistency_residual_v3", False)
        )
        self.positive_evidence_consistency_residual_v4 = bool(
            _get_arg(args, "positive_evidence_consistency_residual_v4", False)
        )
        self.positive_confidence_source = str(
            _get_arg(args, "positive_confidence_source", "legacy")
        ).lower()
        self.positive_confidence_use_support = bool(
            _get_arg(args, "positive_confidence_use_support", True)
        )
        self.positive_confidence_use_hlocal = bool(
            _get_arg(args, "positive_confidence_use_hlocal", True)
        )
        self.positive_confidence_use_q_node = bool(
            _get_arg(args, "positive_confidence_use_q_node", True)
        )
        self.positive_confidence_use_role = bool(
            _get_arg(args, "positive_confidence_use_role", True)
        )
        self.positive_confidence_neutral_suppress = bool(
            _get_arg(args, "positive_confidence_neutral_suppress", False)
        )
        self.positive_intensity_gate = bool(_get_arg(args, "positive_intensity_gate", False))
        self.positive_intensity_threshold = float(
            _get_arg(args, "positive_intensity_threshold", 0.55)
        )
        self.positive_weak_suppress_threshold = float(
            _get_arg(args, "positive_weak_suppress_threshold", 0.35)
        )
        self.positive_evidence_consistency_scale = float(
            _get_arg(args, "positive_evidence_consistency_scale", 0.0)
        )
        self.positive_evidence_threshold = float(_get_arg(args, "positive_evidence_threshold", 0.65))
        self.positive_neutral_suppress_threshold = float(
            _get_arg(args, "positive_neutral_suppress_threshold", 0.4)
        )
        self.positive_consistency_warmup_epochs = int(
            _get_arg(args, "positive_consistency_warmup_epochs", 0)
        )
        self.segment_window = int(_get_arg(args, "segment_window", 4))
        self.segment_pooling = str(_get_arg(args, "segment_pooling", "mean")).lower()
        self.attention_temperature = float(_get_arg(args, "attention_temperature", 1.0))
        self.use_soft_role = bool(_get_arg(args, "use_soft_role", True))
        self.num_heads = int(_get_arg(args, "num_heads", 4))
        feature_dims = _get_arg(args, "feature_dims", [768, 5, 20])
        output_dim = _get_arg(args, "num_classes", 1) if _get_arg(args, "train_mode", "regression") == "classification" else 1

        self.text_encoder = UnimodalSequenceEncoder(feature_dims[0], self.hidden_dim, self.dropout)
        self.audio_encoder = UnimodalSequenceEncoder(feature_dims[1], self.hidden_dim, self.dropout)
        self.vision_encoder = UnimodalSequenceEncoder(feature_dims[2], self.hidden_dim, self.dropout)
        self.local_fusion = LocalMultimodalFusion(
            self.hidden_dim, self.dropout, self.num_heads
        )
        self.global_init = _build_mlp(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.segment_builder = SegmentBuilder(
            self.segment_window,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            pooling=self.segment_pooling,
            attention_temperature=self.attention_temperature,
        )
        self.role_classifier = RoleClassifier(self.hidden_dim, self.dropout)
        self.role_set_builder = RoleSetBuilder(self.use_soft_role)
        self.node_initializer = NodeInitializer(
            self.hidden_dim,
            self.dropout,
            residual_scale=self.node_init_residual_scale,
        )
        self.acceptability_initializer = AcceptabilityUpdater(
            self.hidden_dim,
            self.dropout,
            self.lambda_q,
            temperature=self.q_temperature,
            state_floor=self.q_state_floor,
            q_logit_centering=self.q_logit_centering,
            q_logit_centering_mode=self.q_logit_centering_mode,
            q_logit_centering_momentum=self.q_logit_centering_momentum,
            q_logit_norm=self.q_logit_norm,
            q_logit_scale=self.q_logit_scale,
            q_relation_aware=False,
            q_relation_aware_v2=False,
            q_relation_input=self.q_relation_input,
            q_relation_feature_scale=0.0,
            q_preserve_delta=self.q_preserve_delta,
            q_preserve_delta_scale=self.q_preserve_delta_scale,
            q_delta_temperature=self.q_delta_temperature,
            q_min_entropy_guard=self.q_min_entropy_guard,
        )
        self.reasoning_blocks = nn.ModuleList(
            [
                IterativeReasoningBlock(
                    self.hidden_dim,
                    self.dropout,
                    self.lambda_role,
                    self.lambda_q,
                    q_floor=self.q_floor,
                    q_floor_mode=self.q_floor_mode,
                    node_residual_keep=self.node_residual_keep,
                    global_residual_keep=self.global_residual_keep,
                    node_input_residual_scale=self.node_input_residual_scale,
                    global_pooled_residual_scale=self.global_pooled_residual_scale,
                    relation_evidence_residual_scale=self.relation_evidence_residual_scale,
                    relation_temperature=self.relation_temperature,
                    relation_confidence_residual=self.relation_confidence_residual,
                    relation_confidence_scale=self.relation_confidence_scale,
                    gate_temperature=self.gate_temperature,
                    undercut_gate_balance=self.undercut_gate_balance,
                    undercut_gate_scale=self.undercut_gate_scale,
                    relation_aware_gate=self.relation_aware_gate,
                    attack_gate_boost=self.attack_gate_boost,
                    undercut_gate_boost=self.undercut_gate_boost,
                    q_temperature=self.q_temperature,
                    q_state_floor=self.q_state_floor,
                    q_logit_centering=self.q_logit_centering,
                    q_logit_centering_mode=self.q_logit_centering_mode,
                    q_logit_centering_momentum=self.q_logit_centering_momentum,
                    q_logit_norm=self.q_logit_norm,
                    q_logit_scale=self.q_logit_scale,
                    q_relation_aware=self.q_relation_aware,
                    q_relation_aware_v2=self.q_relation_aware_v2,
                    q_relation_input=self.q_relation_input,
                    q_relation_feature_scale=self.q_relation_feature_scale,
                    q_preserve_delta=self.q_preserve_delta,
                    q_preserve_delta_scale=self.q_preserve_delta_scale,
                    q_delta_temperature=self.q_delta_temperature,
                    q_min_entropy_guard=self.q_min_entropy_guard,
                    q_delta_bound=self.q_delta_bound,
                    q_delta_bound_max=self.q_delta_bound_max,
                )
                for _ in range(self.num_iters)
            ]
        )
        self.final_predictor = FinalPredictor(
            self.hidden_dim,
            output_dim,
            self.dropout,
            q_floor=self.q_floor,
            q_floor_mode=self.q_floor_mode,
            local_residual_scale=self.local_residual_scale,
            final_residual_scale=self.final_residual_scale,
            final_output_residual_scale=self.final_output_residual_scale,
            negative_output_residual_scale=self.negative_output_residual_scale,
            negative_residual_threshold=self.negative_residual_threshold,
            pos_output_residual_scale=self.pos_output_residual_scale,
            neg_output_residual_scale=self.neg_output_residual_scale,
            negative_residual_warmup_epochs=self.negative_residual_warmup_epochs,
            negative_residual_max_abs=self.negative_residual_max_abs,
            final_q_normalize=self.final_q_normalize,
            sign_aware_final=self.sign_aware_final,
            sign_negative_scale=self.sign_negative_scale,
            sign_positive_scale=self.sign_positive_scale,
            final_amplitude_gate=self.final_amplitude_gate,
            final_amplitude_scale=self.final_amplitude_scale,
            final_amplitude_temperature=self.final_amplitude_temperature,
            final_amplitude_warmup_epochs=self.final_amplitude_warmup_epochs,
            final_amplitude_warmup_start=self.final_amplitude_warmup_start,
            negative_evidence_gate=self.negative_evidence_gate,
            negative_evidence_scale=self.negative_evidence_scale,
            negative_evidence_threshold=self.negative_evidence_threshold,
            sign_aware_final_v2=self.sign_aware_final_v2,
            negative_evidence_projection=self.negative_evidence_projection,
            positive_evidence_scale=self.positive_evidence_scale,
            negative_gate_temperature=self.negative_gate_temperature,
            positive_amplitude_recovery=self.positive_amplitude_recovery,
            positive_amplitude_scale=self.positive_amplitude_scale,
            positive_threshold=self.positive_threshold,
            positive_amplitude_temperature=self.positive_amplitude_temperature,
            negative_amplitude_recovery=self.negative_amplitude_recovery,
            negative_amplitude_scale=self.negative_amplitude_scale,
            negative_threshold=self.negative_threshold,
            negative_amplitude_temperature=self.negative_amplitude_temperature,
            attack_aware_negative_final=self.attack_aware_negative_final,
            attack_evidence_scale=self.attack_evidence_scale,
            undercut_evidence_scale=self.undercut_evidence_scale,
            negative_final_temperature=self.negative_final_temperature,
            amplitude_preserving_fusion=self.amplitude_preserving_fusion,
            local_amplitude_residual_scale=self.local_amplitude_residual_scale,
            global_amplitude_residual_scale=self.global_amplitude_residual_scale,
            final_fusion_retention_repair=self.final_fusion_retention_repair,
            local_retention_residual_scale=self.local_retention_residual_scale,
            global_retention_residual_scale=self.global_retention_residual_scale,
            retention_gate_temperature=self.retention_gate_temperature,
            sign_preserve_gate=self.sign_preserve_gate,
            local_global_balance_gate=self.local_global_balance_gate,
            local_preserve_scale=self.local_preserve_scale,
            global_preserve_scale=self.global_preserve_scale,
            positive_final_retention=self.positive_final_retention,
            negative_output_amplitude_scale=self.negative_output_amplitude_scale,
            positive_output_amplitude_scale=self.positive_output_amplitude_scale,
            retention_output_residual_scale=self.retention_output_residual_scale,
            positive_retention_output_scale=self.positive_retention_output_scale,
            negative_retention_output_scale=self.negative_retention_output_scale,
            retention_output_residual=self.retention_output_residual,
            retention_output_mode=self.retention_output_mode,
            negative_retention_scale=self.negative_retention_scale,
            positive_retention_scale=self.positive_retention_scale,
            negative_confidence_threshold=self.negative_confidence_threshold,
            positive_confidence_threshold=self.positive_confidence_threshold,
            positive_residual_max_abs=self.positive_residual_max_abs,
            positive_residual_warmup_epochs=self.positive_residual_warmup_epochs,
            negative_evidence_consistency_residual=(
                self.negative_evidence_consistency_residual
                or self.negative_evidence_consistency_residual_v2
                or self.negative_evidence_consistency_residual_v3
                or self.negative_evidence_consistency_residual_v4
            ),
            negative_evidence_consistency_residual_v2=self.negative_evidence_consistency_residual_v2,
            negative_evidence_consistency_residual_v3=self.negative_evidence_consistency_residual_v3,
            negative_evidence_consistency_residual_v4=self.negative_evidence_consistency_residual_v4,
            negative_evidence_residual_v6=self.negative_evidence_residual_v6,
            negative_v6_threshold=self.negative_v6_threshold,
            negative_v6_scale=self.negative_v6_scale,
            negative_v6_max_abs=self.negative_v6_max_abs,
            negative_v6_warmup_epochs=self.negative_v6_warmup_epochs,
            negative_v6_trace=self.negative_v6_trace,
            negative_evidence_residual_v7=self.negative_evidence_residual_v7,
            negative_v7_threshold=self.negative_v7_threshold,
            negative_v7_scale=self.negative_v7_scale,
            negative_v7_max_abs=self.negative_v7_max_abs,
            negative_v7_warmup_epochs=self.negative_v7_warmup_epochs,
            negative_v7_trace=self.negative_v7_trace,
            negative_residual_path_gain=self.negative_residual_path_gain,
            negative_path_gain_mode=self.negative_path_gain_mode,
            negative_path_gain_scale=self.negative_path_gain_scale,
            negative_path_gain_max=self.negative_path_gain_max,
            negative_path_gain_min_gate=self.negative_path_gain_min_gate,
            negative_path_gain_require_neutral_block=self.negative_path_gain_require_neutral_block,
            negative_path_gain_require_weak_block=self.negative_path_gain_require_weak_block,
            negative_path_gain_trace=self.negative_path_gain_trace,
            negative_evidence_consistency_scale=self.negative_evidence_consistency_scale,
            negative_neutral_suppress_threshold=self.negative_neutral_suppress_threshold,
            negative_consistency_warmup_epochs=self.negative_consistency_warmup_epochs,
            positive_evidence_consistency_residual=(
                self.positive_evidence_consistency_residual
                or self.positive_evidence_consistency_residual_v2
                or self.positive_evidence_consistency_residual_v3
                or self.positive_evidence_consistency_residual_v4
            ),
            positive_evidence_consistency_residual_v2=self.positive_evidence_consistency_residual_v2,
            positive_evidence_consistency_residual_v3=self.positive_evidence_consistency_residual_v3,
            positive_evidence_consistency_residual_v4=self.positive_evidence_consistency_residual_v4,
            positive_evidence_consistency_scale=self.positive_evidence_consistency_scale,
            positive_evidence_threshold=self.positive_evidence_threshold,
            positive_neutral_suppress_threshold=self.positive_neutral_suppress_threshold,
            positive_consistency_warmup_epochs=self.positive_consistency_warmup_epochs,
            final_dispersion_preserve=self.final_dispersion_preserve,
            final_dispersion_target_std=self.final_dispersion_target_std,
            final_dispersion_max_expand=self.final_dispersion_max_expand,
            final_dispersion_decay_epochs=self.final_dispersion_decay_epochs,
            final_dispersion_boundary_safe=self.final_dispersion_boundary_safe,
            final_dispersion_boundary_margin=self.final_dispersion_boundary_margin,
            final_dispersion_boundary_scale=self.final_dispersion_boundary_scale,
            final_dispersion_residual_mix=self.final_dispersion_residual_mix,
        )
        self.argument_sentiment_probe = nn.Linear(self.hidden_dim, output_dim)

    def set_training_epoch(self, epoch: int) -> None:
        self.final_predictor.set_training_epoch(epoch)

    def _should_collect_iter_history(self, collect_iter_history: Optional[bool]) -> bool:
        if collect_iter_history is not None:
            return bool(collect_iter_history)
        return bool(
            _get_arg(self.args, "enable_iter_analysis", False)
            or _get_arg(self.args, "return_debug_tensors", False)
        )

    def _build_iter_history(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        q_i: torch.Tensor,
        g_t: torch.Tensor,
        segmask: torch.Tensor,
    ) -> Dict[str, list]:
        return {
            # State index 0 stores the initialization before the first internal
            # reasoning iteration. State index t stores the result after t rounds.
            "v_states": [v_i],
            "p_states": [p_i],
            "q_states": [q_i],
            "g_states": [g_t],
            "iter_predictions": [],
            "segment_mask": segmask,
        }

    def _retention_confidences(
        self,
        v_i: torch.Tensor,
        p_i: torch.Tensor,
        q_i: torch.Tensor,
        segmask: torch.Tensor,
        support_evidence: Optional[torch.Tensor],
        attack_evidence: Optional[torch.Tensor],
        undercut_evidence: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = segmask.float().unsqueeze(-1)
        batch_size = v_i.size(0)
        device = v_i.device
        dtype = v_i.dtype
        zero = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        q_effective = _apply_q_floor(q_i, self.q_floor, mode=self.q_floor_mode, mask=segmask) * mask
        node_sentiment = self.argument_sentiment_probe(v_i)
        argument_score = _masked_weighted_mean(node_sentiment, q_effective, segmask)
        q_sum = q_effective.sum(dim=1).clamp(min=1e-6)
        h_local_weighted = (v_i * q_effective).sum(dim=1) / q_sum
        h_local_score = self.argument_sentiment_probe(h_local_weighted)
        role_support = _masked_weighted_mean(p_i[..., 0:1], q_effective, segmask)
        role_attack = _masked_weighted_mean(p_i[..., 1:2], q_effective, segmask)

        support_signal = support_evidence.to(device=device, dtype=dtype) if support_evidence is not None else zero
        attack_signal = attack_evidence.to(device=device, dtype=dtype) if attack_evidence is not None else zero
        undercut_signal = undercut_evidence.to(device=device, dtype=dtype) if undercut_evidence is not None else zero

        use_negative_v2 = (
            self.negative_evidence_consistency_residual_v2
            and self.negative_confidence_source == "node_relation_hlocal"
        )
        use_positive_v2 = (
            self.positive_evidence_consistency_residual_v2
            and self.positive_confidence_source == "node_relation_hlocal"
        )
        use_negative_v3 = (
            self.negative_evidence_consistency_residual_v3
            and self.negative_confidence_source == "decomposed_node_relation_hlocal"
        )
        use_positive_v3 = (
            self.positive_evidence_consistency_residual_v3
            and self.positive_confidence_source == "decomposed_node_relation_hlocal"
        )
        use_negative_v4 = (
            self.negative_evidence_consistency_residual_v4
            and self.negative_confidence_source == "direction_filtered_node_hlocal"
        )
        use_negative_v6 = bool(self.negative_evidence_residual_v6)
        use_negative_v7 = bool(self.negative_evidence_residual_v7)
        use_positive_v4 = (
            self.positive_evidence_consistency_residual_v4
            and self.positive_confidence_source == "intensity_aware_node_hlocal"
        )

        support_conf = torch.sigmoid(support_signal)
        attack_conf = torch.sigmoid(attack_signal)
        undercut_conf = torch.sigmoid(undercut_signal)
        positive_argument_conf = torch.sigmoid(argument_score)
        negative_argument_conf = torch.sigmoid(-argument_score)
        positive_hlocal_conf = torch.sigmoid(h_local_score)
        negative_hlocal_conf = torch.sigmoid(-h_local_score)
        self._last_retention_confidence_components = {}

        if (
            use_negative_v2
            or use_positive_v2
            or use_negative_v3
            or use_positive_v3
            or use_negative_v4
            or use_negative_v6
            or use_negative_v7
            or use_positive_v4
        ):
            q_weighted_negative_node = _masked_weighted_mean(
                F.softplus(-node_sentiment),
                q_effective,
                segmask,
            )
            q_weighted_positive_node = _masked_weighted_mean(
                F.softplus(node_sentiment),
                q_effective,
                segmask,
            )
            q_negative_conf = torch.sigmoid(q_weighted_negative_node - q_weighted_positive_node)
            q_positive_conf = torch.sigmoid(q_weighted_positive_node - q_weighted_negative_node)
            attack_margin_conf = torch.sigmoid(attack_signal + 0.5 * undercut_signal - support_signal)
            support_margin_conf = torch.sigmoid(support_signal - attack_signal - 0.5 * undercut_signal)
            role_attack_margin = torch.sigmoid((role_attack - role_support) / 0.15)
            role_support_margin = torch.sigmoid((role_support - role_attack) / 0.15)
            node_abs_conf = torch.sigmoid((node_sentiment.abs() - 0.35) / 0.10)
            h_local_abs_conf = torch.sigmoid((h_local_score.abs() - 0.35) / 0.10)
            q_weighted_node_strength = _masked_weighted_mean(
                node_abs_conf,
                q_effective,
                segmask,
            )
            negative_attack_component = torch.sigmoid(attack_signal - support_signal)
            positive_support_component = torch.sigmoid(support_signal - attack_signal)
            negative_q_node_component = q_weighted_node_strength * role_attack_margin
            positive_q_node_component = q_weighted_node_strength * role_support_margin
            negative_hlocal_component = h_local_abs_conf * role_attack_margin
            positive_hlocal_component = h_local_abs_conf * role_support_margin
            negative_neutral_suppression = 1.0 - torch.maximum(
                torch.maximum(negative_attack_component, negative_q_node_component),
                torch.maximum(negative_hlocal_component, role_attack_margin),
            )
            positive_neutral_suppression = 1.0 - torch.maximum(
                torch.maximum(positive_support_component, positive_q_node_component),
                torch.maximum(positive_hlocal_component, role_support_margin),
            )
            negative_neutral_suppression = negative_neutral_suppression.clamp(0.0, 1.0)
            positive_neutral_suppression = positive_neutral_suppression.clamp(0.0, 1.0)
            negative_node_strength = _masked_weighted_mean(
                torch.sigmoid(
                    ((-node_sentiment) - self.negative_neutral_suppress_threshold) / 0.08
                ),
                q_effective,
                segmask,
            )
            negative_hlocal_strength = torch.sigmoid(
                ((-h_local_score) - self.negative_neutral_suppress_threshold) / 0.08
            )
            negative_strength_agreement = torch.sqrt(
                (negative_node_strength * negative_hlocal_strength).clamp(min=0.0)
            )
            positive_node_strength = _masked_weighted_mean(
                torch.sigmoid(
                    (node_sentiment - self.positive_intensity_threshold) / 0.08
                ),
                q_effective,
                segmask,
            )
            positive_hlocal_strength = torch.sigmoid(
                (h_local_score - self.positive_intensity_threshold) / 0.08
            )
            positive_intensity_gate = torch.sqrt(
                (positive_node_strength * positive_hlocal_strength).clamp(min=0.0)
            )
            positive_weak_suppression = torch.sigmoid(
                (
                    torch.maximum(q_weighted_positive_node, h_local_score)
                    - self.positive_weak_suppress_threshold
                )
                / 0.08
            )
            negative_polarity = 0.40 + 0.60 * torch.sigmoid(
                (
                    torch.maximum(-argument_score, -h_local_score)
                    - self.negative_neutral_suppress_threshold
                )
                / 0.08
            )
            positive_polarity = 0.40 + 0.60 * torch.sigmoid(
                (
                    torch.maximum(argument_score, h_local_score)
                    - self.positive_neutral_suppress_threshold
                )
                / 0.08
            )
            negative_score_raw = (
                attack_margin_conf
                + undercut_conf
                + role_attack_margin
                + q_negative_conf
                + negative_argument_conf
                + negative_hlocal_conf
            ) / 6.0
            positive_score_raw = (
                support_margin_conf
                + role_support_margin
                + q_positive_conf
                + positive_argument_conf
                + positive_hlocal_conf
            ) / 5.0
            negative_score_v2 = (
                negative_score_raw - 0.10 * positive_score_raw
            ).clamp(0.0, 1.0) * negative_polarity
            positive_score_v2 = (
                positive_score_raw - 0.10 * negative_score_raw
            ).clamp(0.0, 1.0) * positive_polarity

            def mean_selected(items):
                selected = [value for enabled, value in items if enabled]
                if not selected:
                    return torch.zeros_like(zero)
                return torch.stack(selected, dim=0).mean(dim=0)

            negative_score_v3 = mean_selected(
                [
                    (self.negative_confidence_use_attack, negative_attack_component),
                    (self.negative_confidence_use_hlocal, negative_hlocal_component),
                    (self.negative_confidence_use_q_node, negative_q_node_component),
                    (self.negative_confidence_use_role, role_attack_margin),
                    (self.negative_confidence_use_undercut, undercut_conf),
                ]
            )
            if self.negative_confidence_neutral_suppress:
                negative_score_v3 = (
                    negative_score_v3 - 0.25 * negative_neutral_suppression
                ).clamp(0.0, 1.0)
            positive_score_v3 = mean_selected(
                [
                    (self.positive_confidence_use_support, positive_support_component),
                    (self.positive_confidence_use_hlocal, positive_hlocal_component),
                    (self.positive_confidence_use_q_node, positive_q_node_component),
                    (self.positive_confidence_use_role, role_support_margin),
                ]
            )
            if self.positive_confidence_neutral_suppress:
                positive_score_v3 = (
                    positive_score_v3 - 0.25 * positive_neutral_suppression
                ).clamp(0.0, 1.0)
            attack_component_v4 = negative_attack_component
            if _config_auto_filter(self.negative_confidence_use_attack):
                attack_component_v4 = attack_component_v4 * negative_strength_agreement
            role_component_v4 = role_attack_margin
            if _config_auto_filter(self.negative_confidence_use_role):
                role_component_v4 = role_component_v4 * negative_strength_agreement
            negative_residual_direction_proxy = negative_strength_agreement
            negative_score_v4 = mean_selected(
                [
                    (_config_enabled(self.negative_confidence_use_hlocal), negative_hlocal_strength),
                    (_config_enabled(self.negative_confidence_use_q_node), negative_node_strength),
                    (
                        _config_enabled(self.negative_confidence_use_residual_direction),
                        negative_residual_direction_proxy,
                    ),
                    (_config_enabled(self.negative_confidence_use_attack), attack_component_v4),
                    (_config_enabled(self.negative_confidence_use_role), role_component_v4),
                    (_config_enabled(self.negative_confidence_use_undercut), undercut_conf),
                ]
            )
            if self.negative_confidence_neutral_suppress:
                negative_score_v4 = (
                    negative_score_v4 - 0.35 * (1.0 - negative_strength_agreement)
                ).clamp(0.0, 1.0)
            v6_items = []
            if self.negative_v6_use_hlocal:
                v6_items.append(1.15 * negative_hlocal_strength)
            if self.negative_v6_use_q_node:
                v6_items.append(1.10 * negative_node_strength)
            if self.negative_v6_use_residual_direction:
                v6_items.append(negative_strength_agreement)
            if self.negative_v6_use_attack_role:
                attack_role_score = torch.sqrt(
                    (negative_attack_component * role_attack_margin).clamp(min=0.0)
                )
                v6_items.append(attack_role_score * negative_strength_agreement)
            if v6_items:
                negative_score_v6_raw = torch.stack(v6_items, dim=0).mean(dim=0)
            else:
                negative_score_v6_raw = torch.zeros_like(zero)
            v6_positive_suppression = torch.stack(
                [
                    q_positive_conf,
                    positive_hlocal_strength,
                    positive_support_component,
                    role_support_margin,
                ],
                dim=0,
            ).mean(dim=0)
            v6_neutral_suppression = (1.0 - torch.maximum(
                negative_node_strength,
                negative_hlocal_strength,
            )).clamp(0.0, 1.0)
            if self.negative_v6_use_positive_suppression:
                negative_score_v6_raw = negative_score_v6_raw - 0.35 * v6_positive_suppression
            if self.negative_v6_use_neutral_suppression:
                negative_score_v6_raw = negative_score_v6_raw - 0.30 * v6_neutral_suppression
            if self.negative_v6_score_norm == "batch_zscore":
                score_mean = negative_score_v6_raw.mean(dim=0, keepdim=True)
                score_std = negative_score_v6_raw.std(dim=0, unbiased=False, keepdim=True)
                negative_score_v6_norm = (negative_score_v6_raw - score_mean) / score_std.clamp(min=1e-6)
                negative_score_v6_conf = torch.sigmoid(negative_score_v6_norm)
            elif self.negative_v6_score_norm == "minmax":
                score_min = negative_score_v6_raw.min(dim=0, keepdim=True).values
                score_max = negative_score_v6_raw.max(dim=0, keepdim=True).values
                negative_score_v6_norm = (
                    (negative_score_v6_raw - score_min)
                    / (score_max - score_min).clamp(min=1e-6)
                )
                negative_score_v6_conf = negative_score_v6_norm.clamp(0.0, 1.0)
            else:
                negative_score_v6_norm = negative_score_v6_raw
                negative_score_v6_conf = torch.sigmoid(negative_score_v6_raw)
            v7_items = []
            if self.negative_v7_use_hlocal:
                v7_items.append(negative_hlocal_strength)
            if self.negative_v7_use_q_node:
                v7_items.append(negative_node_strength)
            if self.negative_v7_use_residual_direction:
                v7_items.append(negative_strength_agreement)
            v7_attack_role_score = torch.sqrt(
                (negative_attack_component * role_attack_margin).clamp(min=0.0)
            )
            if self.negative_v7_use_attack_role:
                v7_items.append(v7_attack_role_score * negative_strength_agreement)
            if v7_items:
                negative_score_v7_base = torch.stack(v7_items, dim=0).mean(dim=0)
            else:
                negative_score_v7_base = torch.zeros_like(zero)
            max_negative_strength = torch.maximum(negative_node_strength, negative_hlocal_strength)
            strength_min = max_negative_strength.min(dim=0, keepdim=True).values
            strength_max = max_negative_strength.max(dim=0, keepdim=True).values
            v7_relative_strength = (
                (max_negative_strength - strength_min)
                / (strength_max - strength_min).clamp(min=1e-6)
            ).clamp(0.0, 1.0)
            v7_positive_suppression = torch.stack(
                [
                    q_positive_conf,
                    positive_hlocal_strength,
                    positive_support_component,
                    role_support_margin,
                ],
                dim=0,
            ).mean(dim=0)
            v7_neutral_score = torch.sigmoid(
                (self.negative_v7_neutral_suppress_threshold - v7_relative_strength) / 0.08
            ).clamp(0.0, 1.0)
            v7_weak_negative_score = (
                torch.sigmoid(
                    (v7_relative_strength - self.negative_v7_weak_suppress_threshold) / 0.08
                )
                * torch.sigmoid(
                    (self.negative_v7_strong_threshold - v7_relative_strength) / 0.08
                )
            ).clamp(0.0, 1.0)
            negative_score_v7_raw = negative_score_v7_base
            if self.negative_v7_use_positive_suppression:
                negative_score_v7_raw = negative_score_v7_raw - 0.35 * v7_positive_suppression
            if self.negative_v7_use_neutral_suppression:
                negative_score_v7_raw = negative_score_v7_raw - 0.45 * v7_neutral_score
            if self.negative_v7_use_weak_negative_suppression:
                negative_score_v7_raw = negative_score_v7_raw - 0.50 * v7_weak_negative_score
            if self.negative_v7_score_norm == "batch_zscore":
                v7_score_mean = negative_score_v7_raw.mean(dim=0, keepdim=True)
                v7_score_std = negative_score_v7_raw.std(dim=0, unbiased=False, keepdim=True)
                negative_score_v7_norm = (
                    negative_score_v7_raw - v7_score_mean
                ) / v7_score_std.clamp(min=1e-6)
                negative_score_v7_base_conf = torch.sigmoid(negative_score_v7_norm)
            elif self.negative_v7_score_norm == "minmax":
                v7_score_min = negative_score_v7_raw.min(dim=0, keepdim=True).values
                v7_score_max = negative_score_v7_raw.max(dim=0, keepdim=True).values
                negative_score_v7_norm = (
                    (negative_score_v7_raw - v7_score_min)
                    / (v7_score_max - v7_score_min).clamp(min=1e-6)
                )
                negative_score_v7_base_conf = negative_score_v7_norm.clamp(0.0, 1.0)
            else:
                negative_score_v7_norm = negative_score_v7_raw
                negative_score_v7_base_conf = torch.sigmoid(negative_score_v7_raw)
            v7_strong_gate_score = torch.sigmoid(
                (v7_relative_strength - self.negative_v7_strong_threshold) / 0.08
            ).clamp(0.0, 1.0)
            v7_neutral_keep = torch.sigmoid(
                (self.negative_v7_neutral_suppress_threshold - v7_neutral_score) / 0.08
            ).clamp(0.0, 1.0)
            v7_weak_keep = torch.sigmoid(
                (self.negative_v7_weak_suppress_threshold - v7_weak_negative_score) / 0.08
            ).clamp(0.0, 1.0)
            negative_score_v7_conf = (
                negative_score_v7_base_conf
                * v7_strong_gate_score
                * v7_neutral_keep
                * v7_weak_keep
            ).clamp(0.0, 1.0)
            positive_base_v4 = mean_selected(
                [
                    (self.positive_confidence_use_support, positive_support_component),
                    (self.positive_confidence_use_hlocal, positive_hlocal_strength),
                    (self.positive_confidence_use_q_node, positive_node_strength),
                    (self.positive_confidence_use_role, role_support_margin),
                ]
            )
            if self.positive_intensity_gate:
                positive_score_v4 = positive_base_v4 * positive_intensity_gate * positive_weak_suppression
            else:
                positive_score_v4 = positive_base_v4 * positive_weak_suppression
            if self.positive_confidence_neutral_suppress:
                positive_score_v4 = (
                    positive_score_v4 - 0.25 * positive_neutral_suppression
                ).clamp(0.0, 1.0)
            self._last_retention_confidence_components = {
                "negative_confidence_component_attack_evidence_strength": negative_attack_component,
                "negative_confidence_component_direction_filtered_attack_evidence": attack_component_v4,
                "negative_confidence_component_q_weighted_negative_node_evidence": negative_q_node_component,
                "negative_confidence_component_valid_q_weighted_negative_node": negative_node_strength,
                "negative_confidence_component_h_local_negative_projection": negative_hlocal_component,
                "negative_confidence_component_valid_negative_hlocal_projection": negative_hlocal_strength,
                "negative_confidence_component_role_attack_probability": role_attack_margin,
                "negative_confidence_component_direction_filtered_role_attack_probability": role_component_v4,
                "negative_confidence_component_undercut_evidence_strength": undercut_conf,
                "negative_confidence_component_negative_residual_direction_proxy": negative_residual_direction_proxy,
                "negative_confidence_component_near_neutral_suppression": negative_neutral_suppression,
                "negative_confidence_component_v6_raw_score": negative_score_v6_raw,
                "negative_confidence_component_v6_norm_score": negative_score_v6_norm,
                "negative_confidence_component_v6_confidence": negative_score_v6_conf,
                "negative_confidence_component_v6_hlocal_score": negative_hlocal_strength,
                "negative_confidence_component_v6_q_node_score": negative_node_strength,
                "negative_confidence_component_v6_residual_direction_score": negative_strength_agreement,
                "negative_confidence_component_v6_attack_role_score": torch.sqrt(
                    (negative_attack_component * role_attack_margin).clamp(min=0.0)
                ) * negative_strength_agreement,
                "negative_confidence_component_v6_positive_suppression": v6_positive_suppression,
                "negative_confidence_component_v6_neutral_suppression": v6_neutral_suppression,
                "negative_confidence_component_v7_raw_score": negative_score_v7_raw,
                "negative_confidence_component_v7_norm_score": negative_score_v7_norm,
                "negative_confidence_component_v7_base_confidence": negative_score_v7_base_conf,
                "negative_confidence_component_v7_confidence": negative_score_v7_conf,
                "negative_confidence_component_v7_hlocal_score": negative_hlocal_strength,
                "negative_confidence_component_v7_q_node_score": negative_node_strength,
                "negative_confidence_component_v7_residual_direction_score": negative_strength_agreement,
                "negative_confidence_component_v7_relative_strength": v7_relative_strength,
                "negative_confidence_component_v7_attack_role_score": v7_attack_role_score * negative_strength_agreement,
                "negative_confidence_component_v7_positive_suppression": v7_positive_suppression,
                "negative_confidence_component_v7_neutral_score": v7_neutral_score,
                "negative_confidence_component_v7_weak_negative_score": v7_weak_negative_score,
                "negative_confidence_component_v7_strong_gate_score": v7_strong_gate_score,
                "negative_confidence_component_v7_neutral_keep": v7_neutral_keep,
                "negative_confidence_component_v7_weak_keep": v7_weak_keep,
                "negative_confidence_component_legacy_negative_argument": negative_argument_conf,
                "negative_confidence_component_legacy_negative_hlocal": negative_hlocal_conf,
                "positive_confidence_component_support_evidence_strength": positive_support_component,
                "positive_confidence_component_q_weighted_positive_node_evidence": positive_q_node_component,
                "positive_confidence_component_intensity_q_weighted_positive_node": positive_node_strength,
                "positive_confidence_component_h_local_positive_projection": positive_hlocal_component,
                "positive_confidence_component_intensity_h_local_positive_projection": positive_hlocal_strength,
                "positive_confidence_component_role_support_probability": role_support_margin,
                "positive_confidence_component_positive_intensity_gate": positive_intensity_gate,
                "positive_confidence_component_positive_weak_suppression": positive_weak_suppression,
                "positive_confidence_component_near_neutral_suppression": positive_neutral_suppression,
                "positive_confidence_component_legacy_positive_argument": positive_argument_conf,
                "positive_confidence_component_legacy_positive_hlocal": positive_hlocal_conf,
            }

        negative_score = (
            attack_conf
            + undercut_conf
            + role_attack
            + negative_argument_conf
            + negative_hlocal_conf
        ) / 5.0
        positive_score = (
            support_conf
            + role_support
            + positive_argument_conf
            + positive_hlocal_conf
        ) / 4.0
        if use_negative_v2:
            negative_score = negative_score_v2
        if use_positive_v2:
            positive_score = positive_score_v2
        if use_negative_v3:
            negative_score = negative_score_v3
        if use_positive_v3:
            positive_score = positive_score_v3
        if use_negative_v4:
            negative_score = negative_score_v4
        if use_negative_v6:
            negative_score = negative_score_v6_conf
        if use_negative_v7:
            negative_score = negative_score_v7_conf
        if use_positive_v4:
            positive_score = positive_score_v4
        negative_center = (
            0.35 if use_negative_v4 else (0.45 if use_negative_v3 else (0.40 if use_negative_v2 else 0.50))
        )
        positive_center = (
            0.30 if use_positive_v4 else (0.45 if use_positive_v3 else (0.40 if use_positive_v2 else 0.50))
        )
        negative_temperature = 0.08 if (use_negative_v2 or use_negative_v3 or use_negative_v4) else 0.05
        positive_temperature = 0.08 if (use_positive_v2 or use_positive_v3 or use_positive_v4) else 0.02
        negative_confidence = torch.sigmoid((negative_score - negative_center) / negative_temperature)
        positive_confidence = torch.sigmoid((positive_score - positive_center) / positive_temperature)
        if use_negative_v6:
            negative_confidence = negative_score_v6_conf
        if use_negative_v7:
            negative_confidence = negative_score_v7_conf
        negative_confidence = negative_confidence.clamp(0.0, 1.0)
        positive_confidence = positive_confidence.clamp(0.0, 1.0)
        if self._last_retention_confidence_components:
            self._last_retention_confidence_components.update(
                {
                    "negative_confidence_component_negative_confidence_score": negative_confidence,
                    "positive_confidence_component_positive_confidence_score": positive_confidence,
                }
            )
        return negative_confidence, positive_confidence

    def forward(
        self,
        text_x: torch.Tensor,
        audio_x: torch.Tensor,
        video_x: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        vision_lengths: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        collect_iter_history: Optional[bool] = None,
        collect_layer_trace: Optional[bool] = None,
        return_analysis: bool = False,
        sample_id: Optional[Any] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # return_analysis is used only by offline motivation experiments. Default
        # forward behavior stays unchanged for normal train/valid/test calls.
        collect_layer_trace = bool(
            return_analysis
            or collect_layer_trace
            or _get_arg(self.args, "residual_sign_debias", False)
        )
        collect_iter_history = bool(return_analysis) or self._should_collect_iter_history(
            collect_iter_history
        ) or collect_layer_trace
        text_mask, text_lengths = _prepare_mask_and_lengths(text_x, text_lengths)
        audio_mask, audio_lengths = _prepare_mask_and_lengths(audio_x, audio_lengths)
        vision_mask, vision_lengths = _prepare_mask_and_lengths(video_x, vision_lengths)

        # u_t^T, u_t^A, u_t^V: local unimodal contextual encodings.
        encoder_traces = None
        if collect_layer_trace:
            u_t_t, text_trace = self.text_encoder(
                text_x, text_lengths, return_trace=True
            )
            u_t_a, audio_trace = self.audio_encoder(
                audio_x, audio_lengths, return_trace=True
            )
            u_t_v, vision_trace = self.vision_encoder(
                video_x, vision_lengths, return_trace=True
            )
            encoder_traces = {
                "text": text_trace,
                "audio": audio_trace,
                "vision": vision_trace,
            }
        else:
            u_t_t = self.text_encoder(text_x, text_lengths)
            u_t_a = self.audio_encoder(audio_x, audio_lengths)
            u_t_v = self.vision_encoder(video_x, vision_lengths)

        # h_t: local multimodal segment representation in a shared hidden space.
        h_t, h_mask = self.local_fusion(
            u_t_t, u_t_a, u_t_v, text_mask, audio_mask, vision_mask
        )

        pooled_h = _masked_mean(h_t, h_mask)
        g_init_raw = self.global_init(pooled_h)
        g_t = g_init_raw
        if self.global_init_residual_scale > 0.0:
            g_t = g_t + self.global_init_residual_scale * pooled_h

        # a_i: initial argument units from fixed-window segment pooling.
        a_i, segmask, c_i, attention_entropy = self.segment_builder(h_t, h_mask)

        # p_i: support / attack / undercut role probabilities for each segment.
        p_i = self.role_classifier(a_i, g_t)
        p_initial = p_i
        role_sets, role_weights = self.role_set_builder(a_i, p_i, segmask)

        if collect_layer_trace:
            v_i, node_proj_input = self.node_initializer(
                a_i, p_i, g_t, c_i, return_input=True
            )
        else:
            v_i = self.node_initializer(a_i, p_i, g_t, c_i)
            node_proj_input = None
        v_i = v_i * segmask.float().unsqueeze(-1)
        v_initial = v_i
        g_initial = g_t
        if collect_layer_trace:
            q_i, q_init_debug = self.acceptability_initializer.initialize(
                v_i, p_i, g_t, segmask, return_debug=True
            )
        else:
            q_i = self.acceptability_initializer.initialize(v_i, p_i, g_t, segmask)
            q_init_debug = {}

        iter_history = None
        if collect_iter_history:
            iter_history = self._build_iter_history(v_i, p_i, q_i, g_t, segmask)

        last_debug = {}
        iter_debugs = []
        final_support_evidence = None
        final_attack_evidence = None
        final_undercut_evidence = None
        for reasoning_block in self.reasoning_blocks:
            v_i, p_i, q_i, g_t, last_debug = reasoning_block(
                v_i,
                p_i,
                q_i,
                g_t,
                segmask,
                node_anchor=v_initial,
                pooled_anchor=pooled_h,
            )
            iter_debugs.append(last_debug)
            support_message = last_debug.get("support_message")
            attack_message = last_debug.get("attack_message")
            undercut_message = last_debug.get("undercut_message")
            gate_beta = last_debug.get("gate_beta")
            if torch.is_tensor(support_message) and torch.is_tensor(gate_beta):
                mask = segmask.float().unsqueeze(-1)
                final_support_evidence = (
                    support_message.norm(dim=-1, keepdim=True) * gate_beta[..., 0:1] * mask
                ).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if torch.is_tensor(attack_message) and torch.is_tensor(gate_beta):
                mask = segmask.float().unsqueeze(-1)
                final_attack_evidence = (
                    attack_message.norm(dim=-1, keepdim=True) * gate_beta[..., 1:2] * mask
                ).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if torch.is_tensor(undercut_message) and torch.is_tensor(gate_beta):
                mask = segmask.float().unsqueeze(-1)
                final_undercut_evidence = (
                    undercut_message.norm(dim=-1, keepdim=True) * gate_beta[..., 2:3] * mask
                ).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if iter_history is not None:
                # y_hat^(t) is the prediction after the t-th internal reasoning step.
                iter_negative_confidence, iter_positive_confidence = self._retention_confidences(
                    v_i,
                    p_i,
                    q_i,
                    segmask,
                    final_support_evidence,
                    final_attack_evidence,
                    final_undercut_evidence,
                )
                iter_y_hat = self.final_predictor(
                    v_i,
                    q_i,
                    g_t,
                    segmask,
                    attack_final_evidence=final_attack_evidence,
                    undercut_final_evidence=final_undercut_evidence,
                    negative_retention_confidence=iter_negative_confidence,
                    positive_retention_confidence=iter_positive_confidence,
                )[0]
                iter_history["v_states"].append(v_i)
                iter_history["p_states"].append(p_i)
                iter_history["q_states"].append(q_i)
                iter_history["g_states"].append(g_t)
                iter_history["iter_predictions"].append(iter_y_hat)

        final_predictor_trace = {}
        negative_retention_confidence, positive_retention_confidence = self._retention_confidences(
            v_i,
            p_i,
            q_i,
            segmask,
            final_support_evidence,
            final_attack_evidence,
            final_undercut_evidence,
        )
        if collect_layer_trace:
            (
                y_hat,
                h_final,
                h_local,
                final_fusion_input,
                h_local_weighted,
                h_local_residual,
                q_effective,
                final_predictor_trace,
            ) = self.final_predictor(
                v_i,
                q_i,
                g_t,
                segmask,
                attack_final_evidence=final_attack_evidence,
                undercut_final_evidence=final_undercut_evidence,
                negative_retention_confidence=negative_retention_confidence,
                positive_retention_confidence=positive_retention_confidence,
                return_trace=True,
            )
        else:
            (
                y_hat,
                h_final,
                h_local,
                final_fusion_input,
                h_local_weighted,
                h_local_residual,
                q_effective,
            ) = self.final_predictor(
                v_i,
                q_i,
                g_t,
                segmask,
                attack_final_evidence=final_attack_evidence,
                undercut_final_evidence=final_undercut_evidence,
                negative_retention_confidence=negative_retention_confidence,
                positive_retention_confidence=positive_retention_confidence,
            )

        outputs = {
            "Feature_t": _masked_mean(u_t_t, text_mask),
            "Feature_a": _masked_mean(u_t_a, audio_mask),
            "Feature_v": _masked_mean(u_t_v, vision_mask),
            "Feature_f": h_final,
            "M": y_hat,
            "h_local": h_local,
            "argument_units": a_i,
            "node_states": v_i,
            "role_sets": role_sets,
            "role_weights": role_weights,
            "segment_acceptability": q_i,
            "segment_acceptability_effective": q_effective,
            "attention_entropy": attention_entropy,
            "global_state": g_t,
            "argument_sentiment": _masked_weighted_mean(
                self.argument_sentiment_probe(v_i),
                q_effective,
                segmask,
            ),
            "text_mask_valid": text_mask.long().sum(dim=1),
            "audio_mask_valid": audio_mask.long().sum(dim=1),
            "vision_mask_valid": vision_mask.long().sum(dim=1),
            "segment_mask_valid": segmask.long().sum(dim=1),
        }
        if iter_history is not None:
            outputs["iter_history"] = iter_history
        if collect_layer_trace:
            layer_trace = {
                "raw_text": encoder_traces["text"]["raw"],
                "raw_audio": encoder_traces["audio"]["raw"],
                "raw_vision": encoder_traces["vision"]["raw"],
                "clean_text": encoder_traces["text"]["clean"],
                "clean_audio": encoder_traces["audio"]["clean"],
                "clean_vision": encoder_traces["vision"]["clean"],
                "norm_text": encoder_traces["text"]["norm"],
                "norm_audio": encoder_traces["audio"]["norm"],
                "norm_vision": encoder_traces["vision"]["norm"],
                "gru_input_text": encoder_traces["text"]["gru_input"],
                "gru_input_audio": encoder_traces["audio"]["gru_input"],
                "gru_input_vision": encoder_traces["vision"]["gru_input"],
                "gru_output_text": encoder_traces["text"]["gru_output"],
                "gru_output_audio": encoder_traces["audio"]["gru_output"],
                "gru_output_vision": encoder_traces["vision"]["gru_output"],
                "encoded_text": u_t_t,
                "encoded_audio": u_t_a,
                "encoded_vision": u_t_v,
                "Feature_t": outputs["Feature_t"],
                "Feature_a": outputs["Feature_a"],
                "Feature_v": outputs["Feature_v"],
                "h_t": h_t,
                "pooled_h": pooled_h,
                "global_init_raw": g_init_raw,
                "argument_units": a_i,
                "attention_entropy": attention_entropy,
                "role_probs_initial": p_initial,
                "node_proj_input": node_proj_input,
                "node_states_initial": v_initial,
                "q_input_iter_0": q_init_debug.get("q_input"),
                "q_logits_iter_0": q_init_debug.get("q_logits"),
                "q_logits_prepared_iter_0": q_init_debug.get("q_logits_prepared"),
                "h_local": h_local,
                "h_local_weighted": h_local_weighted,
                "h_local_residual": h_local_residual,
                "q_times_node_final": v_i * q_effective,
                "final_fusion_input": final_fusion_input,
                "segment_acceptability_effective": q_effective,
                "global_state_initial": g_initial,
                "global_state": g_t,
                "role_probs_final": p_i,
                "h_final": h_final,
                "M": y_hat,
                "text_mask_valid": outputs["text_mask_valid"],
                "audio_mask_valid": outputs["audio_mask_valid"],
                "vision_mask_valid": outputs["vision_mask_valid"],
                "segment_mask_valid": outputs["segment_mask_valid"],
            }
            for trace_key, trace_value in final_predictor_trace.items():
                layer_trace[trace_key] = trace_value
            for trace_key, trace_value in getattr(
                self, "_last_retention_confidence_components", {}
            ).items():
                if torch.is_tensor(trace_value):
                    layer_trace[trace_key] = trace_value
            if iter_history is not None:
                for state_idx, state in enumerate(iter_history["v_states"][1:], start=1):
                    layer_trace[f"node_states_after_iter_{state_idx}"] = state
                for state_idx, state in enumerate(iter_history["g_states"][1:], start=1):
                    layer_trace[f"global_state_after_iter_{state_idx}"] = state
                for state_idx, state in enumerate(iter_history["q_states"], start=0):
                    layer_trace[f"segment_acceptability_iter_{state_idx}"] = state
                for iter_idx, debug in enumerate(iter_debugs, start=1):
                    for debug_key in [
                        "node_states_before",
                        "support_message",
                        "attack_message",
                        "undercut_message",
                        "fused_message",
                        "gate_beta",
                        "support",
                        "attack",
                        "undercut",
                        "relation_input",
                        "relation_input_loc",
                        "relation_input_role",
                        "relation_input_global",
                        "relation_fuse",
                        "relation_confidence",
                        "relation_logits",
                        "support_logits",
                        "attack_logits",
                        "undercut_logits",
                        "q_input",
                        "q_base_input",
                        "q_relation_features",
                        "q_logits",
                        "q_logits_prepared",
                        "q_hat",
                        "q_before_floor",
                        "q_after_floor",
                        "global_update_input",
                        "global_before",
                        "global_after",
                        "global_provisional_input",
                    ]:
                        debug_value = debug.get(debug_key)
                        if torch.is_tensor(debug_value):
                            layer_trace[f"{debug_key}_iter_{iter_idx}"] = debug_value
            outputs["layer_trace"] = layer_trace
        if return_analysis:
            outputs["analysis"] = {
                "sample_id": sample_id,
                "label": label,
                "final_pred": y_hat,
                "argument_units": a_i,
                "node_representations": v_i,
                "role_probs": p_i,
                "initial_role_probs": role_weights,
                "q_initial": iter_history["q_states"][0] if iter_history is not None else q_i,
                "q_states": iter_history["q_states"] if iter_history is not None else [q_i],
                "q_final": q_i,
                "pooled_text": outputs["Feature_t"],
                "pooled_audio": outputs["Feature_a"],
                "pooled_vision": outputs["Feature_v"],
                "support_matrix": last_debug.get("support"),
                "attack_matrix": last_debug.get("attack"),
                "undercut_matrix": last_debug.get("undercut"),
                "segment_mask": segmask,
            }
        if _get_arg(self.args, "return_debug_tensors", False):
            outputs.update(
                {
                    "support_matrix": last_debug.get("support"),
                    "attack_matrix": last_debug.get("attack"),
                    "undercut_matrix": last_debug.get("undercut"),
                    "gate_beta": last_debug.get("gate_beta"),
                }
            )
        return outputs
