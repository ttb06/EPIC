import math
from dataclasses import dataclass
from itertools import combinations
from logging import getLogger
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import log
from LLaDA import LLaDAConfig, LLaDAModelLM
from LLaDA.modeling_llada import (CausalLMOutputWithPast, ModuleType,
                                  init_weights)


@dataclass
class LLaDARecOutPut:
    loss: torch.Tensor
    his_mask_loss: torch.Tensor
    target_mask_loss: torch.Tensor
    item_arch_loss: Optional[torch.Tensor] = None
    item_arch_gate_mean: Optional[torch.Tensor] = None
    item_arch_candidate_count: Optional[torch.Tensor] = None


@dataclass
class ItemArchitectureOutput:
    """Outputs from the independent joint-SID item expert."""

    logits: torch.Tensor
    loss: torch.Tensor
    gate_mean: torch.Tensor
    candidate_count: torch.Tensor
    selected_count: torch.Tensor
    top1_accuracy: torch.Tensor


class LLaDARec(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(LLaDARec, self).__init__(config, dataset, tokenizer)

        self.logger = getLogger()

        self.register_buffer(
            "item_id2tokens",
            self._map_item_tokens().to(self.config['device']),
            persistent=False,
        )
        self.register_buffer(
            "valid_item_tuples",
            torch.unique(self.item_id2tokens[1:], dim=0),
            persistent=False,
        )
        self.register_buffer(
            "item_id2catalog_index",
            self._map_item_catalog_indices(),
            persistent=False,
        )
        self.posValidTokens = self.get_posValidToken().to(
            self.config['device'])

        self.mask_token_id = tokenizer.mask_token
        lladaconfig = LLaDAConfig(
            activation_type=config['activation_type'],
            attention_dropout=config['dropout_rate'],
            bias_for_layer_norm=config['bias_for_layer_norm'],
            block_type=config['block_type'],
            d_model=config['d_model'],
            mlp_ratio=config['mlp_ratio'],
            embedding_dropout=config['dropout_rate'],
            init_fn=config['init_fn'],
            layer_norm_type=config['layer_norm_type'],
            max_sequence_length=(config['max_item_seq_len'] + 1) *
            self.tokenizer.n_digit,
            n_heads=config['n_heads'],
            n_kv_heads=config['n_kv_heads'],
            n_layers=config['n_layers'],
            residual_dropout=config['dropout_rate'],
            rope=config['rope'],
            rope_theta=config['rope_theta'],
            weight_tying=config['weight_tying'],
            vocab_size=tokenizer.vocab_size,
            embedding_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            mask_token_id=self.mask_token_id)

        self.llada = LLaDAModelLM(lladaconfig)

        self.item_pos_emb = nn.Embedding(
            num_embeddings=config['max_item_seq_len'] + 1,
            embedding_dim=config['d_model'])
        init_weights(self.llada.model.config,
                     self.item_pos_emb,
                     type_of_module=ModuleType.emb)

        self.use_hierarchy_embedding = bool(
            config.get("use_hierarchy_embedding", False))
        if self.use_hierarchy_embedding:
            self.hierarchy_pos_emb = nn.Embedding(
                num_embeddings=self.tokenizer.n_digit,
                embedding_dim=config["d_model"],
            )
            init_weights(self.llada.model.config,
                         self.hierarchy_pos_emb,
                         type_of_module=ModuleType.emb)

        self.masking_strategy = str(config.get("masking_strategy", "token"))
        valid_masking = {
            "token",
            "item",
            "token_item_mixed",
            "token_suffix_mixed",
        }
        if self.masking_strategy not in valid_masking:
            raise ValueError(
                f"masking_strategy must be one of {sorted(valid_masking)}, "
                f"got {self.masking_strategy!r}")
        self.structured_mask_prob = float(
            config.get("structured_mask_prob", 0.25))
        if not 0.0 <= self.structured_mask_prob <= 1.0:
            raise ValueError("structured_mask_prob must be in [0, 1]")
        self.target_noising_strategy = str(
            config.get("target_noising_strategy", "random"))
        if self.target_noising_strategy not in {
                "random", "on_policy_coherent"
        }:
            raise ValueError(
                "target_noising_strategy must be random or "
                "on_policy_coherent")
        self.ocn_confidence_metric = str(
            config.get("ocn_confidence_metric", "msp"))
        if self.ocn_confidence_metric not in {"msp", "entropy"}:
            raise ValueError(
                "ocn_confidence_metric must be msp or entropy")
        self.ocn_num_views = int(
            config.get("ocn_num_views", tokenizer.n_digit))
        if not 1 <= self.ocn_num_views <= tokenizer.n_digit:
            raise ValueError(
                "ocn_num_views must be between 1 and the SID length")

        self.soft_diffusion_mode = str(
            config.get("soft_diffusion_mode", "off"))
        if self.soft_diffusion_mode not in {
                "off",
                "tuple_simplex",
                "local_tuple_simplex",
                "posterior_tuple_simplex",
                "prefix_posterior_simplex",
                "marginal_simplex",
        }:
            raise ValueError(
                "soft_diffusion_mode must be off, tuple_simplex, "
                "local_tuple_simplex, posterior_tuple_simplex, "
                "prefix_posterior_simplex, or marginal_simplex; got "
                f"{self.soft_diffusion_mode!r}")
        self.soft_diffusion_strength = float(
            config.get("soft_diffusion_strength", 0.35))
        self.soft_diffusion_prob = float(
            config.get("soft_diffusion_prob", 0.5))
        self.soft_diffusion_neighbors = int(
            config.get("soft_diffusion_neighbors", 4))
        self.soft_local_pool = int(config.get("soft_local_pool", 64))
        self.soft_simplex_alpha = float(
            config.get("soft_simplex_alpha", 1.0))
        self.soft_fraction_schedule = bool(
            config.get("soft_fraction_schedule", True))
        self.soft_fraction_power = float(
            config.get("soft_fraction_power", 1.0))
        self.soft_strength_mode = str(
            config.get(
                "soft_strength_mode",
                "fraction" if self.soft_fraction_schedule else "constant",
            ))
        self.soft_gate_hidden = int(config.get("soft_gate_hidden", 16))
        self.soft_posterior_pool = int(
            config.get("soft_posterior_pool", 128))
        self.soft_posterior_temperature = float(
            config.get("soft_posterior_temperature", 0.25))
        self.soft_posterior_hierarchy_decay = float(
            config.get("soft_posterior_hierarchy_decay", 1.0))
        self.soft_view_consistency_w = float(
            config.get("soft_view_consistency_w", 0.0))
        self.soft_view_temperature = float(
            config.get("soft_view_temperature", 1.0))
        self.soft_self_condition_mode = str(
            config.get("soft_self_condition_mode", "off"))
        self.soft_self_condition_strength = float(
            config.get("soft_self_condition_strength", 0.35))
        self.soft_self_condition_topk = int(
            config.get("soft_self_condition_topk", 4))
        self.soft_self_condition_temperature = float(
            config.get("soft_self_condition_temperature", 1.0))
        self.soft_self_condition_aux_w = float(
            config.get("soft_self_condition_aux_w", 0.25))
        self.soft_self_condition_schedule = str(
            config.get("soft_self_condition_schedule", "noise"))
        self.soft_self_condition_power = float(
            config.get("soft_self_condition_power", 1.0))
        # Long-SID diffusion: keep the identifying prefix discrete and place
        # masked suffixes on the exact catalog-posterior simplex induced by
        # that prefix. The clean suffix is never read directly; singleton
        # prefixes can nevertheless identify it, which is why the
        # prefix_soft_min_bucket control can exclude those cases.
        self.soft_prefix_length = int(
            config.get("soft_prefix_length", 3))
        self.soft_suffix_strength_scale = float(
            config.get("soft_suffix_strength_scale", 1.0))
        self.prefix_loss_boost = float(
            config.get("prefix_loss_boost", 1.0))
        self.prefix_loss_mode = str(
            config.get("prefix_loss_mode", "uniform"))
        self.resolved_suffix_loss_weight = float(
            config.get("resolved_suffix_loss_weight", 1.0))
        self.prefix_soft_min_bucket = int(
            config.get("prefix_soft_min_bucket", 1))
        self.prefix_posterior_dropout = float(
            config.get("prefix_posterior_dropout", 0.0))
        self.prefix_resolved_decode = bool(
            config.get("prefix_resolved_decode", False))
        self.prefix_completion_score_weight = float(
            config.get("prefix_completion_score_weight", 1.0))
        self.prefix_soft_decode_strength = float(
            config.get(
                "prefix_soft_decode_strength",
                self.soft_diffusion_strength,
            ))
        self.subset_resolved_decode = bool(
            config.get("subset_resolved_decode", False))
        self.subset_resolved_tokens = int(
            config.get("subset_resolved_tokens", 2))
        self.subset_completion_score_weight = float(
            config.get("subset_completion_score_weight", 1.0))
        self.subset_transfer_policy = str(
            config.get("subset_transfer_policy", "confidence"))
        self.subset_adaptive_bucket_threshold = int(
            config.get("subset_adaptive_bucket_threshold", 0))
        self.subset_catalog_preselection_fusion = bool(
            config.get("subset_catalog_preselection_fusion", False))
        # Evaluation-only timing intervention.  ``inline`` is the trained
        # decoder: item evidence is injected after every denoising forward.
        # ``posthoc`` keeps the same scorer/checkpoint/candidate budget, but
        # never injects it into token logits.  It is applied exactly once to
        # catalog scores after the token trajectory has finished.
        self.item_arch_inference_timing = str(
            config.get("item_arch_inference_timing", "inline"))
        # Evaluation-only mechanism controls. ``static`` freezes the
        # all-mask item query and full-head scoring view, while still using
        # the exact candidate support induced by every evolving partial SID.
        # ``max_score`` replaces posterior-mass projection with support from
        # only the best compatible completion for each token value.
        self.item_arch_inference_evidence = str(
            config.get("item_arch_inference_evidence", "dynamic"))
        self.item_arch_projection_mode = str(
            config.get("item_arch_projection_mode", "posterior_mass"))
        # Evaluation-only gate interventions. ``mean_matched`` preserves the
        # native gate's signed mean on every feedback call while removing all
        # state/position selectivity. ``shuffled`` preserves its full value
        # distribution but breaks alignment with the state. ``off`` is the
        # same-checkpoint no-feedback lower bound.
        self.item_arch_gate_inference_mode = str(
            config.get("item_arch_gate_inference_mode", "native"))
        self._item_arch_static_query = None
        self._item_arch_static_query_initializations = 0
        # Evaluation-only intervention: delay item feedback until a chosen
        # denoising step.  Zero is the trained/default inline decoder.  A
        # value equal to subset_resolved_tokens applies feedback only in the
        # final conditional completion pass.
        self.item_arch_feedback_start_step = int(
            config.get("item_arch_feedback_start_step", 0))
        self.denoising_trace_enabled = bool(
            config.get("denoising_trace_enabled", False))
        self.denoising_trace = []
        self._denoising_trace_batch = 0
        self._logged_subset_completion_stats = False
        self._logged_subset_adaptive_stats = False
        self._logged_subset_preselection_stats = False
        self._logged_subset_transfer_stats = False
        self._item_arch_generation_feedback_calls = 0
        self._item_arch_catalog_rerank_calls = 0
        self._item_arch_gate_audit = {
            "calls": 0,
            "count": 0,
            "native_sum": 0.0,
            "native_abs_sum": 0.0,
            "native_negative_count": 0,
            "used_sum": 0.0,
            "used_abs_sum": 0.0,
            "used_negative_count": 0,
        }
        if not 0.0 <= self.soft_diffusion_strength <= 1.0:
            raise ValueError("soft_diffusion_strength must be in [0, 1]")
        if not 0.0 <= self.soft_diffusion_prob <= 1.0:
            raise ValueError("soft_diffusion_prob must be in [0, 1]")
        if self.soft_diffusion_neighbors < 1:
            raise ValueError("soft_diffusion_neighbors must be positive")
        if self.soft_local_pool < self.soft_diffusion_neighbors:
            raise ValueError(
                "soft_local_pool must be >= soft_diffusion_neighbors")
        if self.soft_simplex_alpha <= 0:
            raise ValueError("soft_simplex_alpha must be positive")
        if self.soft_fraction_power <= 0:
            raise ValueError("soft_fraction_power must be positive")
        valid_strength_modes = {
            "constant",
            "fraction",
            "mid_snr",
            "learned_hierarchy",
        }
        if self.soft_strength_mode not in valid_strength_modes:
            raise ValueError(
                "soft_strength_mode must be one of "
                f"{sorted(valid_strength_modes)}, got "
                f"{self.soft_strength_mode!r}")
        if self.soft_gate_hidden < 1:
            raise ValueError("soft_gate_hidden must be positive")
        if self.soft_posterior_pool < self.soft_diffusion_neighbors:
            raise ValueError(
                "soft_posterior_pool must be >= soft_diffusion_neighbors")
        if self.soft_posterior_temperature <= 0:
            raise ValueError("soft_posterior_temperature must be positive")
        if self.soft_posterior_hierarchy_decay <= 0:
            raise ValueError(
                "soft_posterior_hierarchy_decay must be positive")
        if self.soft_view_consistency_w < 0:
            raise ValueError(
                "soft_view_consistency_w must be non-negative")
        if self.soft_view_temperature <= 0:
            raise ValueError("soft_view_temperature must be positive")
        if (self.soft_view_consistency_w > 0
                and self.soft_diffusion_mode == "off"):
            raise ValueError(
                "soft_view_consistency_w requires a soft diffusion mode")
        if self.soft_self_condition_mode not in {
                "off", "catalog_posterior"
        }:
            raise ValueError(
                "soft_self_condition_mode must be off or catalog_posterior")
        if not 0.0 <= self.soft_self_condition_strength <= 1.0:
            raise ValueError(
                "soft_self_condition_strength must be in [0, 1]")
        if self.soft_self_condition_topk < 1:
            raise ValueError("soft_self_condition_topk must be positive")
        if self.soft_self_condition_temperature <= 0:
            raise ValueError(
                "soft_self_condition_temperature must be positive")
        if not 0.0 <= self.soft_self_condition_aux_w <= 1.0:
            raise ValueError(
                "soft_self_condition_aux_w must be in [0, 1]")
        if self.soft_self_condition_schedule not in {
                "constant", "noise", "complement"
        }:
            raise ValueError(
                "soft_self_condition_schedule must be constant, noise, "
                "or complement")
        if self.soft_self_condition_power <= 0:
            raise ValueError(
                "soft_self_condition_power must be positive")
        if not 1 <= self.soft_prefix_length <= self.tokenizer.n_digit:
            raise ValueError(
                "soft_prefix_length must be in [1, n_digit]")
        if (self.prefix_resolved_decode
                and self.soft_prefix_length >= self.tokenizer.n_digit):
            raise ValueError(
                "prefix-resolved decoding requires prefix_length < n_digit")
        if self.soft_suffix_strength_scale <= 0:
            raise ValueError(
                "soft_suffix_strength_scale must be positive")
        if self.prefix_loss_boost <= 0:
            raise ValueError("prefix_loss_boost must be positive")
        if self.prefix_loss_mode not in {
                "uniform",
                "catalog_info",
                "resolution",
                "catalog_info_resolution",
        }:
            raise ValueError(
                "prefix_loss_mode must be uniform, catalog_info, "
                "resolution, or catalog_info_resolution")
        if not 0.0 <= self.resolved_suffix_loss_weight <= 1.0:
            raise ValueError(
                "resolved_suffix_loss_weight must be in [0, 1]")
        if self.prefix_soft_min_bucket < 1:
            raise ValueError(
                "prefix_soft_min_bucket must be positive")
        if not 0.0 <= self.prefix_posterior_dropout < 1.0:
            raise ValueError(
                "prefix_posterior_dropout must be in [0, 1)")
        if self.prefix_completion_score_weight < 0:
            raise ValueError(
                "prefix_completion_score_weight must be non-negative")
        if not 0.0 <= self.prefix_soft_decode_strength <= 1.0:
            raise ValueError(
                "prefix_soft_decode_strength must be in [0, 1]")
        if (self.subset_resolved_decode
                and not 0 <= self.subset_resolved_tokens
                <= self.tokenizer.n_digit):
            raise ValueError(
                "subset-resolved decoding requires "
                "0 <= subset_resolved_tokens <= n_digit")
        if self.subset_completion_score_weight < 0:
            raise ValueError(
                "subset_completion_score_weight must be non-negative")
        if self.subset_transfer_policy not in {
                "confidence", "catalog_reduction", "sequential"
        }:
            raise ValueError(
                "subset_transfer_policy must be confidence, "
                "catalog_reduction, or sequential")
        if (self.subset_transfer_policy in {"catalog_reduction", "sequential"}
                and not self.subset_resolved_decode):
            raise ValueError(
                "catalog-reduction/sequential transfer requires "
                "subset_resolved_decode")
        if self.subset_adaptive_bucket_threshold < 0:
            raise ValueError(
                "subset_adaptive_bucket_threshold must be non-negative")
        if self.subset_adaptive_bucket_threshold > 0:
            if not self.subset_resolved_decode:
                raise ValueError(
                    "adaptive subset stopping requires "
                    "subset_resolved_decode")
            if self.subset_resolved_tokens != 2:
                raise ValueError(
                    "adaptive subset stopping starts from exactly two "
                    "resolved SID tokens")
            if self.subset_resolved_tokens + 1 > self.tokenizer.n_digit:
                raise ValueError(
                    "adaptive subset stopping requires at least one "
                    "additional SID token")
        if self.subset_catalog_preselection_fusion:
            if not self.subset_resolved_decode:
                raise ValueError(
                    "subset catalog preselection fusion requires "
                    "subset_resolved_decode")
        if self.item_arch_inference_timing not in {"inline", "posthoc"}:
            raise ValueError(
                "item_arch_inference_timing must be inline or posthoc")
        if self.item_arch_feedback_start_step < 0:
            raise ValueError(
                "item_arch_feedback_start_step must be non-negative")
        if self.item_arch_inference_evidence not in {"dynamic", "static"}:
            raise ValueError(
                "item_arch_inference_evidence must be dynamic or static")
        if self.item_arch_projection_mode not in {
                "posterior_mass", "max_score"
        }:
            raise ValueError(
                "item_arch_projection_mode must be posterior_mass or "
                "max_score")
        if self.prefix_resolved_decode and self.subset_resolved_decode:
            raise ValueError(
                "prefix_resolved_decode and subset_resolved_decode "
                "are mutually exclusive")

        self.soft_strength_router = None
        if self.soft_strength_mode == "learned_hierarchy":
            self.soft_strength_router = nn.Sequential(
                nn.Linear(2, self.soft_gate_hidden),
                nn.SiLU(),
                nn.Linear(self.soft_gate_hidden, 1),
            )
            # Start from the already-successful fraction schedule. The router
            # learns only a residual multiplier at initialization.
            nn.init.zeros_(self.soft_strength_router[-1].weight)
            nn.init.zeros_(self.soft_strength_router[-1].bias)

        self.structured_attention_mode = str(
            config.get("structured_attention_mode", "flat"))
        valid_attention = {
            "flat",
            "interleave",
            "interleave_reverse",
            "interleave_bridge",
            "interleave_graph",
            "soft_interleave",
            "soft_interleave_bridge",
            "structure_bias",
            "residual_graph_bias",
            "prq_graph_bias",
            "catalog_graph_bias",
            "catalog_graph_interleave",
            "mask_routed_interleave",
            "mask_routed_interleave_bridge",
            "mask_routed_dual",
            "learned_soft_interleave",
            "learned_soft_interleave_bridge",
            "learned_dual_interleave",
        }
        if self.structured_attention_mode not in valid_attention:
            raise ValueError(
                "structured_attention_mode must be one of "
                f"{sorted(valid_attention)}, got "
                f"{self.structured_attention_mode!r}")
        self.structure_bias_strength = float(
            config.get("structure_bias_strength", 1.0))
        if self.structure_bias_strength < 0:
            raise ValueError("structure_bias_strength must be non-negative")
        self.interleave_local_scale = float(
            config.get("interleave_local_scale", 1.0))
        self.interleave_temporal_scale = float(
            config.get("interleave_temporal_scale", 1.0))
        self.interleave_gate_init = float(
            config.get("interleave_gate_init", 0.05))
        if self.interleave_local_scale < 0:
            raise ValueError(
                "interleave_local_scale must be non-negative")
        if self.interleave_temporal_scale < 0:
            raise ValueError(
                "interleave_temporal_scale must be non-negative")
        self.interleave_layer_gates = None
        if self.structured_attention_mode.startswith("learned_"):
            # Signed residual gates start close to the flat transformer and
            # learn whether each layer benefits from local or temporal bias.
            self.interleave_layer_gates = nn.Parameter(torch.full(
                (int(config["n_layers"]), 2),
                self.interleave_gate_init,
                device=self.config["device"],
                dtype=torch.float32,
            ))
        if self.structured_attention_mode.startswith("catalog_graph"):
            self.register_buffer(
                "catalog_token_graph",
                self._build_catalog_token_graph(),
                persistent=False,
            )
        else:
            self.catalog_token_graph = None

        self.tuple_consistency_w = float(
            config.get("tuple_consistency_w", 0.0))
        if self.tuple_consistency_w < 0:
            raise ValueError("tuple_consistency_w must be non-negative")
        self.tuple_constrained_decode = bool(
            config.get("tuple_constrained_decode", False))

        # Ambiguity-routed dual-granularity denoising.  Unlike the legacy
        # catalog item objective below, this branch has an independent,
        # trainable joint representation for a complete SID tuple.  It is
        # built only when requested, so the default checkpoint/state dict and
        # forward path remain exactly backward compatible.
        self.item_arch_mode = str(config.get("item_arch_mode", "off"))
        self.item_arch_loss_w = float(
            config.get("item_arch_loss_w", 0.0))
        self.item_arch_candidate_topk = int(
            config.get("item_arch_candidate_topk", 128))
        self.item_arch_temperature = float(
            config.get("item_arch_temperature", 0.1))
        self.item_arch_hidden_dim = int(
            config.get("item_arch_hidden_dim", 128))
        self.item_arch_feedback_strength = float(
            config.get("item_arch_feedback_strength", 0.5))
        self.item_arch_decode_weight = float(
            config.get("item_arch_decode_weight", -1.0))
        self.item_arch_feedback_clip = float(
            config.get("item_arch_feedback_clip", 8.0))
        self.item_arch_gate_hidden = int(
            config.get("item_arch_gate_hidden", 32))
        self.item_arch_gate_mode = str(
            config.get("item_arch_gate_mode", "learned"))
        self.item_arch_compatible_only = bool(
            config.get("item_arch_compatible_only", True))
        self.item_arch_score_chunk_size = int(
            config.get("item_arch_score_chunk_size", 64))
        self.item_arch_tuple_mode = str(
            config.get("item_arch_tuple_mode", "flat"))
        self.item_arch_score_mode = str(
            config.get("item_arch_score_mode", "dot"))
        self.item_arch_transition_mode = str(
            config.get("item_arch_transition_mode", "off"))
        self.item_arch_transition_recent_k = int(
            config.get("item_arch_transition_recent_k", 4))
        self.item_arch_transition_attention_temperature = float(
            config.get(
                "item_arch_transition_attention_temperature", 0.25))
        self.item_arch_transition_route_mode = str(
            config.get("item_arch_transition_route_mode", "fixed"))
        self.item_arch_transition_semantic_heads = int(
            config.get("item_arch_transition_semantic_heads", 0))
        self.item_arch_pairwise_semantic_heads = int(
            config.get("item_arch_pairwise_semantic_heads", 0))
        self.item_arch_bigram_encoder_mode = str(
            config.get("item_arch_bigram_encoder_mode", "mlp"))
        self.item_arch_head_gate_init = str(
            config.get("item_arch_head_gate_init", "uniform"))
        # Eval-only validity controls for the transition field. Never used
        # during training (always run with steps=0 against an existing
        # checkpoint): ``reverse_order`` flips which retained position gets
        # the learned recency bias, testing whether the model actually
        # relies on true recency rather than an arbitrary position; ``
        # swap_batch`` gives every example a different example's history
        # (rolled by one), testing whether the mechanism relies on the
        # specific user's own history at all rather than a generic prior.
        self.item_arch_transition_history_ablation = str(
            config.get("item_arch_transition_history_ablation", "off"))
        self.item_arch_loss_min_bucket = int(
            config.get("item_arch_loss_min_bucket", 2))
        self.item_arch_loss_max_bucket = int(
            config.get("item_arch_loss_max_bucket", 0))
        self.item_arch_loss_mode = str(
            config.get("item_arch_loss_mode", "listwise"))
        self.item_arch_rank_topk = int(
            config.get("item_arch_rank_topk", 8))
        self.item_arch_rank_margin = float(
            config.get("item_arch_rank_margin", 0.1))
        self.item_arch_rank_temperature = float(
            config.get("item_arch_rank_temperature", 1.0))
        self.item_arch_rank_weight = float(
            config.get("item_arch_rank_weight", 0.25))
        self.item_arch_rank_overlap_weight = float(
            config.get("item_arch_rank_overlap_weight", 0.0))
        self.item_arch_metric_weight = float(
            config.get("item_arch_metric_weight", 0.25))
        self.item_arch_sid_kernel_w = float(
            config.get("item_arch_sid_kernel_w", 0.0))
        self.item_arch_sid_kernel_temperature = float(
            config.get("item_arch_sid_kernel_temperature", 0.25))
        self.item_arch_sid_kernel_semantic_heads = int(
            config.get("item_arch_sid_kernel_semantic_heads", 0))
        self.item_arch_lattice_mode = str(
            config.get("item_arch_lattice_mode", "off"))
        self.item_arch_lattice_w = float(
            config.get("item_arch_lattice_w", 0.0))
        self.item_arch_lattice_max_order = int(
            config.get("item_arch_lattice_max_order", 3))
        self.item_arch_lattice_max_subsets = int(
            config.get("item_arch_lattice_max_subsets", 32))
        self.item_arch_lattice_min_group = int(
            config.get("item_arch_lattice_min_group", 1))
        self.item_arch_lattice_max_group = int(
            config.get("item_arch_lattice_max_group", 0))
        self.item_arch_lattice_weight_mode = str(
            config.get("item_arch_lattice_weight_mode", "information"))
        self.item_arch_lattice_semantic_heads = int(
            config.get("item_arch_lattice_semantic_heads", 0))
        self.item_arch_belief_mode = str(
            config.get("item_arch_belief_mode", "independent"))
        self.item_arch_belief_weight = float(
            config.get("item_arch_belief_weight", 0.25))
        self.item_arch_feedback_max_bucket = int(
            config.get("item_arch_feedback_max_bucket", 0))
        self.item_arch_history_mode = str(
            config.get("item_arch_history_mode", "off"))
        self._logged_item_arch_candidates = False
        self._logged_item_arch_gate = False
        self._logged_item_arch_rerank = False
        self._logged_item_arch_lattice = False
        self._logged_item_arch_sid_kernel = False
        self._logged_item_arch_empty_fallback = False
        self._logged_item_arch_frontier = False
        self._logged_item_arch_transition = False

        if self.item_arch_mode not in {"off", "joint_energy", "epic"}:
            raise ValueError(
                "item_arch_mode must be off, joint_energy, or epic")
        if self.item_arch_loss_w < 0:
            raise ValueError("item_arch_loss_w must be non-negative")
        if self.item_arch_candidate_topk < 2:
            raise ValueError(
                "item_arch_candidate_topk must be at least 2")
        if self.item_arch_temperature <= 0:
            raise ValueError("item_arch_temperature must be positive")
        if self.item_arch_hidden_dim < 1:
            raise ValueError("item_arch_hidden_dim must be positive")
        if self.item_arch_feedback_strength < 0:
            raise ValueError(
                "item_arch_feedback_strength must be non-negative")
        if self.item_arch_decode_weight < -1:
            raise ValueError(
                "item_arch_decode_weight must be -1 or non-negative")
        if self.item_arch_feedback_clip <= 0:
            raise ValueError("item_arch_feedback_clip must be positive")
        if self.item_arch_gate_hidden < 1:
            raise ValueError("item_arch_gate_hidden must be positive")
        if self.item_arch_gate_mode not in {
                "fixed", "ambiguity", "learned"
        }:
            raise ValueError(
                "item_arch_gate_mode must be fixed, ambiguity, or learned")
        if self.item_arch_gate_inference_mode not in {
                "native", "mean_matched", "shuffled", "off"
        }:
            raise ValueError(
                "item_arch_gate_inference_mode must be native, "
                "mean_matched, shuffled, or off")
        if self.item_arch_score_chunk_size < 1:
            raise ValueError(
                "item_arch_score_chunk_size must be positive")
        if self.item_arch_tuple_mode not in {"flat", "pairwise"}:
            raise ValueError(
                "item_arch_tuple_mode must be flat or pairwise")
        if self.item_arch_score_mode not in {"dot", "head_attention"}:
            raise ValueError(
                "item_arch_score_mode must be dot or head_attention")
        if self.item_arch_transition_mode not in {
                "off",
                "last_pair",
                "recent_attn",
                "headwise_last",
                "recent_head",
                "recent_selfattn",
                "interleave",
                "bigram_attn",
        }:
            raise ValueError(
                "item_arch_transition_mode must be off, last_pair, "
                "recent_attn, headwise_last, recent_head, "
                "recent_selfattn, interleave, or bigram_attn")
        if self.item_arch_head_gate_init not in {"uniform", "infogain"}:
            raise ValueError(
                "item_arch_head_gate_init must be uniform or infogain")
        if self.item_arch_bigram_encoder_mode not in {
                "mlp", "average", "concat"}:
            raise ValueError(
                "item_arch_bigram_encoder_mode must be mlp, average, or "
                "concat")
        if self.item_arch_transition_history_ablation not in {
                "off", "reverse_order", "swap_batch",
                "uniform_attention"}:
            raise ValueError(
                "item_arch_transition_history_ablation must be off, "
                "reverse_order, swap_batch, or uniform_attention")
        if self.item_arch_transition_recent_k < 1:
            raise ValueError(
                "item_arch_transition_recent_k must be positive")
        if self.item_arch_transition_attention_temperature <= 0:
            raise ValueError(
                "item_arch_transition_attention_temperature must be "
                "positive")
        if self.item_arch_transition_route_mode not in {
                "fixed", "ambiguity"
        }:
            raise ValueError(
                "item_arch_transition_route_mode must be fixed or "
                "ambiguity")
        if not 0 <= self.item_arch_transition_semantic_heads <= (
                self.tokenizer.n_digit):
            raise ValueError(
                "item_arch_transition_semantic_heads must be zero or no "
                "larger than the SID length")
        if not 0 <= self.item_arch_pairwise_semantic_heads <= (
                self.tokenizer.n_digit):
            raise ValueError(
                "item_arch_pairwise_semantic_heads must be zero or no "
                "larger than the SID length")
        if self.item_arch_loss_min_bucket < 2:
            raise ValueError(
                "item_arch_loss_min_bucket must be at least 2")
        if (self.item_arch_loss_max_bucket > 0
                and self.item_arch_loss_max_bucket
                < self.item_arch_loss_min_bucket):
            raise ValueError(
                "item_arch_loss_max_bucket must be zero or no smaller "
                "than item_arch_loss_min_bucket")
        if self.item_arch_loss_mode not in {
                "listwise", "top_rank", "soft_ndcg5"
        }:
            raise ValueError(
                "item_arch_loss_mode must be listwise, top_rank, or "
                "soft_ndcg5")
        if self.item_arch_rank_topk < 1:
            raise ValueError("item_arch_rank_topk must be positive")
        if self.item_arch_rank_margin < 0:
            raise ValueError("item_arch_rank_margin must be non-negative")
        if self.item_arch_rank_temperature <= 0:
            raise ValueError(
                "item_arch_rank_temperature must be positive")
        if self.item_arch_rank_weight < 0:
            raise ValueError("item_arch_rank_weight must be non-negative")
        if self.item_arch_rank_overlap_weight < 0:
            raise ValueError(
                "item_arch_rank_overlap_weight must be non-negative")
        if self.item_arch_metric_weight < 0:
            raise ValueError("item_arch_metric_weight must be non-negative")
        if self.item_arch_sid_kernel_w < 0:
            raise ValueError(
                "item_arch_sid_kernel_w must be non-negative")
        if self.item_arch_sid_kernel_temperature <= 0:
            raise ValueError(
                "item_arch_sid_kernel_temperature must be positive")
        if not 0 <= self.item_arch_sid_kernel_semantic_heads <= (
                self.tokenizer.n_digit):
            raise ValueError(
                "item_arch_sid_kernel_semantic_heads must be zero or no "
                "larger than the SID length")
        if self.item_arch_lattice_mode not in {
                "off", "arbitrary", "prefix"
        }:
            raise ValueError(
                "item_arch_lattice_mode must be off, arbitrary, or prefix")
        if self.item_arch_lattice_w < 0:
            raise ValueError("item_arch_lattice_w must be non-negative")
        if self.item_arch_lattice_max_order < 2:
            raise ValueError(
                "item_arch_lattice_max_order must be at least 2")
        if self.item_arch_lattice_max_subsets < 1:
            raise ValueError(
                "item_arch_lattice_max_subsets must be positive")
        if self.item_arch_lattice_min_group < 1:
            raise ValueError(
                "item_arch_lattice_min_group must be positive")
        if (self.item_arch_lattice_max_group > 0
                and self.item_arch_lattice_max_group
                < self.item_arch_lattice_min_group):
            raise ValueError(
                "item_arch_lattice_max_group must be zero or no smaller "
                "than item_arch_lattice_min_group")
        if self.item_arch_lattice_weight_mode not in {
                "information", "frontier"
        }:
            raise ValueError(
                "item_arch_lattice_weight_mode must be information or "
                "frontier")
        if not 0 <= self.item_arch_lattice_semantic_heads <= (
                self.tokenizer.n_digit):
            raise ValueError(
                "item_arch_lattice_semantic_heads must be zero or no "
                "larger than the SID length")
        if self.item_arch_belief_mode not in {
                "independent", "poe", "cavity"
        }:
            raise ValueError(
                "item_arch_belief_mode must be independent, poe, or cavity")
        if self.item_arch_belief_weight < 0:
            raise ValueError(
                "item_arch_belief_weight must be non-negative")
        if self.item_arch_feedback_max_bucket < 0:
            raise ValueError(
                "item_arch_feedback_max_bucket must be non-negative")
        if (self.item_arch_feedback_max_bucket > 0
                and self.item_arch_feedback_max_bucket
                > self.item_arch_candidate_topk):
            raise ValueError(
                "item_arch_feedback_max_bucket must not exceed "
                "item_arch_candidate_topk")
        if self.item_arch_history_mode not in {
                "off", "gru", "mean", "attention"
        }:
            raise ValueError(
                "item_arch_history_mode must be off, gru, mean, or "
                "attention")

        self.item_arch_digit_emb = None
        self.item_arch_tuple_encoder = None
        self.item_arch_query_encoder = None
        self.item_arch_gate = None
        self.item_arch_residual_alpha = None
        self.item_arch_history_cell = None
        self.item_arch_history_alpha = None
        self.item_arch_pair_proj = None
        self.item_arch_pair_out = None
        self.item_arch_pair_alpha = None
        self.item_arch_head_proj = None
        self.item_arch_head_gate = None
        self.item_arch_head_alpha = None
        self.item_arch_transition_mlp = None
        self.item_arch_transition_alpha = None
        self.item_arch_transition_recency_bias = None
        self.item_arch_transition_head_proj = None
        self.item_arch_transition_head_mlp = None
        self.item_arch_transition_head_gate = None
        self.item_arch_transition_head_alpha = None
        self.item_arch_recent_selfattn_q = None
        self.item_arch_recent_selfattn_k = None
        self.item_arch_recent_selfattn_v = None
        self.item_arch_recent_selfattn_alpha = None
        self.item_arch_interleave_qkv = None
        self.item_arch_interleave_cross_q = None
        self.item_arch_interleave_cross_k = None
        self.item_arch_interleave_mlp = None
        self.item_arch_interleave_head_gate = None
        self.item_arch_interleave_alpha = None
        self.item_arch_bigram_encoder = None
        self.item_arch_bigram_mlp = None
        self.item_arch_bigram_recency_bias = None
        self.item_arch_bigram_alpha = None
        if self.item_arch_mode != "off":
            h = self.tokenizer.n_digit
            d_model = int(config["d_model"])
            hidden = self.item_arch_hidden_dim
            self.item_arch_digit_emb = nn.Embedding(h, d_model)
            self.item_arch_tuple_encoder = nn.Sequential(
                nn.LayerNorm(h * d_model),
                nn.Linear(h * d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.item_arch_query_encoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.item_arch_gate = nn.Sequential(
                nn.Linear(d_model + 5, self.item_arch_gate_hidden),
                nn.SiLU(),
                nn.Linear(self.item_arch_gate_hidden, 1),
            )
            if self.item_arch_tuple_mode == "pairwise":
                self.item_arch_pair_proj = nn.Linear(
                    d_model, hidden, bias=False)
                self.item_arch_pair_out = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                )
                self.item_arch_pair_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_score_mode == "head_attention":
                self.item_arch_head_proj = nn.Linear(
                    d_model, hidden, bias=False)
                self.item_arch_head_gate = nn.Linear(
                    hidden, h)
                self.item_arch_head_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_transition_mode in {
                    "last_pair", "recent_attn", "recent_head",
                    "recent_selfattn",
            }:
                self.item_arch_transition_mlp = nn.Sequential(
                    nn.LayerNorm(2 * hidden),
                    nn.Linear(2 * hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                self.item_arch_transition_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_transition_mode in {
                    "recent_attn", "recent_head", "recent_selfattn",
            }:
                self.item_arch_transition_recency_bias = nn.Parameter(
                    torch.zeros(self.item_arch_transition_recent_k))
            if self.item_arch_transition_mode == "recent_selfattn":
                # Let the K retained recent items attend to each other
                # before being scored against the candidate, modeling them
                # as a sequence rather than an independent set.
                self.item_arch_recent_selfattn_q = nn.Linear(
                    hidden, hidden, bias=False)
                self.item_arch_recent_selfattn_k = nn.Linear(
                    hidden, hidden, bias=False)
                self.item_arch_recent_selfattn_v = nn.Linear(
                    hidden, hidden, bias=False)
                self.item_arch_recent_selfattn_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_transition_mode == "interleave":
                # Sequential intra-item (within-SID-tuple) mixing followed
                # by inter-item (same-head, across recent history) mixing,
                # instead of the usual independent parallel residual terms.
                self.item_arch_interleave_qkv = nn.Linear(
                    d_model, 3 * hidden, bias=False)
                self.item_arch_interleave_cross_q = nn.Linear(
                    hidden, hidden, bias=False)
                self.item_arch_interleave_cross_k = nn.Linear(
                    d_model, hidden, bias=False)
                self.item_arch_interleave_mlp = nn.Sequential(
                    nn.LayerNorm(2 * hidden),
                    nn.Linear(2 * hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                self.item_arch_interleave_head_gate = nn.Linear(
                    hidden, h)
                self.item_arch_interleave_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_transition_mode == "bigram_attn":
                # Second-order transition structure: encode each
                # consecutive (older, newer) pair among the retained recent
                # items as one whole-item edge key, then score the candidate
                # against those edges instead of single items.
                # ``mlp`` (original): a freshly-learned encoder maps the
                # product/abs-difference pair feature into a NEW embedding
                # space that must be learned from scratch alongside the
                # comparison against it. ``average`` instead keeps the edge
                # key in the exact same normalized item-key space recent_attn
                # already uses (no new encoder), testing whether bigram_attn's
                # underperformance came from that foreign representation
                # space rather than from second-order structure itself.
                if self.item_arch_bigram_encoder_mode in {"mlp", "concat"}:
                    # ``mlp`` feeds a symmetric product/abs-difference
                    # feature (order-invariant: swapping newer/older gives
                    # the same input). ``concat`` instead feeds
                    # [newer, older] directly, so the encoder can learn a
                    # genuinely order-aware ("A then B" != "B then A") edge
                    # representation -- both bg00-06 encodings collapse
                    # order information entirely.
                    self.item_arch_bigram_encoder = nn.Sequential(
                        nn.LayerNorm(2 * hidden),
                        nn.Linear(2 * hidden, hidden),
                        nn.SiLU(),
                        nn.Linear(hidden, hidden),
                    )
                self.item_arch_bigram_mlp = nn.Sequential(
                    nn.LayerNorm(2 * hidden),
                    nn.Linear(2 * hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                self.item_arch_bigram_recency_bias = nn.Parameter(
                    torch.zeros(max(1, self.item_arch_transition_recent_k - 1)))
                self.item_arch_bigram_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_transition_mode in {
                    "headwise_last", "recent_head"
            }:
                self.item_arch_transition_head_proj = nn.Linear(
                    d_model, hidden, bias=False)
                self.item_arch_transition_head_mlp = nn.Sequential(
                    nn.LayerNorm(2 * hidden),
                    nn.Linear(2 * hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                self.item_arch_transition_head_gate = nn.Linear(
                    hidden, h)
                self.item_arch_transition_head_alpha = nn.Parameter(
                    torch.zeros(()))
            if self.item_arch_history_mode == "gru":
                self.item_arch_history_cell = nn.GRUCell(hidden, hidden)
            if self.item_arch_history_mode != "off":
                self.item_arch_history_alpha = nn.Parameter(
                    torch.zeros(()))
            # Fixed/heuristic gates use this zero-initialized scalar so their
            # decoder is also an exact warm-start no-op. Learned gates instead
            # get the same property from their zero-initialized final layer.
            self.item_arch_residual_alpha = nn.Parameter(torch.zeros(()))
            nn.init.normal_(
                self.item_arch_digit_emb.weight, mean=0.0, std=0.02)
            item_modules = [
                self.item_arch_tuple_encoder,
                self.item_arch_query_encoder,
                self.item_arch_gate,
                self.item_arch_pair_proj,
                self.item_arch_pair_out,
                self.item_arch_head_proj,
                self.item_arch_head_gate,
                self.item_arch_transition_mlp,
                self.item_arch_transition_head_proj,
                self.item_arch_transition_head_mlp,
                self.item_arch_transition_head_gate,
                self.item_arch_recent_selfattn_q,
                self.item_arch_recent_selfattn_k,
                self.item_arch_recent_selfattn_v,
                self.item_arch_interleave_qkv,
                self.item_arch_interleave_cross_q,
                self.item_arch_interleave_cross_k,
                self.item_arch_interleave_mlp,
                self.item_arch_interleave_head_gate,
                self.item_arch_bigram_encoder,
                self.item_arch_bigram_mlp,
            ]
            for module in (
                    value for value in item_modules
                    if value is not None):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            # The learned EPIC branch is exactly the token-only model at
            # initialization while retaining a nonzero gradient through tanh.
            nn.init.zeros_(self.item_arch_gate[-1].weight)
            nn.init.zeros_(self.item_arch_gate[-1].bias)
            # Structured-SID-generation-aware prior: initialize the
            # head-selection gate bias from each SID head's empirical
            # marginal entropy over the actual catalog, instead of a
            # uniform zero bias, so head selection starts already aware of
            # which heads the tokenizer made more discriminative.
            if self.item_arch_head_gate_init == "infogain":
                head_gain = self._item_arch_head_information_gain()
                for gate in (
                        self.item_arch_head_gate,
                        self.item_arch_transition_head_gate,
                ):
                    if gate is not None:
                        with torch.no_grad():
                            gate.bias.copy_(
                                head_gain[:gate.out_features].to(
                                    gate.bias.dtype))
            if self.item_arch_history_cell is not None:
                for name, parameter in (
                        self.item_arch_history_cell.named_parameters()):
                    if "weight" in name:
                        nn.init.xavier_uniform_(parameter)
                    else:
                        nn.init.zeros_(parameter)

        self.catalog_item_loss_w = float(
            config.get("catalog_item_loss_w", 0.0))
        self.catalog_item_loss_mode = str(
            config.get("catalog_item_loss_mode", "full"))
        self.catalog_item_loss_topk = int(
            config.get("catalog_item_loss_topk", 128))
        self.catalog_item_loss_temperature = float(
            config.get("catalog_item_loss_temperature", 1.0))
        self.catalog_item_loss_min_masked = int(
            config.get("catalog_item_loss_min_masked", 1))
        self.catalog_item_loss_max_masked = int(
            config.get("catalog_item_loss_max_masked", 0))
        self.catalog_item_loss_focal_gamma = float(
            config.get("catalog_item_loss_focal_gamma", 0.0))
        self._logged_catalog_item_stats = False
        self.catalog_decode_mode = str(
            config.get("catalog_decode_mode", "beam"))
        self.catalog_decode_weight = float(
            config.get("catalog_decode_weight", 0.5))
        self.catalog_decode_digit_weight = str(
            config.get("catalog_decode_digit_weight", "uniform"))
        self.catalog_score_normalization = str(
            config.get("catalog_score_normalization", "none"))
        self.catalog_decode_weight_mode = str(
            config.get("catalog_decode_weight_mode", "fixed"))
        self.catalog_decode_weight_min = float(
            config.get("catalog_decode_weight_min",
                       self.catalog_decode_weight))
        self.catalog_decode_weight_max = float(
            config.get("catalog_decode_weight_max",
                       self.catalog_decode_weight))
        self.catalog_decode_confidence_threshold = float(
            config.get("catalog_decode_confidence_threshold", 0.25))
        self.catalog_decode_confidence_temperature = float(
            config.get("catalog_decode_confidence_temperature", 0.08))
        self.catalog_decode_conflict_threshold = float(
            config.get("catalog_decode_conflict_threshold", 0.05))
        self.catalog_decode_conflict_temperature = float(
            config.get("catalog_decode_conflict_temperature", 0.05))
        self._logged_catalog_query_weight = False
        if self.catalog_item_loss_w < 0:
            raise ValueError("catalog_item_loss_w must be non-negative")
        if self.catalog_item_loss_mode not in {
                "full",
                "hard",
                "inbatch",
                "conditional",
                "conditional_hard",
                "stage_adaptive",
        }:
            raise ValueError(
                "catalog_item_loss_mode must be full, hard, inbatch, "
                "conditional, conditional_hard, or stage_adaptive")
        if self.catalog_item_loss_topk < 2:
            raise ValueError("catalog_item_loss_topk must be at least 2")
        if self.catalog_item_loss_temperature <= 0:
            raise ValueError(
                "catalog_item_loss_temperature must be positive")
        if self.catalog_item_loss_min_masked < 1:
            raise ValueError(
                "catalog_item_loss_min_masked must be positive")
        if (self.catalog_item_loss_max_masked > 0
                and self.catalog_item_loss_max_masked
                < self.catalog_item_loss_min_masked):
            raise ValueError(
                "catalog_item_loss_max_masked must be zero or >= min")
        if self.catalog_item_loss_focal_gamma < 0:
            raise ValueError(
                "catalog_item_loss_focal_gamma must be non-negative")
        if self.catalog_decode_mode not in {
                "beam",
                "parallel",
                "hybrid",
                "guided",
                "guided_hybrid",
        }:
            raise ValueError(
                "catalog_decode_mode must be beam, parallel, hybrid, "
                "guided, or guided_hybrid")
        if self.catalog_decode_weight < 0:
            raise ValueError(
                "catalog_decode_weight must be non-negative")
        if self.catalog_decode_digit_weight not in {
                "uniform",
                "catalog_info",
                "confidence",
                "info_confidence",
        }:
            raise ValueError(
                "catalog_decode_digit_weight must be uniform, catalog_info, "
                "confidence, or info_confidence")
        if self.catalog_score_normalization not in {
                "none", "zscore", "rank"
        }:
            raise ValueError(
                "catalog_score_normalization must be none, zscore, or rank")
        if self.catalog_decode_weight_mode not in {
                "fixed",
                "digit_confidence",
                "candidate_entropy",
                "candidate_margin",
                "disagreement_confidence",
                "selective_disagreement",
                "conflict_confidence",
        }:
            raise ValueError(
                "catalog_decode_weight_mode must be fixed, "
                "digit_confidence, candidate_entropy, candidate_margin, "
                "disagreement_confidence, selective_disagreement, "
                "or conflict_confidence")
        if (self.catalog_decode_weight_min < 0
                or self.catalog_decode_weight_max
                < self.catalog_decode_weight_min):
            raise ValueError(
                "catalog decode adaptive weights require "
                "0 <= min <= max")
        if (self.catalog_decode_weight_mode != "fixed"
                and self.catalog_decode_mode
                not in {"hybrid", "guided_hybrid"}):
            raise ValueError(
                "adaptive catalog decode weights require hybrid decoding")
        if (self.subset_resolved_decode
                and self.catalog_decode_mode not in {"beam", "hybrid"}):
            raise ValueError(
                "subset-resolved decoding supports catalog_decode_mode "
                "beam or hybrid")
        if (self.subset_catalog_preselection_fusion
                and self.catalog_decode_mode != "hybrid"):
            raise ValueError(
                "subset catalog preselection fusion requires "
                "catalog_decode_mode=hybrid")
        if (self.catalog_decode_confidence_temperature <= 0
                or self.catalog_decode_conflict_temperature <= 0):
            raise ValueError(
                "adaptive catalog temperatures must be positive")
        self._build_prefix_catalog_index()
        self._build_prefix_loss_statistics()

        self.temperature = self.config['temperature']
        self.log("scale logits by temperature: {}".format(self.temperature))
        self.log(
            "structured variant: attention={} masking={} p_struct={} "
            "target_noising={} ocn_metric={} ocn_views={} "
            "soft={} soft_strength={} soft_prob={} soft_k={} "
            "soft_pool={} soft_alpha={} soft_power={} strength_mode={} "
            "posterior_pool={} posterior_temp={} posterior_decay={} "
            "view_w={} self_condition={} self_strength={} self_topk={} "
            "prefix_len={} suffix_scale={} prefix_loss={} "
            "prefix_loss_mode={} resolved_suffix_w={} "
            "prefix_min_bucket={} prefix_dropout={} "
            "prefix_decode={} prefix_decode_strength={} "
            "subset_decode={} subset_tokens={} subset_score_w={} "
            "subset_transfer_policy={} subset_adaptive_threshold={} "
            "subset_preselect_fusion={} "
            "tuple_loss_w={} tuple_decode={} catalog_item_w={} "
            "catalog_item_mode={} catalog_decode={} "
            "catalog_decode_w={} catalog_digit_w={} score_norm={} "
            "catalog_weight_mode={} catalog_weight_range={}/{} "
            "item_arch={} item_arch_w={} item_arch_k={} item_arch_dim={} "
            "item_arch_temp={} item_arch_gate={} item_arch_feedback={} "
            "item_arch_decode_w={} item_arch_compatible={} "
            "item_arch_tuple={} item_arch_pair_heads={} item_arch_score={} "
            "item_arch_loss_mode={} item_arch_sid_kernel={}/{}/{} "
            "item_arch_lattice={}/{} "
            "item_arch_belief={}/{} item_arch_bucket_cap={} "
            "item_arch_history={} "
            "interleave_scales={}/{} gate_init={}".format(
                self.structured_attention_mode,
                self.masking_strategy,
                self.structured_mask_prob,
                self.target_noising_strategy,
                self.ocn_confidence_metric,
                self.ocn_num_views,
                self.soft_diffusion_mode,
                self.soft_diffusion_strength,
                self.soft_diffusion_prob,
                self.soft_diffusion_neighbors,
                self.soft_local_pool,
                self.soft_simplex_alpha,
                self.soft_fraction_power,
                self.soft_strength_mode,
                self.soft_posterior_pool,
                self.soft_posterior_temperature,
                self.soft_posterior_hierarchy_decay,
                self.soft_view_consistency_w,
                self.soft_self_condition_mode,
                self.soft_self_condition_strength,
                self.soft_self_condition_topk,
                self.soft_prefix_length,
                self.soft_suffix_strength_scale,
                self.prefix_loss_boost,
                self.prefix_loss_mode,
                self.resolved_suffix_loss_weight,
                self.prefix_soft_min_bucket,
                self.prefix_posterior_dropout,
                self.prefix_resolved_decode,
                self.prefix_soft_decode_strength,
                self.subset_resolved_decode,
                self.subset_resolved_tokens,
                self.subset_completion_score_weight,
                self.subset_transfer_policy,
                self.subset_adaptive_bucket_threshold,
                self.subset_catalog_preselection_fusion,
                self.tuple_consistency_w,
                self.tuple_constrained_decode,
                self.catalog_item_loss_w,
                self.catalog_item_loss_mode,
                self.catalog_decode_mode,
                self.catalog_decode_weight,
                self.catalog_decode_digit_weight,
                self.catalog_score_normalization,
                self.catalog_decode_weight_mode,
                self.catalog_decode_weight_min,
                self.catalog_decode_weight_max,
                self.item_arch_mode,
                self.item_arch_loss_w,
                self.item_arch_candidate_topk,
                self.item_arch_hidden_dim,
                self.item_arch_temperature,
                self.item_arch_gate_mode,
                self.item_arch_feedback_strength,
                self.item_arch_decode_weight,
                self.item_arch_compatible_only,
                self.item_arch_tuple_mode,
                self.item_arch_pairwise_semantic_heads,
                self.item_arch_score_mode,
                self.item_arch_loss_mode,
                self.item_arch_sid_kernel_w,
                self.item_arch_sid_kernel_temperature,
                self.item_arch_sid_kernel_semantic_heads,
                self.item_arch_lattice_mode,
                self.item_arch_lattice_w,
                self.item_arch_belief_mode,
                self.item_arch_belief_weight,
                self.item_arch_feedback_max_bucket,
                self.item_arch_history_mode,
                self.interleave_local_scale,
                self.interleave_temporal_scale,
                self.interleave_gate_init,
            ))
        self.log(
            "item structural frontier: transition={} recent_k={} "
            "transition_temp={} transition_route={} transition_heads={} "
            "loss_bucket={}/{} lattice_group={}/{} lattice_weight={} "
            "lattice_heads={}".format(
                self.item_arch_transition_mode,
                self.item_arch_transition_recent_k,
                self.item_arch_transition_attention_temperature,
                self.item_arch_transition_route_mode,
                self.item_arch_transition_semantic_heads,
                self.item_arch_loss_min_bucket,
                self.item_arch_loss_max_bucket,
                self.item_arch_lattice_min_group,
                self.item_arch_lattice_max_group,
                self.item_arch_lattice_weight_mode,
                self.item_arch_lattice_semantic_heads,
            ))

        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label, reduction='none')
        self.his_mask_w = self.config['his_mask_w']

        self.gen_steps = config['gen_steps']

        self.val_num_beams = config['val_num_beams']
        self.log(f"using num beams: {self.val_num_beams} for validation")
        self.num_beams = config['num_beams']

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(
                self.tokenizer.item2tokens[item])
        return item_id2tokens

    def _map_item_catalog_indices(self) -> torch.Tensor:
        """Map dataset item IDs to rows in the unique catalog SID table."""
        catalog = {
            tuple(tokens): index
            for index, tokens in enumerate(
                self.valid_item_tuples.detach().cpu().tolist())
        }
        indices = torch.full(
            (self.item_id2tokens.shape[0], ),
            -1,
            device=self.item_id2tokens.device,
            dtype=torch.long,
        )
        for item_id, tokens in enumerate(
                self.item_id2tokens.detach().cpu().tolist()):
            if item_id == 0:
                continue
            indices[item_id] = catalog[tuple(tokens)]
        return indices

    def get_posValidToken(self) -> torch.Tensor:
        posValidTokens = {}

        for item in self.tokenizer.item2tokens:
            cur_tokens = self.tokenizer.item2tokens[item]
            for pos in range(len(cur_tokens)):
                if pos not in posValidTokens.keys():
                    posValidTokens[pos] = set()
                posValidTokens[pos].add(cur_tokens[pos])
        max_token_num = 0

        for pos in posValidTokens:
            posValidTokens[pos] = sorted(list(posValidTokens[pos]))
            max_token_num = max(max_token_num, len(posValidTokens[pos]))

        posValidTokens_pt = torch.zeros((len(posValidTokens), max_token_num),
                                        dtype=torch.long)

        for pos in posValidTokens:
            posValidTokens_pt[
                pos, :len(posValidTokens[pos])] = torch.LongTensor(
                    posValidTokens[pos])

        return posValidTokens_pt

    def _build_prefix_catalog_index(self) -> None:
        """Build a compact exact prefix-to-catalog bucket index on device."""
        prefix_len = self.soft_prefix_length
        base = self.tokenizer.vocab_size + 1
        powers = torch.tensor(
            [base ** power for power in range(prefix_len - 1, -1, -1)],
            device=self.valid_item_tuples.device,
            dtype=torch.long,
        )
        codes = (
            self.valid_item_tuples[:, :prefix_len] * powers[None, :]
        ).sum(dim=-1)
        sorted_codes, order = torch.sort(codes)
        unique_codes, counts = torch.unique_consecutive(
            sorted_codes, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        bucket_ids = torch.repeat_interleave(
            torch.arange(
                counts.numel(),
                device=counts.device,
                dtype=torch.long,
            ),
            counts,
        )
        positions = (
            torch.arange(
                order.numel(),
                device=order.device,
                dtype=torch.long,
            )
            - torch.repeat_interleave(starts, counts)
        )
        members = torch.full(
            (counts.numel(), int(counts.max().item())),
            -1,
            device=order.device,
            dtype=torch.long,
        )
        members[bucket_ids, positions] = order
        self.register_buffer(
            "prefix_code_powers", powers, persistent=False)
        self.register_buffer(
            "prefix_bucket_codes", unique_codes, persistent=False)
        self.register_buffer(
            "prefix_bucket_counts", counts, persistent=False)
        self.register_buffer(
            "prefix_bucket_members", members, persistent=False)

    def _build_prefix_loss_statistics(self) -> None:
        """Measure the catalog information revealed by each SID position."""
        tuples = self.valid_item_tuples.detach().cpu()
        n_items = max(1, tuples.shape[0])
        previous_entropy = 0.0
        information_gain = []
        for length in range(1, tuples.shape[1] + 1):
            _, counts = torch.unique(
                tuples[:, :length], dim=0, return_counts=True)
            probability = counts.to(torch.float64) / n_items
            entropy = float(
                -(probability * probability.clamp_min(1e-12).log()).sum())
            information_gain.append(max(0.0, entropy - previous_entropy))
            previous_entropy = entropy
        gains = torch.tensor(
            information_gain,
            device=self.valid_item_tuples.device,
            dtype=torch.float32,
        )
        gains = gains / gains.mean().clamp_min(1e-6)
        self.register_buffer(
            "catalog_information_gain", gains, persistent=False)

    def _lookup_prefix_buckets(
        self,
        tuples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return bucket row and validity for rows with a visible prefix."""
        prefix = tuples[:, :self.soft_prefix_length]
        visible = (
            prefix.ne(self.mask_token_id)
            & prefix.ne(0)
        ).all(dim=-1)
        codes = (
            prefix * self.prefix_code_powers[None, :]
        ).sum(dim=-1)
        rows = torch.searchsorted(self.prefix_bucket_codes, codes)
        safe_rows = rows.clamp_max(self.prefix_bucket_codes.numel() - 1)
        found = (
            visible
            & rows.lt(self.prefix_bucket_codes.numel())
            & self.prefix_bucket_codes[safe_rows].eq(codes)
        )
        return safe_rows, found

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.llada.get_input_embeddings().parameters()
            if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def add_item_pos_emb(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        item_seq_len = input_ids.shape[1] // self.tokenizer.n_digit
        item_pos_idx = torch.arange(item_seq_len, device=input_ids.device)

        token_pos_idx = item_pos_idx.repeat_interleave(
            repeats=self.tokenizer.n_digit, dim=0)

        token_pos_emb = self.item_pos_emb(token_pos_idx)
        token_pos_emb = token_pos_emb.unsqueeze(0).expand(
            (input_ids.shape[0], -1, -1))

        if input_embeds is None:
            input_embeds = self.llada.model.transformer.wte(input_ids)

        output = input_embeds + token_pos_emb
        if self.use_hierarchy_embedding:
            hierarchy_idx = torch.arange(
                self.tokenizer.n_digit,
                device=input_ids.device,
            ).repeat(item_seq_len)
            output = output + self.hierarchy_pos_emb(hierarchy_idx)[None, :, :]
        return output

    def _structured_attention_bias(
        self,
        sequence_length: int,
        device: torch.device,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Build per-layer attention geometry over the items x SID grid.

        ``interleave`` alternates within-item and same-hierarchy temporal
        message passing. ``interleave_graph`` is the sparse-chain control:
        it deliberately assumes a serial residual topology and is therefore
        expected to be a poor inductive bias for parallel VQ heads.
        """
        mode = self.structured_attention_mode
        if mode == "flat":
            return None

        h = self.tokenizer.n_digit
        positions = torch.arange(sequence_length, device=device)
        item = torch.div(positions, h, rounding_mode="floor")
        hierarchy = positions.remainder(h)
        same_item = item[:, None].eq(item[None, :])
        same_hierarchy = hierarchy[:, None].eq(hierarchy[None, :])
        local_dense = same_item
        local_graph = same_item & (
            (hierarchy[:, None] - hierarchy[None, :]).abs() <= 1)
        temporal = same_hierarchy
        global_mask = torch.ones_like(local_dense)

        def shaped(mask: torch.Tensor) -> torch.Tensor:
            return mask[None, None, :, :]

        def additive(mask: torch.Tensor) -> torch.Tensor:
            return shaped(
                mask.to(torch.float32) * self.structure_bias_strength)

        local_additive = (
            additive(local_dense) * self.interleave_local_scale)
        temporal_additive = (
            additive(temporal) * self.interleave_temporal_scale)

        if mode == "structure_bias":
            bias = self.structure_bias_strength * (
                same_item.to(torch.float32)
                + same_hierarchy.to(torch.float32))
            return bias[None, None, :, :]

        if mode == "residual_graph_bias":
            return additive(local_graph) + additive(temporal)

        if mode == "prq_graph_bias":
            if h != 17:
                raise ValueError(
                    "prq_graph_bias requires PRQ P4xK4 plus CID (17 digits), "
                    f"got {h}")
            semantic = hierarchy.lt(16)
            branch = torch.div(hierarchy.clamp_max(15), 4,
                               rounding_mode="floor")
            stage = hierarchy.remainder(4)
            branch_edge = (
                same_item
                & semantic[:, None]
                & semantic[None, :]
                & branch[:, None].eq(branch[None, :])
                & (stage[:, None] - stage[None, :]).abs().le(1)
            )
            root_edge = (
                same_item
                & semantic[:, None]
                & semantic[None, :]
                & stage[:, None].eq(0)
                & stage[None, :].eq(0)
            )
            # The CID query may read every semantic node. Semantic nodes do
            # not receive an extra edge back from CID, avoiding an ordinal
            # collision-ID shortcut.
            cid_edge = same_item & hierarchy[:, None].eq(16)
            return additive(
                branch_edge | root_edge | cid_edge) + additive(temporal)

        catalog_bias = None
        if mode.startswith("catalog_graph"):
            if input_ids is None or self.catalog_token_graph is None:
                raise ValueError(
                    f"{mode} requires input IDs and a catalog token graph")
            catalog_bias = self.catalog_token_graph[
                input_ids[:, :, None], input_ids[:, None, :]]
            catalog_bias = (
                self.structure_bias_strength
                * catalog_bias[:, None, :, :])
            if mode == "catalog_graph_bias":
                return catalog_bias

        n_layers = int(self.config["n_layers"])
        if mode == "interleave":
            schedule = [local_dense, temporal]
        elif mode == "interleave_reverse":
            schedule = [temporal, local_dense]
        elif mode == "interleave_bridge":
            schedule = [local_dense, temporal, global_mask, global_mask]
        elif mode == "interleave_graph":
            schedule = [local_graph, temporal]
        elif mode == "soft_interleave":
            soft_schedule = [local_additive, temporal_additive]
            return [
                soft_schedule[i % len(soft_schedule)]
                for i in range(n_layers)
            ]
        elif mode == "soft_interleave_bridge":
            zero = torch.zeros(
                (1, 1, sequence_length, sequence_length),
                device=device,
                dtype=torch.float32,
            )
            soft_schedule = [
                local_additive,
                temporal_additive,
                zero,
                zero,
            ]
            return [
                soft_schedule[i % len(soft_schedule)]
                for i in range(n_layers)
            ]
        elif mode.startswith("mask_routed_"):
            if input_ids is None:
                raise ValueError(
                    f"{mode} requires input IDs for corruption routing")
            # A masked query needs evidence from the same semantic head over
            # time. Once resolved, it receives within-item product-code
            # context. Zero residual bias still leaves global attention
            # available, unlike the earlier hard axial masks.
            mask_query = input_ids.eq(
                self.mask_token_id).to(torch.float32)[:, None, :, None]
            visible_query = 1.0 - mask_query
            routed_local = local_additive * visible_query
            routed_temporal = temporal_additive * mask_query
            if mode == "mask_routed_dual":
                routed = routed_local + routed_temporal
                return [routed for _ in range(n_layers)]
            if mode == "mask_routed_interleave":
                routed_schedule = [routed_local, routed_temporal]
            else:
                zero = torch.zeros(
                    (1, 1, sequence_length, sequence_length),
                    device=device,
                    dtype=torch.float32,
                )
                routed_schedule = [
                    routed_local,
                    routed_temporal,
                    zero,
                    zero,
                ]
            return [
                routed_schedule[i % len(routed_schedule)]
                for i in range(n_layers)
            ]
        elif mode.startswith("learned_"):
            if self.interleave_layer_gates is None:
                raise RuntimeError("learned interleave gates were not built")
            learned_schedule = []
            for layer in range(n_layers):
                local_gate = self.interleave_layer_gates[layer, 0]
                temporal_gate = self.interleave_layer_gates[layer, 1]
                if mode == "learned_dual_interleave":
                    layer_bias = (
                        local_gate * shaped(
                            local_dense.to(torch.float32))
                        + temporal_gate * shaped(
                            temporal.to(torch.float32))
                    )
                elif mode == "learned_soft_interleave":
                    layer_bias = (
                        local_gate * shaped(
                            local_dense.to(torch.float32))
                        if layer % 2 == 0
                        else temporal_gate * shaped(
                            temporal.to(torch.float32))
                    )
                else:
                    bridge_position = layer % 4
                    if bridge_position == 0:
                        layer_bias = local_gate * shaped(
                            local_dense.to(torch.float32))
                    elif bridge_position == 1:
                        layer_bias = temporal_gate * shaped(
                            temporal.to(torch.float32))
                    else:
                        layer_bias = torch.zeros(
                            (1, 1, sequence_length, sequence_length),
                            device=device,
                            dtype=torch.float32,
                        )
                learned_schedule.append(layer_bias)
            return learned_schedule
        elif mode == "catalog_graph_interleave":
            soft_schedule = [
                catalog_bias + local_additive,
                catalog_bias + temporal_additive,
            ]
            return [
                soft_schedule[i % len(soft_schedule)]
                for i in range(n_layers)
            ]
        else:
            raise RuntimeError(f"Unhandled attention mode: {mode}")
        return [shaped(schedule[i % len(schedule)]) for i in range(n_layers)]

    def _build_catalog_token_graph(self) -> torch.Tensor:
        """Build normalized token co-occurrence edges from valid SID tuples."""
        vocab_size = self.tokenizer.vocab_size
        tuples = self.valid_item_tuples.detach().cpu()
        flat_pairs = []
        for source in range(tuples.shape[1]):
            for target in range(tuples.shape[1]):
                if source == target:
                    continue
                flat_pairs.append(
                    tuples[:, source] * vocab_size + tuples[:, target])
        pair_index = torch.cat(flat_pairs)
        counts = torch.bincount(
            pair_index, minlength=vocab_size * vocab_size
        ).reshape(vocab_size, vocab_size).to(torch.float32)
        frequency = torch.bincount(
            tuples.reshape(-1), minlength=vocab_size).to(torch.float32)
        denominator = torch.sqrt(
            frequency[:, None] * frequency[None, :]).clamp_min(1.0)
        association = counts / denominator
        association.fill_diagonal_(0.0)
        max_edge = association.max()
        if max_edge > 0:
            association = association / max_edge
        return association.to(self.config["device"])

    def llada_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[Union[torch.Tensor,
                                       List[torch.Tensor]]] = None,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.add_item_pos_emb(input_ids, inputs_embeds)
        if attention_bias is None:
            attention_bias = self._structured_attention_bias(
                input_ids.shape[1], input_ids.device, input_ids)

        return self.llada(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )

    def his_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
    ) -> CausalLMOutputWithPast:
        masked_input_tokens, masked_indices, p_mask = self.get_masked_seqs(
            input_tokens, input_attention_mask)
        soft_inputs = self._soft_diffusion_embeddings(
            masked_input_tokens, masked_indices, p_mask)

        outputs = self.llada_forward(
            input_ids=masked_input_tokens,
            attention_mask=input_attention_mask.long(),
            inputs_embeds=soft_inputs)

        logits = outputs.logits
        logits = logits / self.temperature

        input_tokens = input_tokens.masked_fill(~input_attention_mask,
                                                self.tokenizer.ignored_label)

        token_loss = self.loss_fct(
            logits[masked_indices],
            input_tokens[masked_indices]) / p_mask[masked_indices]
        token_loss = token_loss * self._prefix_loss_weights(
            masked_indices, input_tokens)
        loss = token_loss.sum() / (input_tokens.shape[0] *
                                   input_tokens.shape[1])
        outputs.loss = loss

        return outputs

    def target_mask_loss(
        self,
        input_tokens,
        input_attention_mask,
        label_tokens,
        label_attention_mask,
        label_catalog_index,
    ) -> CausalLMOutputWithPast:

        if self.target_noising_strategy == "on_policy_coherent":
            (
                input_tokens,
                input_attention_mask,
                label_tokens,
                label_attention_mask,
                label_catalog_index,
                masked_label_tokens,
                masked_indices,
                p_mask,
            ) = self._on_policy_coherent_target_views(
                input_tokens,
                input_attention_mask,
                label_tokens,
                label_attention_mask,
                label_catalog_index,
            )
        else:
            masked_label_tokens, masked_indices, p_mask = \
                self.get_masked_seqs(
                    label_tokens, label_attention_mask, is_target=True)
        soft_label_inputs = self._soft_diffusion_embeddings(
            masked_label_tokens, masked_indices, p_mask)

        final_inputs_ids = torch.cat([input_tokens, masked_label_tokens],
                                     dim=1)
        final_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)

        history_embeds = None
        soft_inputs = None
        if soft_label_inputs is not None:
            history_embeds = self.llada.model.transformer.wte(input_tokens)
            soft_inputs = torch.cat([history_embeds, soft_label_inputs], dim=1)

        outputs = self.llada_forward(
            input_ids=final_inputs_ids,
            attention_mask=final_attention_mask.long(),
            inputs_embeds=soft_inputs)

        logits = outputs.logits
        logits = logits[:, -masked_label_tokens.shape[1]:, :]

        logits = logits / self.temperature

        label_tokens = label_tokens.masked_fill(~label_attention_mask,
                                                self.tokenizer.ignored_label)

        def denoising_loss(current_logits: torch.Tensor) -> torch.Tensor:
            token_loss = self.loss_fct(
                current_logits[masked_indices],
                label_tokens[masked_indices]) / p_mask[masked_indices]
            token_loss = (
                token_loss * self._prefix_loss_weights(
                    masked_indices, label_tokens))
            return token_loss.sum() / (
                label_tokens.shape[0] * label_tokens.shape[1])

        classification_loss = denoising_loss(logits)
        objective_logits = logits
        view_penalty = logits.new_zeros(())

        if self.soft_view_consistency_w > 0:
            # A stop-gradient stochastic teacher avoids retaining a second
            # full transformer graph while still penalizing trajectory
            # disagreement. Across batches either draw can be the student.
            with torch.no_grad():
                second_label_inputs = self._soft_diffusion_embeddings(
                    masked_label_tokens, masked_indices, p_mask)
                second_history_embeds = (
                    self.llada.model.transformer.wte(input_tokens))
                second_inputs = torch.cat(
                    [second_history_embeds, second_label_inputs], dim=1)
                second_outputs = self.llada_forward(
                    input_ids=final_inputs_ids,
                    attention_mask=final_attention_mask.long(),
                    inputs_embeds=second_inputs)
                second_logits = second_outputs.logits[
                    :, -masked_label_tokens.shape[1]:, :]
                second_logits = second_logits / self.temperature
            view_penalty = self._soft_view_js_loss(
                logits, second_logits, masked_indices)
            objective_logits = 0.5 * (logits + second_logits)

        if self.soft_self_condition_mode == "catalog_posterior":
            refined_label_inputs = self._catalog_posterior_embeddings(
                masked_label_tokens,
                masked_indices,
                p_mask,
                objective_logits.detach(),
            )
            if history_embeds is None:
                history_embeds = self.llada.model.transformer.wte(
                    input_tokens)
            refined_inputs = torch.cat(
                [history_embeds, refined_label_inputs], dim=1)
            refined_outputs = self.llada_forward(
                input_ids=final_inputs_ids,
                attention_mask=final_attention_mask.long(),
                inputs_embeds=refined_inputs)
            refined_logits = refined_outputs.logits[
                :, -masked_label_tokens.shape[1]:, :]
            refined_logits = refined_logits / self.temperature
            refined_loss = denoising_loss(refined_logits)
            classification_loss = (
                (1.0 - self.soft_self_condition_aux_w) * refined_loss
                + self.soft_self_condition_aux_w * classification_loss)
            objective_logits = refined_logits
            outputs = refined_outputs

        item_arch_loss = objective_logits.new_zeros(())
        item_arch_gate_mean = objective_logits.new_zeros(())
        item_arch_candidate_count = objective_logits.new_zeros(())
        if self.item_arch_mode != "off":
            if outputs.hidden_states is None:
                raise RuntimeError(
                    "item architecture requires transformer hidden states")
            item_output = self._item_architecture_forward(
                outputs.hidden_states[-1][
                    :, -masked_label_tokens.shape[1]:, :],
                objective_logits,
                masked_label_tokens,
                masked_indices,
                target_index=label_catalog_index,
                history_ids=input_tokens,
            )
            item_arch_loss = item_output.loss
            item_arch_gate_mean = item_output.gate_mean
            item_arch_candidate_count = item_output.candidate_count
            if self.item_arch_mode == "epic":
                objective_logits = item_output.logits
                classification_loss = denoising_loss(objective_logits)

        loss = (
            classification_loss
            + self.soft_view_consistency_w * view_penalty)
        if self.item_arch_loss_w > 0:
            loss = loss + self.item_arch_loss_w * item_arch_loss
        if self.tuple_consistency_w > 0:
            loss = loss + self.tuple_consistency_w * \
                self._tuple_manifold_loss(objective_logits)
        if self.catalog_item_loss_w > 0:
            loss = loss + self.catalog_item_loss_w * \
                self._catalog_item_classification_loss(
                    objective_logits,
                    label_catalog_index,
                    masked_indices,
                )
        outputs.loss = loss
        if self.item_arch_mode != "off":
            outputs.item_arch_loss = item_arch_loss
            outputs.item_arch_gate_mean = item_arch_gate_mean
            outputs.item_arch_candidate_count = item_arch_candidate_count

        return outputs

    def _on_policy_coherent_target_views(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        label_attention_mask: torch.Tensor,
        label_catalog_index: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build DiffGRM-style nested target views ordered by uncertainty.

        The policy observes the history and an entirely masked target. It
        never reads the clean target SID. A single deterministic policy pass
        ranks SID positions from least to most confident, after which the
        training views mask increasingly large prefixes of that difficulty
        order. This is the least-confidence/static OCN variant that performs
        best in DiffGRM's reported ablation.
        """
        batch_size, sid_length = label_tokens.shape
        all_mask_target = torch.full_like(
            label_tokens, self.mask_token_id)
        policy_input_ids = torch.cat(
            [input_tokens, all_mask_target], dim=1)
        policy_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            policy_outputs = self.llada_forward(
                input_ids=policy_input_ids,
                attention_mask=policy_attention_mask.long(),
            )
            policy_logits = (
                policy_outputs.logits[:, -sid_length:, :]
                / self.temperature)
        if was_training:
            self.train()

        confidence_by_digit = []
        for digit in range(sid_length):
            valid_tokens = self.posValidTokens[digit]
            valid_tokens = valid_tokens[valid_tokens.ne(0)]
            digit_logits = policy_logits[:, digit, valid_tokens]
            probability = F.softmax(digit_logits.float(), dim=-1)
            if self.ocn_confidence_metric == "entropy":
                confidence = (
                    probability
                    * probability.clamp_min(1e-12).log()
                ).sum(dim=-1)
            else:
                confidence = probability.max(dim=-1).values
            confidence_by_digit.append(confidence)
        confidence = torch.stack(confidence_by_digit, dim=1)
        difficulty_order = torch.argsort(
            confidence, dim=1, descending=False)

        mask_counts = torch.linspace(
            1,
            sid_length,
            steps=self.ocn_num_views,
            device=label_tokens.device,
        ).round().long()
        view_masks = []
        for mask_count in mask_counts.tolist():
            mask = torch.zeros_like(label_tokens, dtype=torch.bool)
            mask.scatter_(
                1, difficulty_order[:, :mask_count], True)
            view_masks.append(mask)
        masked_indices = torch.cat(view_masks, dim=0)

        repeats = self.ocn_num_views
        expanded_labels = label_tokens.repeat(repeats, 1)
        masked_label_tokens = torch.where(
            masked_indices,
            self.mask_token_id,
            expanded_labels,
        )
        p_mask = (
            masked_indices.float().mean(dim=1, keepdim=True)
            .expand_as(masked_indices))

        return (
            input_tokens.repeat(repeats, 1),
            input_attention_mask.repeat(repeats, 1),
            expanded_labels,
            label_attention_mask.repeat(repeats, 1),
            label_catalog_index.repeat(repeats),
            masked_label_tokens,
            masked_indices,
            p_mask,
        )

    def forward(self, batch: dict):
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_tokens = input_tokens.reshape((input_tokens.shape[0], -1))
        input_attention_mask = input_tokens != 0

        assert 'labels' in batch, 'The batch must contain the labels.'
        label_item_ids = batch['labels'].squeeze(1)
        label_tokens = self.item_id2tokens[label_item_ids]
        label_tokens = label_tokens.reshape((label_tokens.shape[0], -1))
        label_catalog_index = self.item_id2catalog_index[label_item_ids]
        if label_catalog_index.lt(0).any():
            raise RuntimeError("target item is missing from SID catalog")
        label_attention_mask = torch.ones_like(
            label_tokens, device=label_tokens.device).bool()

        his_inputs = torch.cat([input_tokens, label_tokens], dim=1)
        his_attention_mask = torch.cat(
            [input_attention_mask, label_attention_mask], dim=1)
        his_outputs = self.his_mask_loss(his_inputs, his_attention_mask)
        his_loss = his_outputs.loss

        target_outputs = self.target_mask_loss(input_tokens,
                                               input_attention_mask,
                                               label_tokens,
                                               label_attention_mask,
                                               label_catalog_index)
        target_loss = target_outputs.loss

        loss = target_loss + self.his_mask_w * his_loss

        return LLaDARecOutPut(loss=loss,
                              his_mask_loss=his_loss,
                              target_mask_loss=target_loss,
                              item_arch_loss=getattr(
                                  target_outputs,
                                  "item_arch_loss",
                                  None,
                              ),
                              item_arch_gate_mean=getattr(
                                  target_outputs,
                                  "item_arch_gate_mean",
                                  None,
                              ),
                              item_arch_candidate_count=getattr(
                                  target_outputs,
                                  "item_arch_candidate_count",
                                  None,
                              ))

    def get_masked_seqs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_target:
            return self.get_masked_targets(input_ids, attention_mask)
        else:
            return self.get_masked_seqs_his(input_ids, attention_mask)

    def get_masked_seqs_his(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape

        h = self.tokenizer.n_digit
        n_items = l // h
        masked_num = torch.randint(low=0,
                                   high=h + 1,
                                   size=(b, n_items),
                                   device=input_ids.device)

        masked_prob = masked_num / h
        masked_prob = torch.repeat_interleave(masked_prob,
                                              h,
                                              dim=1)

        masked_indices = torch.rand((b, l),
                                    device=input_ids.device) < masked_prob

        use_structured = self.masking_strategy == "item"
        if self.masking_strategy in {
                "token_item_mixed", "token_suffix_mixed"
        }:
            structured_rows = (
                torch.rand(b, device=input_ids.device)
                < self.structured_mask_prob)
        else:
            structured_rows = torch.full(
                (b, ), use_structured, device=input_ids.device,
                dtype=torch.bool)

        if structured_rows.any():
            row_index = structured_rows.nonzero(as_tuple=False).squeeze(1)
            row_prob = masked_num[row_index] / h
            if self.masking_strategy in {"item", "token_item_mixed"}:
                item_mask = (
                    torch.rand_like(row_prob, dtype=torch.float32) < row_prob)
                structured_mask = item_mask.repeat_interleave(h, dim=1)
            else:
                start = torch.randint(
                    0, h, (row_index.numel(), n_items),
                    device=input_ids.device)
                active_item = (
                    torch.rand_like(row_prob, dtype=torch.float32) < row_prob)
                digits = torch.arange(h, device=input_ids.device)
                structured_mask = (
                    active_item[:, :, None]
                    & (digits[None, None, :] >= start[:, :, None])
                ).reshape(row_index.numel(), l)
            masked_indices[row_index] = structured_mask
            structured_prob = structured_mask.reshape(
                row_index.numel(), n_items, h).float().mean(dim=-1)
            masked_prob[row_index] = structured_prob.repeat_interleave(
                h, dim=1)

        noisy_batch = torch.where(masked_indices, self.mask_token_id,
                                  input_ids)

        noisy_batch = noisy_batch.masked_fill(~attention_mask, 0)
        masked_indices = masked_indices.masked_fill(~attention_mask, False)

        return noisy_batch, masked_indices, masked_prob

    def get_masked_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape
        h = self.tokenizer.n_digit

        mask_num = torch.randint(1,
                                 h + 1, (b, ),
                                 device=input_ids.device)
        p_mask = mask_num / h

        rand_scores = torch.rand((b, l), device=input_ids.device)
        rand_scores = rand_scores.masked_fill(attention_mask == 0, 2.0)
        sorted_idx = torch.argsort(rand_scores, dim=1)  # (b, l)

        sort_input_ids = input_ids.clone()
        sort_input_ids = sort_input_ids[torch.arange(b).unsqueeze(1),
                                        sorted_idx]
        sort_mask_pos = torch.arange(
            l, device=input_ids.device)[None, :] < mask_num[:, None]
        sort_input_ids[sort_mask_pos] = self.mask_token_id

        masked_input_ids = input_ids.clone()
        masked_input_ids[torch.arange(b).unsqueeze(1),
                         sorted_idx] = sort_input_ids

        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[masked_input_ids == self.mask_token_id] = True

        use_structured = self.masking_strategy == "item"
        if self.masking_strategy in {
                "token_item_mixed", "token_suffix_mixed"
        }:
            structured_rows = (
                torch.rand(b, device=input_ids.device)
                < self.structured_mask_prob)
        else:
            structured_rows = torch.full(
                (b, ), use_structured, device=input_ids.device,
                dtype=torch.bool)
        if structured_rows.any():
            row_index = structured_rows.nonzero(as_tuple=False).squeeze(1)
            if self.masking_strategy in {"item", "token_item_mixed"}:
                mask_positions[row_index] = True
            else:
                suffix_start = torch.randint(
                    0, h, (row_index.numel(), ), device=input_ids.device)
                suffix = torch.arange(h, device=input_ids.device)[None, :] \
                    >= suffix_start[:, None]
                mask_positions[row_index] = suffix
            masked_input_ids[row_index] = torch.where(
                mask_positions[row_index],
                self.mask_token_id,
                input_ids[row_index],
            )
            p_mask[row_index] = mask_positions[row_index].float().mean(dim=1)

        return masked_input_ids, mask_positions, p_mask[:, None].repeat(1, l)

    def _prefix_loss_weights(
        self,
        masked_indices: torch.Tensor,
        clean_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weight tokens by prefix importance and catalog resolution."""
        positions = masked_indices.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return torch.ones(
                positions.shape[0],
                device=masked_indices.device,
                dtype=torch.float32,
            )
        hierarchy = positions[:, 1].remainder(self.tokenizer.n_digit)
        weights = torch.ones(
            hierarchy.shape,
            device=hierarchy.device,
            dtype=torch.float32,
        )
        if "catalog_info" in self.prefix_loss_mode:
            gain = self.catalog_information_gain[hierarchy]
            weights = weights + (self.prefix_loss_boost - 1.0) * gain
        elif self.prefix_loss_boost != 1.0:
            weights = torch.where(
                hierarchy.lt(self.soft_prefix_length),
                torch.full_like(weights, self.prefix_loss_boost),
                weights,
            )

        if ("resolution" in self.prefix_loss_mode
                and self.resolved_suffix_loss_weight < 1.0):
            if clean_tokens is None:
                raise ValueError(
                    "resolution-aware loss requires clean target tokens")
            h = self.tokenizer.n_digit
            n_items = clean_tokens.shape[1] // h
            item_index = torch.div(
                positions[:, 1], h, rounding_mode="floor")
            item_tuples = clean_tokens.reshape(
                clean_tokens.shape[0], n_items, h)[
                    positions[:, 0], item_index]
            bucket_rows, found = self._lookup_prefix_buckets(item_tuples)
            resolved = (
                found
                & self.prefix_bucket_counts[bucket_rows].eq(1)
                & hierarchy.ge(self.soft_prefix_length)
            )
            weights = torch.where(
                resolved,
                weights * self.resolved_suffix_loss_weight,
                weights,
            )
        return weights / weights.mean().clamp_min(1e-6)

    def _soft_diffusion_embeddings(
        self,
        noisy_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Replace selected hard masks with clean-label-free simplex states."""
        if self.soft_diffusion_mode == "off":
            return None
        positions = masked_indices.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return self.llada.model.transformer.wte(noisy_ids)

        h = self.tokenizer.n_digit
        n_items = noisy_ids.shape[1] // h
        group_ids = positions[:, 0] * n_items + positions[:, 1] // h
        unique_groups, inverse = torch.unique(
            group_ids, sorted=True, return_inverse=True)
        selected_groups = (
            torch.rand(unique_groups.numel(), device=noisy_ids.device)
            < self.soft_diffusion_prob)
        selected_positions = selected_groups[inverse]
        if not selected_positions.any():
            return self.llada.model.transformer.wte(noisy_ids)

        positions = positions[selected_positions]
        inverse = inverse[selected_positions]
        # Compact inverse IDs after removing groups that were not selected.
        compact_groups, inverse = torch.unique(
            unique_groups[inverse], sorted=True, return_inverse=True)
        num_groups = int(inverse.max().item()) + 1
        hierarchy = positions[:, 1].remainder(h)
        k = self.soft_diffusion_neighbors
        candidate_weights = None
        group_rows = torch.div(
            compact_groups, n_items, rounding_mode="floor")
        group_items = compact_groups.remainder(n_items)
        group_noisy = noisy_ids.reshape(
            noisy_ids.shape[0], n_items, h)[group_rows, group_items]

        if self.soft_diffusion_mode == "prefix_posterior_simplex":
            # The prefix remains a hard-discrete prediction problem. Soft
            # states are used only when the whole prefix is observed and only
            # for suffix positions. This makes the forward kernel conditional
            # on information available at inference rather than the label.
            bucket_rows, prefix_found = self._lookup_prefix_buckets(
                group_noisy)
            eligible_group = (
                prefix_found
                & self.prefix_bucket_counts[bucket_rows].ge(
                    self.prefix_soft_min_bucket)
            )
            if self.prefix_posterior_dropout > 0:
                eligible_group = (
                    eligible_group
                    & torch.rand(
                        eligible_group.shape,
                        device=noisy_ids.device,
                    ).ge(self.prefix_posterior_dropout)
                )
            keep = (
                eligible_group[inverse]
                & hierarchy.ge(self.soft_prefix_length)
            )
            if not keep.any():
                return self.llada.model.transformer.wte(noisy_ids)
            positions = positions[keep]
            retained_groups = compact_groups[inverse[keep]]
            compact_groups, inverse = torch.unique(
                retained_groups, sorted=True, return_inverse=True)
            num_groups = compact_groups.numel()
            hierarchy = positions[:, 1].remainder(h)
            group_rows = torch.div(
                compact_groups, n_items, rounding_mode="floor")
            group_items = compact_groups.remainder(n_items)
            group_noisy = noisy_ids.reshape(
                noisy_ids.shape[0], n_items, h)[group_rows, group_items]

        if self.soft_diffusion_mode in {
                "tuple_simplex",
                "local_tuple_simplex",
                "posterior_tuple_simplex",
                "prefix_posterior_simplex",
        }:
            if self.soft_diffusion_mode == "local_tuple_simplex":
                pool_size = self.soft_local_pool
                anchor_index = torch.randint(
                    0,
                    self.valid_item_tuples.shape[0],
                    (num_groups, 1),
                    device=noisy_ids.device,
                )
                pool_index = torch.randint(
                    0,
                    self.valid_item_tuples.shape[0],
                    (num_groups, pool_size),
                    device=noisy_ids.device,
                )
                # Always include the randomly sampled anchor. The anchor is
                # independent of the clean target, so this remains
                # label-leakage free.
                pool_index[:, :1] = anchor_index
                anchor = self.valid_item_tuples[anchor_index.squeeze(1)]
                pool = self.valid_item_tuples[pool_index]
                hamming = pool.ne(anchor[:, None, :]).sum(dim=-1)
                nearest = torch.topk(
                    hamming, k=k, dim=1, largest=False).indices
                tuple_index = pool_index.gather(1, nearest)
            elif self.soft_diffusion_mode == "prefix_posterior_simplex":
                bucket_rows, prefix_found = self._lookup_prefix_buckets(
                    group_noisy)
                if not prefix_found.all():
                    raise RuntimeError(
                        "prefix-posterior groups lost their visible prefix")
                bucket_counts = self.prefix_bucket_counts[bucket_rows]
                offsets = torch.floor(
                    torch.rand(
                        num_groups,
                        k,
                        device=noisy_ids.device,
                    ) * bucket_counts[:, None].to(torch.float32)
                ).to(torch.long)
                tuple_index = self.prefix_bucket_members[
                    bucket_rows[:, None], offsets]
            elif self.soft_diffusion_mode == "posterior_tuple_simplex":
                pool_index = torch.randint(
                    0,
                    self.valid_item_tuples.shape[0],
                    (num_groups, self.soft_posterior_pool),
                    device=noisy_ids.device,
                )
                pool = self.valid_item_tuples[pool_index]
                visible = (
                    group_noisy.ne(self.mask_token_id)
                    & group_noisy.ne(0))
                level = torch.arange(
                    h, device=noisy_ids.device, dtype=torch.float32)
                level_weight = torch.full_like(
                    level,
                    self.soft_posterior_hierarchy_decay,
                ).pow(level)
                weighted_visible = (
                    visible.to(torch.float32) * level_weight[None, :])
                match_score = (
                    pool.eq(group_noisy[:, None, :]).to(torch.float32)
                    * weighted_visible[:, None, :]
                ).sum(dim=-1)
                match_score = match_score / weighted_visible.sum(
                    dim=-1, keepdim=True).clamp_min(1.0)
                # Break ties without consulting the clean target. Sampling
                # remains a stochastic forward diffusion kernel.
                match_score = match_score + 1e-4 * torch.rand_like(match_score)
                posterior_score, nearest = torch.topk(
                    match_score, k=k, dim=1)
                tuple_index = pool_index.gather(1, nearest)
                candidate_weights = F.softmax(
                    posterior_score / self.soft_posterior_temperature,
                    dim=-1,
                )
            else:
                tuple_index = torch.randint(
                    0, self.valid_item_tuples.shape[0],
                    (num_groups, k), device=noisy_ids.device)
            sampled = self.valid_item_tuples[tuple_index]
            candidate_tokens = sampled[inverse].gather(
                2,
                hierarchy[:, None, None].expand(-1, k, 1),
            ).squeeze(-1)
            weight_groups = num_groups
            weight_inverse = inverse
        else:
            tuple_index = torch.randint(
                0, self.valid_item_tuples.shape[0],
                (positions.shape[0], k), device=noisy_ids.device)
            sampled = self.valid_item_tuples[tuple_index]
            candidate_tokens = sampled.gather(
                2,
                hierarchy[:, None, None].expand(-1, k, 1),
            ).squeeze(-1)
            weight_groups = positions.shape[0]
            weight_inverse = torch.arange(
                positions.shape[0], device=noisy_ids.device)

        candidate_embeds = self.llada.model.transformer.wte(candidate_tokens)
        if candidate_weights is not None:
            weights = candidate_weights.to(candidate_embeds.dtype)
        else:
            if self.soft_simplex_alpha == 1.0:
                simplex_mass = -torch.log(
                    torch.rand(
                        weight_groups,
                        k,
                        device=noisy_ids.device,
                        dtype=candidate_embeds.dtype,
                    ).clamp_min(1e-6))
            else:
                concentration = torch.full(
                    (weight_groups, k),
                    self.soft_simplex_alpha,
                    device=noisy_ids.device,
                    dtype=candidate_embeds.dtype,
                )
                simplex_mass = torch._standard_gamma(
                    concentration).clamp_min(1e-6)
            weights = simplex_mass / simplex_mass.sum(
                dim=-1, keepdim=True)
        simplex = (
            candidate_embeds
            * weights[weight_inverse, :, None]
        ).sum(dim=1).detach()

        row_fraction = p_mask[
            positions[:, 0], positions[:, 1]].clamp(0.0, 1.0)
        strength = self._soft_transport_strength(
            row_fraction, hierarchy)
        if self.soft_diffusion_mode == "prefix_posterior_simplex":
            suffix_scale = torch.where(
                hierarchy.ge(self.soft_prefix_length),
                torch.full_like(strength, self.soft_suffix_strength_scale),
                torch.ones_like(strength),
            )
            strength = (strength * suffix_scale).clamp(0.0, 1.0)
        hard_mask = self.llada.model.transformer.wte(
            noisy_ids[positions[:, 0], positions[:, 1]])
        replacement = (
            (1.0 - strength[:, None]) * hard_mask
            + strength[:, None] * simplex)

        embeddings = self.llada.model.transformer.wte(noisy_ids).clone()
        embeddings[positions[:, 0], positions[:, 1]] = replacement
        return embeddings

    def _soft_transport_strength(
        self,
        row_fraction: torch.Tensor,
        hierarchy: torch.Tensor,
    ) -> torch.Tensor:
        """Return the continuous transport amplitude at each masked token."""
        base = self.soft_diffusion_strength
        if self.soft_strength_mode == "constant":
            return torch.full_like(row_fraction, base)
        if self.soft_strength_mode == "mid_snr":
            mid_snr = (
                4.0 * row_fraction * (1.0 - row_fraction)
            ).clamp_min(0.0).pow(self.soft_fraction_power)
            return base * mid_snr

        fraction_strength = base * (
            1.0 - row_fraction).pow(self.soft_fraction_power)
        if self.soft_strength_mode == "fraction":
            return fraction_strength
        if self.soft_strength_router is None:
            raise RuntimeError(
                "learned_hierarchy strength requires a router")
        hierarchy_scale = max(self.tokenizer.n_digit - 1, 1)
        router_features = torch.stack(
            [
                row_fraction,
                hierarchy.to(row_fraction.dtype) / hierarchy_scale,
            ],
            dim=-1,
        )
        # 0.5 + sigmoid(.) is initialized to one and provides a bounded,
        # learnable residual correction to the successful fraction schedule.
        residual = (
            0.5 + torch.sigmoid(
                self.soft_strength_router(router_features).squeeze(-1)))
        return (fraction_strength * residual).clamp(0.0, 1.0)

    def _soft_view_js_loss(
        self,
        first_logits: torch.Tensor,
        second_logits: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Jensen-Shannon consistency between two sampled soft trajectories."""
        first = (
            first_logits[masked_indices] / self.soft_view_temperature)
        second = (
            second_logits[masked_indices] / self.soft_view_temperature)
        first_log_prob = F.log_softmax(first, dim=-1)
        second_log_prob = F.log_softmax(second, dim=-1)
        mixture_log_prob = (
            torch.logaddexp(first_log_prob, second_log_prob)
            - np.log(2.0))
        first_kl = (
            first_log_prob.exp()
            * (first_log_prob - mixture_log_prob)).sum(dim=-1)
        second_kl = (
            second_log_prob.exp()
            * (second_log_prob - mixture_log_prob)).sum(dim=-1)
        return 0.5 * (first_kl + second_kl).mean()

    def _catalog_posterior_embeddings(
        self,
        noisy_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Project a denoiser prediction to valid tuples and feed it back.

        The guide is stop-gradient at the call site. Visible SID digits are
        enforced exactly, and no clean masked label is consulted.
        """
        h = self.tokenizer.n_digit
        if noisy_ids.shape[1] != h or guide_logits.shape[1] != h:
            raise ValueError(
                "catalog posterior self-conditioning currently applies to "
                f"one target tuple of length {h}, got "
                f"{noisy_ids.shape=} and {guide_logits.shape=}")

        valid = self.valid_item_tuples
        log_prob = F.log_softmax(
            guide_logits.to(torch.float32)
            / self.soft_self_condition_temperature,
            dim=-1,
        )
        tuple_score = torch.zeros(
            (noisy_ids.shape[0], valid.shape[0]),
            device=noisy_ids.device,
            dtype=log_prob.dtype,
        )
        visible = (
            ~masked_indices
            & noisy_ids.ne(0)
            & noisy_ids.ne(self.mask_token_id))
        for digit in range(h):
            tuple_score = tuple_score + log_prob[:, digit, :][
                :, valid[:, digit]]
            incompatible = (
                visible[:, digit, None]
                & valid[None, :, digit].ne(noisy_ids[:, digit, None]))
            tuple_score = tuple_score.masked_fill(
                incompatible, -torch.inf)

        topk = min(self.soft_self_condition_topk, valid.shape[0])
        posterior_score, posterior_index = torch.topk(
            tuple_score, k=topk, dim=-1)
        if not torch.isfinite(posterior_score[:, 0]).all():
            raise RuntimeError(
                "No catalog tuple is compatible with visible SID digits")
        posterior_weight = F.softmax(posterior_score, dim=-1)
        posterior_tuples = valid[posterior_index]
        posterior_embeds = self.llada.model.transformer.wte(
            posterior_tuples)
        barycenter = (
            posterior_embeds
            * posterior_weight[:, :, None, None].to(
                posterior_embeds.dtype)
        ).sum(dim=1).detach()

        positions = masked_indices.nonzero(as_tuple=False)
        embeddings = self.llada.model.transformer.wte(noisy_ids).clone()
        if positions.numel() == 0:
            return embeddings
        row_fraction = p_mask[
            positions[:, 0], positions[:, 1]].clamp(0.0, 1.0)
        if self.soft_self_condition_schedule == "constant":
            strength = torch.full_like(
                row_fraction, self.soft_self_condition_strength)
        elif self.soft_self_condition_schedule == "noise":
            strength = self.soft_self_condition_strength * (
                row_fraction.pow(self.soft_self_condition_power))
        else:
            strength = self.soft_self_condition_strength * (
                (1.0 - row_fraction).pow(
                    self.soft_self_condition_power))
        hard_mask = self.llada.model.transformer.wte(
            noisy_ids[positions[:, 0], positions[:, 1]])
        posterior_state = barycenter[
            positions[:, 0], positions[:, 1]]
        embeddings[positions[:, 0], positions[:, 1]] = (
            (1.0 - strength[:, None]) * hard_mask
            + strength[:, None] * posterior_state)
        return embeddings

    def _tuple_manifold_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Negative log probability mass assigned to any catalog SID tuple."""
        h = self.tokenizer.n_digit
        if logits.shape[1] != h:
            raise ValueError(
                f"Expected target logits with {h} digits, got {logits.shape}")
        log_prob = F.log_softmax(logits, dim=-1)
        tuple_log_prob = None
        for digit in range(h):
            digit_log_prob = log_prob[:, digit, :][
                :, self.valid_item_tuples[:, digit]]
            tuple_log_prob = (
                digit_log_prob if tuple_log_prob is None
                else tuple_log_prob + digit_log_prob)
        return -torch.logsumexp(tuple_log_prob, dim=1).mean()

    def _catalog_tuple_scores(
        self,
        logits: torch.Tensor,
        candidate_tuples: Optional[torch.Tensor] = None,
        position_mask: Optional[torch.Tensor] = None,
        position_weight: Optional[torch.Tensor] = None,
        normalize_positions: bool = False,
    ) -> torch.Tensor:
        """Score complete catalog tuples under per-position denoiser logits."""
        h = self.tokenizer.n_digit
        if logits.shape[1] != h:
            raise ValueError(
                f"Expected {h} target positions, got {logits.shape}")
        if position_mask is not None and position_mask.shape != logits.shape[:2]:
            raise ValueError(
                "position_mask must match the batch and SID dimensions")
        if (position_weight is not None
                and position_weight.shape[-1] != h):
            raise ValueError(
                "position_weight must have one value per SID digit")
        candidates = (
            self.valid_item_tuples
            if candidate_tuples is None else candidate_tuples)
        log_prob = F.log_softmax(logits.to(torch.float32), dim=-1)
        scores = logits.new_zeros(
            (logits.shape[0], candidates.shape[0]),
            dtype=torch.float32,
        )
        for digit in range(h):
            digit_scores = log_prob[:, digit, :][
                :, candidates[:, digit]]
            if position_mask is not None:
                digit_scores = (
                    digit_scores
                    * position_mask[:, digit, None].to(digit_scores.dtype)
                )
            if position_weight is not None:
                digit_scores = (
                    digit_scores
                    * position_weight[:, digit, None].to(digit_scores.dtype)
                )
            scores = scores + digit_scores
        if position_mask is not None and normalize_positions:
            scores = scores / position_mask.sum(
                dim=1, keepdim=True).clamp_min(1).to(scores.dtype)
        return scores

    def _item_arch_candidate_indices(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_index: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Preselect a bounded exact partial-SID candidate set.

        Token energy is used only for the non-differentiable preselection.
        Candidate ranking itself is performed by the independent joint tuple
        encoder. During training, the gold tuple is assigned infinite
        preselection priority, guaranteeing its inclusion without exposing it
        as an input to the item scorer.
        """
        h = self.tokenizer.n_digit
        if target_ids.shape[1] != h or logits.shape[1] != h:
            raise ValueError(
                "item architecture expects one target SID tuple")
        catalog_size = self.valid_item_tuples.shape[0]
        visible = (
            target_ids.ne(self.mask_token_id)
            & target_ids.ne(0))
        if self.item_arch_compatible_only:
            compatible = (
                target_ids[:, None, :].eq(
                    self.valid_item_tuples[None, :, :])
                | ~visible[:, None, :]
            ).all(dim=-1)
        else:
            compatible = torch.ones(
                (target_ids.shape[0], catalog_size),
                device=target_ids.device,
                dtype=torch.bool,
            )
        candidate_count = compatible.sum(dim=-1)
        empty_compatible = candidate_count.eq(0)
        if empty_compatible.any():
            if target_index is not None:
                raise RuntimeError(
                    "gold-supervised item architecture found an empty "
                    "compatible catalog")
            # Parallel decoding can temporarily form a partial SID prefix
            # that no catalog item owns. Preserve exact compatibility on
            # every valid row, but recover only those dead-end beams through
            # full-catalog token-energy preselection.
            compatible = compatible.clone()
            compatible[empty_compatible] = True
            candidate_count = compatible.sum(dim=-1)
            if not self._logged_item_arch_empty_fallback:
                self.log(
                    "item architecture empty-compatible fallback: "
                    "rows={}/{} (decode-only full-catalog preselection)"
                    .format(
                        empty_compatible.sum().item(),
                        empty_compatible.numel(),
                    ))
                self._logged_item_arch_empty_fallback = True

        with torch.no_grad():
            preselection_score = self._catalog_tuple_scores(
                logits.detach())
            preselection_score = preselection_score.masked_fill(
                ~compatible, -torch.inf)
            if target_index is not None:
                rows = torch.arange(
                    target_ids.shape[0], device=target_ids.device)
                if not compatible[rows, target_index].all():
                    raise RuntimeError(
                        "gold item is incompatible with its visible SID")
                preselection_score[rows, target_index] = torch.inf
            topk = min(self.item_arch_candidate_topk, catalog_size)
            candidate_index = torch.topk(
                preselection_score,
                k=topk,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            candidate_valid = compatible.gather(1, candidate_index)

        target_position = None
        if target_index is not None:
            target_match = candidate_index.eq(target_index[:, None])
            if not target_match.any(dim=-1).all():
                raise RuntimeError(
                    "gold item was not retained by item preselection")
            target_position = target_match.to(torch.long).argmax(dim=-1)

        return (
            candidate_index,
            candidate_valid,
            candidate_count,
            target_position,
        )

    def _item_arch_encode_tuples(
        self,
        tuple_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a complete SID tuple with optional cross-head structure.

        The pairwise residual uses

        ``((sum_j z_j)^2 - sum_j z_j^2) / (H * (H - 1))``

        which is exactly the mean element-wise interaction over unordered
        head pairs without materializing an H-by-H tensor.  The residual is
        zero-gated, so enabling it preserves the flat encoder at warm start.
        """
        if self.item_arch_tuple_encoder is None:
            raise RuntimeError("item tuple encoder was not built")
        flat_item = self.item_arch_tuple_encoder(
            tuple_embed.flatten(start_dim=-2))
        if self.item_arch_tuple_mode == "flat":
            return flat_item
        if (self.item_arch_pair_proj is None
                or self.item_arch_pair_out is None
                or self.item_arch_pair_alpha is None):
            raise RuntimeError(
                "pairwise item tuple modules were not built")
        semantic_heads = (
            self.item_arch_pairwise_semantic_heads
            or tuple_embed.shape[-2])
        h = semantic_heads
        if h < 2:
            raise RuntimeError(
                "pairwise item tuple encoding requires at least two heads")
        projected = self.item_arch_pair_proj(
            tuple_embed[..., :semantic_heads, :])
        pair_mean = (
            projected.sum(dim=-2).square()
            - projected.square().sum(dim=-2)
        ) / float(h * (h - 1))
        pair_residual = self.item_arch_pair_out(pair_mean)
        return (
            flat_item
            + torch.tanh(self.item_arch_pair_alpha) * pair_residual
        )

    def _item_arch_query(
        self,
        target_hidden: torch.Tensor,
        history_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a target item query, optionally through an item bottleneck.

        The history path groups raw SID tokens back into complete items and
        encodes every tuple with the same item encoder used for candidates.
        It can aggregate valid items with a masked GRU, a mean-pool control,
        or parameter-free query-conditioned attention.  Every history mode
        has a zero-initialized scalar residual, so enabling the path preserves
        the original query at initialization while still giving the scalar a
        useful gradient.
        """
        if (self.item_arch_query_encoder is None
                or self.item_arch_tuple_encoder is None
                or self.item_arch_digit_emb is None):
            raise RuntimeError("item architecture modules were not built")
        query = self.item_arch_query_encoder(
            target_hidden.mean(dim=1))
        if self.item_arch_history_mode != "off":
            if self.item_arch_history_alpha is None:
                raise RuntimeError(
                    "item history bottleneck modules were not built")
            if (self.item_arch_history_mode == "gru"
                    and self.item_arch_history_cell is None):
                raise RuntimeError(
                    "item history GRU module was not built")
            if history_ids is None:
                raise RuntimeError(
                    "item history bottleneck requires history token ids")
            h = self.tokenizer.n_digit
            if history_ids.shape[1] % h:
                raise ValueError(
                    "history token length must be divisible by SID length")
            batch_size = history_ids.shape[0]
            n_items = history_ids.shape[1] // h
            history_tuple = history_ids.reshape(batch_size, n_items, h)
            history_valid = history_tuple.ne(
                self.tokenizer.padding_token).any(dim=-1)
            digit = self.item_arch_digit_emb(
                torch.arange(h, device=history_ids.device))
            tuple_embed = self.llada.model.transformer.wte(history_tuple)
            tuple_embed = tuple_embed + digit[None, None, :, :]
            history_item = self._item_arch_encode_tuples(
                tuple_embed)
            if self.item_arch_history_mode == "gru":
                state = history_item.new_zeros(
                    (batch_size, self.item_arch_hidden_dim))
                for item_index in range(n_items):
                    proposed = self.item_arch_history_cell(
                        history_item[:, item_index, :], state)
                    state = torch.where(
                        history_valid[:, item_index, None],
                        proposed,
                        state,
                    )
            elif self.item_arch_history_mode == "mean":
                valid_weight = history_valid.to(
                    history_item.dtype).unsqueeze(-1)
                valid_count = valid_weight.sum(
                    dim=1).clamp_min(1.0)
                state = (
                    history_item * valid_weight
                ).sum(dim=1) / valid_count
            else:
                # Scaled dot-product attention adds no projection matrices:
                # the target query and history items already share the joint
                # item hidden space.  Explicitly zero all-invalid rows to
                # avoid NaNs from a fully masked softmax.
                attention_logits = torch.einsum(
                    "bih,bh->bi", history_item, query)
                attention_logits = attention_logits / math.sqrt(
                    float(self.item_arch_hidden_dim))
                has_history = history_valid.any(
                    dim=-1, keepdim=True)
                attention_logits = attention_logits.masked_fill(
                    ~history_valid, -torch.inf)
                attention_logits = torch.where(
                    has_history,
                    attention_logits,
                    torch.zeros_like(attention_logits),
                )
                attention_weight = F.softmax(
                    attention_logits, dim=-1)
                attention_weight = (
                    attention_weight
                    * history_valid.to(attention_weight.dtype)
                )
                attention_weight = attention_weight / (
                    attention_weight.sum(
                        dim=-1, keepdim=True).clamp_min(1e-12)
                )
                state = torch.einsum(
                    "bi,bih->bh", attention_weight, history_item)
            query = (
                query
                + torch.tanh(self.item_arch_history_alpha) * state
            )
        return F.normalize(query, p=2, dim=-1)

    def _item_arch_head_information_gain(self) -> torch.Tensor:
        """Empirical marginal entropy per SID head over the real catalog.

        This is a cheap, generation-derived proxy for how discriminative
        each semantic-ID head actually is in the tokenizer that was fit for
        this dataset. It is used only to initialize the head-selection gate
        bias (``item_arch_head_gate_init=infogain``); it is not learned and
        it never used the target label.
        """
        tuples = self.valid_item_tuples
        entropies = []
        for head in range(tuples.shape[-1]):
            counts = torch.bincount(tuples[:, head].reshape(-1).long())
            probs = counts.to(torch.float32)
            probs = probs / probs.sum().clamp_min(1.0)
            probs = probs.clamp_min(1e-12)
            entropies.append(-(probs * probs.log()).sum())
        gain = torch.stack(entropies)
        return gain - gain.mean()

    def _item_arch_recent_selfattn_refine(
        self,
        recent_key: torch.Tensor,
        recent_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Let the K retained recent items attend to each other.

        Plain ``recent_attn`` scores each recent item against the candidate
        independently, so the K items are effectively an unordered weighted
        set. This refines the keys with one small self-attention pass over
        the retained items first, so the transition field can express
        within-history sequence structure (e.g. "bought A then B") before
        comparing against the candidate. Zero-initialized gate keeps this an
        exact no-op until training moves it off zero.
        """
        if (self.item_arch_recent_selfattn_q is None
                or self.item_arch_recent_selfattn_k is None
                or self.item_arch_recent_selfattn_v is None
                or self.item_arch_recent_selfattn_alpha is None):
            raise RuntimeError(
                "recent self-attention modules were not built")
        query = self.item_arch_recent_selfattn_q(recent_key)
        key = self.item_arch_recent_selfattn_k(recent_key)
        value = self.item_arch_recent_selfattn_v(recent_key)
        scale = 1.0 / math.sqrt(query.shape[-1])
        logits = torch.einsum("brd,bsd->brs", query, key) * scale
        logits = logits.masked_fill(
            ~recent_valid[:, None, :], -torch.inf)
        has_any = recent_valid.any(dim=-1)
        logits = torch.where(
            has_any[:, None, None],
            logits,
            torch.zeros_like(logits),
        )
        weight = F.softmax(logits, dim=-1)
        weight = weight * recent_valid[:, None, :].to(weight.dtype)
        weight = weight / weight.sum(
            dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.einsum("brs,bsd->brd", weight, value)
        refined = (
            recent_key
            + torch.tanh(self.item_arch_recent_selfattn_alpha) * context
        )
        return refined * recent_valid[:, :, None].to(refined.dtype)

    def _item_arch_bigram_edges(
        self,
        recent_key: torch.Tensor,
        recent_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode each (older, newer) consecutive pair as one edge key.

        Uses the same whole-item product/abs-difference inductive bias as
        the analytic pairwise residual and ``recent_attn`` itself (which is
        the pattern that has actually worked in this project), applied
        across consecutive time steps instead of across SID heads or a
        single item, to test second-order ("A then B implies C") sequence
        structure without assuming SID heads are cross-item-comparable.
        """
        if self.item_arch_bigram_alpha is None:
            raise RuntimeError("bigram transition modules were not built")
        retained = recent_key.shape[1]
        if retained < 2:
            edge_key = recent_key.new_zeros(
                recent_key.shape[0], 0, recent_key.shape[-1])
            edge_valid = recent_valid.new_zeros(recent_valid.shape[0], 0)
            return edge_key, edge_valid
        newer = recent_key[:, :-1, :]
        older = recent_key[:, 1:, :]
        if self.item_arch_bigram_encoder_mode == "average":
            edge_key = F.normalize(newer + older, p=2, dim=-1)
        else:
            if self.item_arch_bigram_encoder is None:
                raise RuntimeError(
                    "bigram transition modules were not built")
            if self.item_arch_bigram_encoder_mode == "concat":
                edge_input = torch.cat([newer, older], dim=-1)
            else:
                edge_input = torch.cat(
                    [newer * older, (newer - older).abs()], dim=-1)
            edge_key = F.normalize(
                self.item_arch_bigram_encoder(edge_input), p=2, dim=-1)
        edge_valid = recent_valid[:, :-1] & recent_valid[:, 1:]
        edge_key = edge_key * edge_valid[:, :, None].to(edge_key.dtype)
        return edge_key, edge_valid

    def _item_arch_recent_history_keys(
        self,
        history_ids: torch.Tensor,
        recent_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the most recent valid complete items.

        Returned items are ordered newest first.  Keeping the per-head token
        embeddings in addition to the joint item keys lets the transition
        field express both sequence-scale and SID-head-aligned interactions
        without re-embedding history once per candidate chunk.
        """
        if self.item_arch_digit_emb is None:
            raise RuntimeError("item architecture modules were not built")
        h = self.tokenizer.n_digit
        if history_ids.shape[1] % h:
            raise ValueError(
                "history token length must be divisible by SID length")
        batch_size = history_ids.shape[0]
        n_items = history_ids.shape[1] // h
        history_tuple = history_ids.reshape(batch_size, n_items, h)
        history_valid = history_tuple.ne(
            self.tokenizer.padding_token).any(dim=-1)
        position = torch.arange(
            n_items, device=history_ids.device)[None, :]
        ranked_position = position.expand(
            batch_size, -1).masked_fill(~history_valid, -1)
        retained = min(recent_k, n_items)
        recent_index = torch.topk(
            ranked_position,
            k=retained,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        recent_valid = history_valid.gather(1, recent_index)
        recent_tuple = history_tuple.gather(
            1, recent_index[:, :, None].expand(-1, -1, h))
        digit = self.item_arch_digit_emb(
            torch.arange(h, device=history_ids.device))
        tuple_embed = self.llada.model.transformer.wte(recent_tuple)
        tuple_embed = tuple_embed + digit[None, None, :, :]
        recent_key = F.normalize(
            self._item_arch_encode_tuples(tuple_embed),
            p=2,
            dim=-1,
        )
        recent_key = (
            recent_key
            * recent_valid[:, :, None].to(recent_key.dtype)
        )
        if self.item_arch_transition_mode == "recent_selfattn":
            recent_key = self._item_arch_recent_selfattn_refine(
                recent_key, recent_valid)
        if self.item_arch_transition_history_ablation == "reverse_order":
            # Flip which retained position gets the learned recency bias.
            # Eval-only validity control: run with steps=0 against an
            # already-trained checkpoint, never during training.
            recent_key = recent_key.flip(dims=[1])
            tuple_embed = tuple_embed.flip(dims=[1])
            recent_valid = recent_valid.flip(dims=[1])
        elif self.item_arch_transition_history_ablation == "swap_batch":
            # Give every example a different example's history (rolled by
            # one across the batch). Eval-only validity control.
            recent_key = torch.roll(recent_key, shifts=1, dims=0)
            tuple_embed = torch.roll(tuple_embed, shifts=1, dims=0)
            recent_valid = torch.roll(recent_valid, shifts=1, dims=0)
        return recent_key, tuple_embed, recent_valid

    def _item_arch_last_history_key(
        self,
        history_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible one-step transition history encoder."""
        recent_key, _, recent_valid = (
            self._item_arch_recent_history_keys(history_ids, 1))
        return recent_key[:, 0, :], recent_valid[:, 0]

    def _item_arch_candidate_scores(
        self,
        item_query: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_valid: torch.Tensor,
        active_heads: Optional[torch.Tensor] = None,
        history_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score joint SID tuples with a learned item representation."""
        if (self.item_arch_digit_emb is None
                or self.item_arch_tuple_encoder is None):
            raise RuntimeError("item architecture modules were not built")
        h = self.tokenizer.n_digit
        digit = self.item_arch_digit_emb(
            torch.arange(h, device=item_query.device))
        use_global_transition = self.item_arch_transition_mode in {
            "last_pair", "recent_attn", "recent_head", "recent_selfattn",
        }
        use_recent_transition = self.item_arch_transition_mode in {
            "recent_attn", "recent_head", "recent_selfattn",
        }
        use_head_transition = self.item_arch_transition_mode in {
            "headwise_last", "recent_head"
        }
        use_interleave_transition = (
            self.item_arch_transition_mode == "interleave")
        use_bigram_transition = (
            self.item_arch_transition_mode == "bigram_attn")
        recent_key = None
        recent_embed = None
        recent_valid = None
        has_history = None
        recent_weight = None
        transition_head_weight = None
        transition_route = None
        edge_key = None
        edge_valid = None
        has_edges = None
        edge_weight = None
        if use_global_transition or use_head_transition or (
                use_interleave_transition or use_bigram_transition):
            if history_ids is None:
                raise RuntimeError(
                    "transition energy requires history ids")
            recent_k = (
                self.item_arch_transition_recent_k
                if (
                    use_recent_transition or use_interleave_transition
                    or use_bigram_transition
                ) else 1
            )
            recent_key, recent_embed, recent_valid = (
                self._item_arch_recent_history_keys(
                    history_ids, recent_k))
            has_history = recent_valid.any(dim=-1)

            if self.item_arch_transition_route_mode == "ambiguity":
                selected_count = candidate_valid.sum(
                    dim=-1).to(item_query.dtype).clamp_min(1.0)
                denominator = math.log(
                    float(max(2, self.item_arch_candidate_topk)))
                transition_route = (
                    selected_count.log() / denominator
                ).clamp(0.0, 1.0)
            else:
                transition_route = item_query.new_ones(
                    item_query.shape[0])

        if use_global_transition:
            if (self.item_arch_transition_mlp is None
                    or self.item_arch_transition_alpha is None):
                raise RuntimeError(
                    "global transition modules were not built")
            if use_recent_transition:
                if self.item_arch_transition_recency_bias is None:
                    raise RuntimeError(
                        "recent transition bias was not built")
                recent_logits = torch.einsum(
                    "bd,brd->br", item_query, recent_key)
                recent_logits = (
                    recent_logits
                    / self.item_arch_transition_attention_temperature
                    + self.item_arch_transition_recency_bias[
                        None, :recent_key.shape[1]]
                )
                if (self.item_arch_transition_history_ablation
                        == "uniform_attention"):
                    recent_logits = torch.zeros_like(recent_logits)
                recent_logits = recent_logits.masked_fill(
                    ~recent_valid, -torch.inf)
                recent_logits = torch.where(
                    has_history[:, None],
                    recent_logits,
                    torch.zeros_like(recent_logits),
                )
                recent_weight = F.softmax(recent_logits, dim=-1)
                recent_weight = (
                    recent_weight
                    * recent_valid.to(recent_weight.dtype)
                )
                recent_weight = recent_weight / (
                    recent_weight.sum(
                        dim=-1, keepdim=True).clamp_min(1e-12)
                )
            else:
                recent_weight = recent_valid.to(item_query.dtype)

        if use_bigram_transition:
            if (self.item_arch_bigram_mlp is None
                    or self.item_arch_bigram_recency_bias is None
                    or self.item_arch_bigram_alpha is None):
                raise RuntimeError(
                    "bigram transition modules were not built")
            edge_key, edge_valid = self._item_arch_bigram_edges(
                recent_key, recent_valid)
            has_edges = edge_valid.any(dim=-1)
            num_edges = edge_key.shape[1]
            if num_edges:
                edge_logits = torch.einsum(
                    "bd,brd->br", item_query, edge_key)
                edge_logits = (
                    edge_logits
                    / self.item_arch_transition_attention_temperature
                    + self.item_arch_bigram_recency_bias[None, :num_edges]
                )
                edge_logits = edge_logits.masked_fill(
                    ~edge_valid, -torch.inf)
                edge_logits = torch.where(
                    has_edges[:, None],
                    edge_logits,
                    torch.zeros_like(edge_logits),
                )
                edge_weight = F.softmax(edge_logits, dim=-1)
                edge_weight = edge_weight * edge_valid.to(edge_weight.dtype)
                edge_weight = edge_weight / (
                    edge_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                )
            else:
                edge_weight = edge_valid.to(item_query.dtype)

        if use_head_transition:
            if (self.item_arch_transition_head_proj is None
                    or self.item_arch_transition_head_mlp is None
                    or self.item_arch_transition_head_gate is None
                    or self.item_arch_transition_head_alpha is None):
                raise RuntimeError(
                    "headwise transition modules were not built")
            semantic_heads = (
                self.item_arch_transition_semantic_heads or h)
            transition_head_logits = self.item_arch_transition_head_gate(
                item_query)[:, :semantic_heads]
            transition_head_weight = F.softmax(
                self._item_arch_mask_head_logits(
                    transition_head_logits,
                    (
                        active_heads[:, :semantic_heads]
                        if active_heads is not None else None
                    ),
                ),
                dim=-1,
            )

        if ((use_global_transition or use_head_transition
                or use_interleave_transition or use_bigram_transition)
                and not self._logged_item_arch_transition):
            attention_summary = "last"
            if use_recent_transition:
                attention_summary = ",".join(
                    f"{value:.3f}"
                    for value in recent_weight.mean(dim=0).tolist())
            self.log(
                "item transition field: mode={} recent_k={} "
                "route={} mean_route={:.4f} mean_valid_history={:.2f} "
                "semantic_heads={} recent_attention=[{}]".format(
                    self.item_arch_transition_mode,
                    recent_key.shape[1],
                    self.item_arch_transition_route_mode,
                    transition_route.mean().item(),
                    recent_valid.sum(
                        dim=-1).to(torch.float32).mean().item(),
                    (
                        self.item_arch_transition_semantic_heads or h
                        if use_head_transition else 0
                    ),
                    attention_summary,
                ))
            self._logged_item_arch_transition = True
        chunks = []
        for start in range(
                0, item_query.shape[0],
                self.item_arch_score_chunk_size):
            stop = min(
                start + self.item_arch_score_chunk_size,
                item_query.shape[0])
            candidate_tuple = self.valid_item_tuples[
                candidate_index[start:stop]]
            tuple_embed = self.llada.model.transformer.wte(
                candidate_tuple)
            tuple_embed = tuple_embed + digit[None, None, :, :]
            item_key = self._item_arch_encode_tuples(
                tuple_embed)
            item_key = F.normalize(item_key, p=2, dim=-1)
            chunk_query = item_query[start:stop]
            chunk_scores = torch.einsum(
                "bd,bkd->bk",
                chunk_query,
                item_key,
            )
            if use_global_transition:
                chunk_recent = recent_key[start:stop, None, :, :]
                transition_feature = torch.cat(
                    [
                        item_key[:, :, None, :] * chunk_recent,
                        (
                            item_key[:, :, None, :] - chunk_recent
                        ).abs(),
                    ],
                    dim=-1,
                )
                transition_score = self.item_arch_transition_mlp(
                    transition_feature).squeeze(-1)
                transition_score = (
                    transition_score
                    * recent_weight[start:stop, None, :]
                ).sum(dim=-1)
                transition_score = (
                    transition_score
                    * has_history[start:stop, None].to(
                        transition_score.dtype)
                    * transition_route[start:stop, None]
                )
                chunk_scores = (
                    chunk_scores
                    + torch.tanh(self.item_arch_transition_alpha)
                    * transition_score
                )
            if use_head_transition:
                semantic_heads = (
                    self.item_arch_transition_semantic_heads or h)
                candidate_head = F.normalize(
                    self.item_arch_transition_head_proj(
                        tuple_embed[..., :semantic_heads, :]),
                    p=2,
                    dim=-1,
                )
                last_head = F.normalize(
                    self.item_arch_transition_head_proj(
                        recent_embed[
                            start:stop, 0, :semantic_heads, :]),
                    p=2,
                    dim=-1,
                )
                head_feature = torch.cat(
                    [
                        candidate_head * last_head[:, None, :, :],
                        (
                            candidate_head - last_head[:, None, :, :]
                        ).abs(),
                    ],
                    dim=-1,
                )
                head_transition_value = (
                    self.item_arch_transition_head_mlp(
                        head_feature).squeeze(-1))
                # Beam decoding may retain a full-H active-head view even
                # when a structured SID excludes CID from the semantic
                # transition tensor.  Slice explicitly at the merge point so
                # the identity suffix cannot re-enter through broadcasting.
                chunk_head_weight = transition_head_weight[
                    start:stop, :semantic_heads]
                if (head_transition_value.shape[-1]
                        != chunk_head_weight.shape[-1]):
                    raise RuntimeError(
                        "semantic transition head mismatch: values={} "
                        "weights={}".format(
                            head_transition_value.shape[-1],
                            chunk_head_weight.shape[-1],
                        ))
                head_transition_score = (
                    head_transition_value
                    * chunk_head_weight[:, None, :]
                ).sum(dim=-1)
                head_transition_score = (
                    head_transition_score
                    * has_history[start:stop, None].to(
                        head_transition_score.dtype)
                    * transition_route[start:stop, None]
                )
                chunk_scores = (
                    chunk_scores
                    + torch.tanh(
                        self.item_arch_transition_head_alpha)
                    * head_transition_score
                )
            if use_interleave_transition:
                if (self.item_arch_interleave_qkv is None
                        or self.item_arch_interleave_cross_q is None
                        or self.item_arch_interleave_cross_k is None
                        or self.item_arch_interleave_mlp is None
                        or self.item_arch_interleave_head_gate is None
                        or self.item_arch_interleave_alpha is None):
                    raise RuntimeError(
                        "interleave transition modules were not built")
                # Intra-item: the candidate's own SID heads self-attend
                # before any cross-item evidence is mixed in, generalizing
                # the analytic pairwise residual into a learned within-item
                # mixing step.
                intra_qkv = self.item_arch_interleave_qkv(tuple_embed)
                intra_q, intra_k, intra_v = intra_qkv.chunk(3, dim=-1)
                intra_scale = 1.0 / math.sqrt(intra_q.shape[-1])
                intra_logits = torch.einsum(
                    "bkid,bkjd->bkij", intra_q, intra_k) * intra_scale
                intra_weight = F.softmax(intra_logits, dim=-1)
                intra_mixed = torch.einsum(
                    "bkij,bkjd->bkid", intra_weight, intra_v)
                # Inter-item: each intra-mixed head then attends over the
                # SAME semantic head across the K most recent history
                # items, so within-item structure is resolved before
                # between-item (sequence) evidence is applied, instead of
                # the two being independent parallel residuals.
                cross_q = self.item_arch_interleave_cross_q(intra_mixed)
                chunk_recent_embed = recent_embed[start:stop]
                cross_k = self.item_arch_interleave_cross_k(
                    chunk_recent_embed)
                cross_scale = 1.0 / math.sqrt(cross_q.shape[-1])
                cross_logits = torch.einsum(
                    "bkhd,brhd->bkhr", cross_q, cross_k) * cross_scale
                chunk_recent_valid = recent_valid[start:stop]
                cross_logits = cross_logits.masked_fill(
                    ~chunk_recent_valid[:, None, None, :], -torch.inf)
                chunk_has_history = has_history[start:stop]
                cross_logits = torch.where(
                    chunk_has_history[:, None, None, None],
                    cross_logits,
                    torch.zeros_like(cross_logits),
                )
                cross_weight = F.softmax(cross_logits, dim=-1)
                cross_weight = (
                    cross_weight
                    * chunk_recent_valid[:, None, None, :].to(
                        cross_weight.dtype)
                )
                cross_weight = cross_weight / cross_weight.sum(
                    dim=-1, keepdim=True).clamp_min(1e-12)
                context = torch.einsum(
                    "bkhr,brhd->bkhd", cross_weight, cross_k)
                interleave_feature = torch.cat(
                    [
                        intra_mixed * context,
                        (intra_mixed - context).abs(),
                    ],
                    dim=-1,
                )
                per_head_score = self.item_arch_interleave_mlp(
                    interleave_feature).squeeze(-1)
                interleave_head_logits = (
                    self.item_arch_interleave_head_gate(chunk_query))
                interleave_head_weight = F.softmax(
                    self._item_arch_mask_head_logits(
                        interleave_head_logits,
                        (
                            active_heads[start:stop]
                            if active_heads is not None else None
                        ),
                    ),
                    dim=-1,
                )
                interleave_score = (
                    per_head_score
                    * interleave_head_weight[:, None, :]
                ).sum(dim=-1)
                interleave_score = (
                    interleave_score
                    * chunk_has_history[:, None].to(
                        interleave_score.dtype)
                    * transition_route[start:stop, None]
                )
                chunk_scores = (
                    chunk_scores
                    + torch.tanh(self.item_arch_interleave_alpha)
                    * interleave_score
                )
            if use_bigram_transition and edge_key.shape[1]:
                chunk_edge = edge_key[start:stop, None, :, :]
                bigram_feature = torch.cat(
                    [
                        item_key[:, :, None, :] * chunk_edge,
                        (item_key[:, :, None, :] - chunk_edge).abs(),
                    ],
                    dim=-1,
                )
                bigram_score = self.item_arch_bigram_mlp(
                    bigram_feature).squeeze(-1)
                bigram_score = (
                    bigram_score * edge_weight[start:stop, None, :]
                ).sum(dim=-1)
                bigram_score = (
                    bigram_score
                    * has_edges[start:stop, None].to(bigram_score.dtype)
                    * transition_route[start:stop, None]
                )
                chunk_scores = (
                    chunk_scores
                    + torch.tanh(self.item_arch_bigram_alpha)
                    * bigram_score
                )
            if self.item_arch_score_mode == "head_attention":
                if (self.item_arch_head_proj is None
                        or self.item_arch_head_gate is None
                        or self.item_arch_head_alpha is None):
                    raise RuntimeError(
                        "head-attention item score modules were not built")
                head_key = F.normalize(
                    self.item_arch_head_proj(tuple_embed),
                    p=2,
                    dim=-1,
                )
                head_score = torch.einsum(
                    "bd,bkhd->bkh",
                    chunk_query,
                    head_key,
                )
                head_weight = F.softmax(
                    self._item_arch_mask_head_logits(
                        self.item_arch_head_gate(chunk_query),
                        (
                            active_heads[start:stop]
                            if active_heads is not None else None
                        ),
                    ),
                    dim=-1)
                structured_score = (
                    head_score * head_weight[:, None, :]
                ).sum(dim=-1)
                chunk_scores = (
                    chunk_scores
                    + torch.tanh(self.item_arch_head_alpha)
                    * structured_score
                )
            chunks.append(chunk_scores)
        scores = torch.cat(chunks, dim=0).to(torch.float32)
        scores = scores / self.item_arch_temperature
        return scores.masked_fill(~candidate_valid, -torch.inf)

    @staticmethod
    def _item_arch_mask_head_logits(
        head_logits: torch.Tensor,
        active_heads: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Prefer unresolved SID heads without creating all-masked rows."""
        if active_heads is None:
            return head_logits
        active_heads = active_heads.to(torch.bool)
        has_active = active_heads.any(dim=-1, keepdim=True)
        masked = head_logits.masked_fill(
            ~active_heads, -torch.inf)
        return torch.where(has_active, masked, head_logits)

    @staticmethod
    def _item_arch_masked_scale(
        scores: torch.Tensor,
        valid: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Scale finite candidate logits without ever forming ``0 * inf``."""
        finite_scores = scores.masked_fill(~valid, 0.0)
        scaled = scale * finite_scores
        return scaled.masked_fill(~valid, -torch.inf)

    def _item_arch_ranking_loss(
        self,
        scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_index: torch.Tensor,
        target_index: torch.Tensor,
        target_position: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Item-head CE plus an optional top-rank/@5 aligned surrogate."""
        per_example = F.cross_entropy(
            scores, target_position, reduction="none")
        if self.item_arch_sid_kernel_w > 0:
            h = self.tokenizer.n_digit
            semantic_heads = (
                self.item_arch_sid_kernel_semantic_heads or h)
            candidate_tuple = self.valid_item_tuples[
                candidate_index, :semantic_heads]
            gold_tuple = self.valid_item_tuples[
                target_index, :semantic_heads][:, None, :]
            sid_distance = candidate_tuple.ne(
                gold_tuple).to(torch.float32).mean(dim=-1)
            kernel_logits = (
                -sid_distance / self.item_arch_sid_kernel_temperature
            ).masked_fill(~candidate_valid, -torch.inf)
            kernel_target = F.softmax(
                kernel_logits, dim=-1)
            score_log_probability = F.log_softmax(
                scores, dim=-1)
            safe_log_probability = score_log_probability.masked_fill(
                ~candidate_valid, 0.0)
            kernel_cross_entropy = -(
                kernel_target * safe_log_probability
            ).sum(dim=-1)
            noise_fraction = masked_indices[
                :, :semantic_heads
            ].to(torch.float32).mean(dim=-1)
            per_example = (
                per_example
                + self.item_arch_sid_kernel_w
                * noise_fraction
                * kernel_cross_entropy
            )
            if not self._logged_item_arch_sid_kernel:
                kernel_entropy = -(
                    kernel_target
                    * kernel_target.clamp_min(1e-12).log()
                ).sum(dim=-1)
                self.log(
                    "item SID-kernel target: semantic_heads={} "
                    "temperature={:.4f} mean_noise={:.4f} "
                    "mean_entropy={:.4f} mean_distance={:.4f}".format(
                        semantic_heads,
                        self.item_arch_sid_kernel_temperature,
                        noise_fraction.mean().item(),
                        kernel_entropy.mean().item(),
                        sid_distance.masked_fill(
                            ~candidate_valid, 0.0).sum().div(
                                candidate_valid.sum().clamp_min(1)
                            ).item(),
                    ))
                self._logged_item_arch_sid_kernel = True
        if self.item_arch_loss_mode == "listwise":
            return per_example.mean()

        positive = scores.gather(
            1, target_position[:, None]).squeeze(1)
        negative_valid = candidate_valid.clone()
        negative_valid.scatter_(
            1, target_position[:, None], False)

        if self.item_arch_loss_mode == "top_rank":
            topk = min(
                self.item_arch_rank_topk,
                max(1, scores.shape[1] - 1),
            )
            negative_scores = scores.masked_fill(
                ~negative_valid, -torch.inf)
            top_negative, top_position = torch.topk(
                negative_scores, k=topk, dim=-1)
            top_valid = negative_valid.gather(1, top_position)
            margin = torch.full_like(
                top_negative, self.item_arch_rank_margin)
            if self.item_arch_rank_overlap_weight > 0:
                selected_index = candidate_index.gather(
                    1, top_position)
                selected_tuple = self.valid_item_tuples[selected_index]
                gold_tuple = self.valid_item_tuples[
                    target_index][:, None, :]
                overlap = selected_tuple.eq(
                    gold_tuple).to(torch.float32).mean(dim=-1)
                margin = (
                    margin
                    + self.item_arch_rank_overlap_weight * overlap
                )
            pair_logits = (
                top_negative - positive[:, None] + margin
            ) / self.item_arch_rank_temperature
            pair_logits = pair_logits.masked_fill(
                ~top_valid, -torch.inf)
            # log(1 + sum exp(s_neg - s_pos + margin)) is a smooth,
            # top-focused rank loss with a finite zero anchor.
            rank_loss = torch.logsumexp(
                torch.cat(
                    [
                        torch.zeros_like(pair_logits[:, :1]),
                        pair_logits,
                    ],
                    dim=1,
                ),
                dim=1,
            )
            per_example = (
                per_example + self.item_arch_rank_weight * rank_loss)
            return per_example.mean()

        # Stable differentiable Recall/NDCG@5 proxy over the hardest local
        # negatives. CE remains the coarse ranking term; restricting soft
        # rank to top-8 keeps initialization near the @5 gradient region.
        negative_scores = scores.masked_fill(
            ~negative_valid, -torch.inf)
        metric_topk = min(
            self.item_arch_rank_topk,
            max(1, scores.shape[1] - 1),
        )
        metric_negative, metric_position = torch.topk(
            negative_scores, k=metric_topk, dim=-1)
        metric_valid = negative_valid.gather(
            1, metric_position)
        pair_probability = torch.sigmoid(
            (metric_negative - positive[:, None])
            / self.item_arch_rank_temperature
        ).masked_fill(~metric_valid, 0.0)
        soft_rank = 1.0 + pair_probability.sum(dim=-1)
        soft_recall_at5 = torch.sigmoid(
            (5.5 - soft_rank) / 0.5)
        soft_discount = 1.0 / torch.log2(1.0 + soft_rank)
        metric_penalty = 1.0 - (
            soft_recall_at5 * soft_discount)
        per_example = (
            per_example
            + self.item_arch_metric_weight * metric_penalty
        )
        return per_example.mean()

    def _item_arch_lattice_loss(
        self,
        scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_index: torch.Tensor,
        target_index: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise mass on arbitrary multi-digit SID item groups.

        Each pair/triple of corrupted SID positions defines a target group:
        all candidates matching the gold item on that subset.  Optimizing the
        posterior mass of these groups teaches genuinely joint item
        structure while retaining uncertainty among items not yet
        distinguishable at the current denoising state.
        """
        zero = scores[torch.isfinite(scores)].sum() * 0.0
        if (self.item_arch_lattice_mode == "off"
                or self.item_arch_lattice_w <= 0):
            return zero
        h = self.tokenizer.n_digit
        structural_heads = (
            self.item_arch_lattice_semantic_heads or h)
        max_order = min(
            structural_heads, self.item_arch_lattice_max_order)
        if self.item_arch_lattice_mode == "prefix":
            subsets = [
                tuple(range(order))
                for order in range(2, max_order + 1)
            ]
        else:
            subsets = [
                subset
                for order in range(2, max_order + 1)
                for subset in combinations(range(structural_heads), order)
            ]
        if len(subsets) > self.item_arch_lattice_max_subsets:
            # Keep the objective bounded for long SIDs (H17 has hundreds of
            # pair/triple subsets). Evenly spaced deterministic sampling
            # covers the full lexicographic lattice without seed noise.
            n_subsets = len(subsets)
            subsets = [
                subsets[
                    min(
                        n_subsets - 1,
                        int(index * n_subsets
                            / self.item_arch_lattice_max_subsets),
                    )
                ]
                for index in range(self.item_arch_lattice_max_subsets)
            ]
        candidate_tuple = self.valid_item_tuples[candidate_index]
        gold_tuple = self.valid_item_tuples[target_index]
        log_partition = torch.logsumexp(scores, dim=-1)
        pool_count = candidate_valid.sum(
            dim=-1).to(torch.float32).clamp_min(1)
        weighted_loss = scores.new_zeros(scores.shape[0])
        total_weight = scores.new_zeros(scores.shape[0])
        active_terms = torch.zeros(
            (), device=scores.device, dtype=torch.long)
        for subset in subsets:
            subset_index = torch.as_tensor(
                subset,
                device=scores.device,
                dtype=torch.long,
            )
            corrupted = masked_indices[
                :, subset_index].all(dim=-1)
            group = (
                candidate_tuple[:, :, subset_index].eq(
                    gold_tuple[:, None, subset_index])
                .all(dim=-1)
                & candidate_valid
            )
            group_count = group.sum(
                dim=-1).to(torch.float32).clamp_min(1)
            information = (
                pool_count / group_count).log()
            active = (
                corrupted
                & information.gt(1e-6)
                & group_count.ge(self.item_arch_lattice_min_group)
            )
            if self.item_arch_lattice_max_group > 0:
                active = (
                    active
                    & group_count.le(self.item_arch_lattice_max_group)
                )
            group_log_mass = torch.logsumexp(
                scores.masked_fill(~group, -torch.inf),
                dim=-1,
            )
            subset_loss = log_partition - group_log_mass
            if self.item_arch_lattice_weight_mode == "frontier":
                # Bounded information-bottleneck weight. It is zero at a
                # singleton/exact-item group and at the unreduced pool, and
                # peaks when the subset resolves the pool to sqrt(|C|).
                pool_log = pool_count.log().clamp_min(1e-6)
                structural_weight = (
                    4.0
                    * group_count.log()
                    * information
                    / pool_log.square()
                ).clamp(min=0.0, max=1.0)
            else:
                structural_weight = information
            weight = (
                structural_weight
                * active.to(structural_weight.dtype)
            )
            weighted_loss = weighted_loss + weight * subset_loss
            total_weight = total_weight + weight
            active_terms = active_terms + active.sum()
        active_rows = total_weight.gt(0)
        if not self._logged_item_arch_lattice:
            self.log(
                "item architecture lattice: mode={} max_order={} "
                "semantic_heads={} group={}/{} weight={} "
                "active_rows={}/{} active_terms={}".format(
                    self.item_arch_lattice_mode,
                    max_order,
                    structural_heads,
                    self.item_arch_lattice_min_group,
                    self.item_arch_lattice_max_group,
                    self.item_arch_lattice_weight_mode,
                    int(active_rows.sum().item()),
                    active_rows.numel(),
                    int(active_terms.item()),
                ))
            self._logged_item_arch_lattice = True
        if not active_rows.any():
            return zero
        return (
            weighted_loss[active_rows]
            / total_weight[active_rows].clamp_min(1e-12)
        ).mean()

    def _item_arch_token_marginal(
        self,
        candidate_scores: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_valid: torch.Tensor,
        token_logits: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project item evidence to tokens by posterior mass or max support."""
        candidate_tuple = self.valid_item_tuples[candidate_index]
        h = self.tokenizer.n_digit
        if self.item_arch_projection_mode == "max_score":
            # A generic best-completion control: each code is supported only
            # by its highest-scoring compatible complete item. Candidate
            # scores already include the configured item temperature; a
            # softmax over code-level maxima yields the token distribution
            # consumed by the unchanged residual feedback interface.
            marginal = torch.full(
                (
                    token_logits.shape[0],
                    h,
                    token_logits.shape[-1],
                ),
                -torch.inf,
                device=token_logits.device,
                dtype=torch.float32,
            )
            valid_scores = candidate_scores.masked_fill(
                ~candidate_valid, -torch.inf)
            for digit in range(h):
                marginal[:, digit, :].scatter_reduce_(
                    1,
                    candidate_tuple[:, :, digit],
                    valid_scores,
                    reduce="amax",
                    include_self=True,
                )
            return F.softmax(marginal, dim=-1)
        if self.item_arch_belief_mode == "independent":
            posterior = F.softmax(candidate_scores, dim=-1)
            posterior = posterior.masked_fill(
                ~candidate_valid, 0.0)
            digit_posterior = posterior[:, None, :].expand(-1, h, -1)
        else:
            token_log_probability = F.log_softmax(
                token_logits.to(torch.float32), dim=-1)
            token_evidence = torch.stack(
                [
                    token_log_probability[:, digit, :].gather(
                        1, candidate_tuple[:, :, digit])
                    for digit in range(h)
                ],
                dim=1,
            )
            mask = masked_indices.to(torch.float32)
            masked_evidence = (
                token_evidence * mask[:, :, None])
            evidence_sum = masked_evidence.sum(dim=1)
            masked_count = mask.sum(
                dim=1, keepdim=True)
            if self.item_arch_belief_mode == "poe":
                normalized_evidence = (
                    evidence_sum
                    / masked_count.clamp_min(1.0)
                )
                belief_scores = (
                    candidate_scores
                    + self.item_arch_belief_weight
                    * normalized_evidence
                )
                posterior = F.softmax(belief_scores, dim=-1)
                posterior = posterior.masked_fill(
                    ~candidate_valid, 0.0)
                digit_posterior = posterior[
                    :, None, :].expand(-1, h, -1)
            else:
                # Leave digit j out of the item belief used to update j.
                # This is a cavity message on the catalog-incidence star and
                # prevents a token prediction from being echoed into itself.
                cavity_sum = (
                    evidence_sum[:, None, :]
                    - masked_evidence
                )
                cavity_count = (
                    masked_count[:, None, :]
                    - mask[:, :, None]
                )
                cavity_evidence = torch.where(
                    cavity_count.gt(0),
                    cavity_sum / cavity_count.clamp_min(1.0),
                    torch.zeros_like(cavity_sum),
                )
                belief_scores = (
                    candidate_scores[:, None, :]
                    + self.item_arch_belief_weight
                    * cavity_evidence
                )
                belief_scores = belief_scores.masked_fill(
                    ~candidate_valid[:, None, :], -torch.inf)
                digit_posterior = F.softmax(
                    belief_scores, dim=-1)
                digit_posterior = digit_posterior.masked_fill(
                    ~candidate_valid[:, None, :], 0.0)

        marginal = torch.zeros(
            (
                token_logits.shape[0],
                h,
                token_logits.shape[-1],
            ),
            device=token_logits.device,
            dtype=torch.float32,
        )
        for digit in range(h):
            marginal[:, digit, :].scatter_add_(
                1,
                candidate_tuple[:, :, digit],
                digit_posterior[:, digit, :],
            )
        return marginal

    def _item_arch_gate_values(
        self,
        target_hidden: torch.Tensor,
        token_logits: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_count: torch.Tensor,
        masked_indices: torch.Tensor,
        decode: bool,
    ) -> torch.Tensor:
        """Return a per-digit residual gate from current catalog ambiguity."""
        if self.item_arch_gate is None:
            raise RuntimeError("item architecture gate was not built")
        posterior = F.softmax(candidate_scores, dim=-1)
        posterior = posterior.masked_fill(~candidate_valid, 0.0)
        item_entropy = -(
            posterior
            * posterior.clamp_min(1e-12).log()).sum(dim=-1)
        selected_count = candidate_valid.sum(dim=-1).clamp_min(1)
        item_entropy = item_entropy / selected_count.to(
            torch.float32).clamp_min(2).log()
        top_probability = torch.topk(
            posterior,
            k=min(2, posterior.shape[1]),
            dim=-1,
        ).values
        if top_probability.shape[1] == 1:
            item_margin = top_probability[:, 0]
        else:
            item_margin = (
                top_probability[:, 0] - top_probability[:, 1])
        token_probability = F.softmax(
            token_logits.to(torch.float32), dim=-1)
        token_entropy = -(
            token_probability
            * token_probability.clamp_min(1e-12).log()).sum(dim=-1)
        token_entropy = token_entropy / math.log(token_logits.shape[-1])
        catalog_log_size = math.log(
            max(2, self.valid_item_tuples.shape[0]))
        ambiguity = (
            candidate_count.to(torch.float32).clamp_min(1).log()
            / catalog_log_size)
        noise_fraction = masked_indices.to(torch.float32).mean(
            dim=-1)
        shared = torch.stack(
            [
                ambiguity,
                item_entropy,
                item_margin,
                noise_fraction,
            ],
            dim=-1,
        )
        shared = shared[:, None, :].expand(
            -1, target_hidden.shape[1], -1)
        gate_features = torch.cat(
            [target_hidden, shared, token_entropy[:, :, None]],
            dim=-1,
        )

        if self.item_arch_gate_mode == "fixed":
            base_gate = torch.ones_like(token_entropy)
        elif self.item_arch_gate_mode == "ambiguity":
            # Item evidence becomes stronger only after revealed SID digits
            # have reduced the compatible bucket.
            base_gate = (
                1.0 - ambiguity[:, None]).expand_as(token_entropy)
        else:
            # Signed zero-centered routing permits the model to suppress an
            # unreliable item expert while remaining an exact no-op at init.
            base_gate = torch.tanh(
                self.item_arch_gate(gate_features).squeeze(-1))

        # Decode-only causal controls for the learned router. These operate on
        # the checkpoint's native gate before the common feedback strength and
        # mask are applied. Consequently mean_matched holds the signed residual
        # scale fixed call-by-call, while shuffled holds the entire empirical
        # gate distribution fixed.
        native_gate = base_gate
        if decode and self.item_arch_gate_inference_mode != "native":
            if self.item_arch_gate_mode != "learned":
                raise RuntimeError(
                    "gate inference interventions require a learned gate")
            active_native = native_gate[masked_indices]
            if self.item_arch_gate_inference_mode == "off":
                base_gate = torch.zeros_like(native_gate)
            elif self.item_arch_gate_inference_mode == "mean_matched":
                mean_gate = (
                    active_native.mean()
                    if active_native.numel() else native_gate.new_zeros(()))
                base_gate = mean_gate.expand_as(native_gate)
            elif self.item_arch_gate_inference_mode == "shuffled":
                base_gate = native_gate.clone()
                if active_native.numel() > 1:
                    shift = max(1, active_native.numel() // 2)
                    base_gate[masked_indices] = torch.roll(
                        active_native, shifts=shift, dims=0)

        strength = self.item_arch_feedback_strength
        if decode and self.item_arch_decode_weight >= 0:
            strength = self.item_arch_decode_weight
        if self.item_arch_gate_mode in {"fixed", "ambiguity"}:
            if self.item_arch_residual_alpha is None:
                raise RuntimeError(
                    "item architecture residual scalar was not built")
            base_gate = (
                base_gate * torch.tanh(
                    self.item_arch_residual_alpha))
        if self.item_arch_feedback_max_bucket > 0:
            # Candidate feedback is exact only once the compatible bucket
            # fits inside the inference-time top-K pool. In particular this
            # suppresses all-mask self-reinforcement when natural gold
            # coverage is still incomplete.
            base_gate = (
                base_gate
                * candidate_count.le(
                    self.item_arch_feedback_max_bucket
                )[:, None].to(base_gate.dtype)
            )
        if decode:
            active_native = native_gate[masked_indices].detach()
            active_used = base_gate[masked_indices].detach()
            audit = self._item_arch_gate_audit
            audit["calls"] += 1
            audit["count"] += int(active_native.numel())
            audit["native_sum"] += float(active_native.sum().item())
            audit["native_abs_sum"] += float(
                active_native.abs().sum().item())
            audit["native_negative_count"] += int(
                active_native.lt(0).sum().item())
            audit["used_sum"] += float(active_used.sum().item())
            audit["used_abs_sum"] += float(active_used.abs().sum().item())
            audit["used_negative_count"] += int(
                active_used.lt(0).sum().item())
        return (
            strength
            * base_gate
            * masked_indices.to(base_gate.dtype)
        )

    def _item_architecture_forward(
        self,
        target_hidden: torch.Tensor,
        token_logits: torch.Tensor,
        target_ids: torch.Tensor,
        masked_indices: torch.Tensor,
        target_index: Optional[torch.Tensor] = None,
        history_ids: Optional[torch.Tensor] = None,
        decode: bool = False,
    ) -> ItemArchitectureOutput:
        """Run joint item energy and optional item-to-token feedback."""
        if self.item_arch_mode == "off":
            raise RuntimeError(
                "item architecture forward called while disabled")
        (
            candidate_index,
            candidate_valid,
            candidate_count,
            _,
        ) = self._item_arch_candidate_indices(
            token_logits,
            target_ids,
        )
        item_query = self._item_arch_query(
            target_hidden,
            history_ids=history_ids,
        )
        score_active_heads = masked_indices
        if decode and self.item_arch_inference_evidence == "static":
            if self._item_arch_static_query is None:
                if not masked_indices.all():
                    raise RuntimeError(
                        "static item evidence must initialize at the "
                        "all-mask state")
                self._item_arch_static_query = item_query.detach().clone()
                self._item_arch_static_query_initializations += 1
            current_query_shape = item_query.shape
            cached_query = self._item_arch_static_query
            if item_query.shape[0] % cached_query.shape[0] != 0:
                raise RuntimeError(
                    "static item query cannot be aligned with the current "
                    "beam rows: cached={} current={}".format(
                        tuple(cached_query.shape), tuple(item_query.shape)))
            repeat_factor = item_query.shape[0] // cached_query.shape[0]
            item_query = cached_query.repeat_interleave(
                repeat_factor, dim=0)
            if item_query.shape != current_query_shape:
                raise RuntimeError(
                    "expanded static item query does not match decode rows")
            # Freeze the scoring view as well as the query: otherwise
            # head-attention variants could change relative item ranks solely
            # because a different subset of SID heads became resolved.
            score_active_heads = torch.ones_like(masked_indices)
        candidate_scores = self._item_arch_candidate_scores(
            item_query,
            candidate_index,
            candidate_valid,
            active_heads=score_active_heads,
            history_ids=history_ids,
        )
        selected_count = candidate_valid.sum(
            dim=-1).to(torch.float32)
        zero = candidate_scores[
            torch.isfinite(candidate_scores)].sum() * 0.0
        item_loss = zero
        top1_accuracy = zero.detach()
        gold_coverage = None
        if target_index is not None:
            # Feedback candidates are selected without consulting the label,
            # exactly as at inference. Gold injection is confined to a
            # separate listwise-loss pool to avoid target leakage.
            target_match = candidate_index.eq(target_index[:, None])
            gold_coverage = target_match.any(dim=-1)
            loss_candidate_index = candidate_index
            loss_candidate_valid = candidate_valid
            loss_candidate_scores = candidate_scores
            if not gold_coverage.all():
                missing = ~gold_coverage
                loss_candidate_index = candidate_index.clone()
                loss_candidate_valid = candidate_valid.clone()
                loss_candidate_index[missing, -1] = target_index[missing]
                loss_candidate_valid[missing, -1] = True
                loss_candidate_scores = self._item_arch_candidate_scores(
                    item_query,
                    loss_candidate_index,
                    loss_candidate_valid,
                    active_heads=masked_indices,
                    history_ids=history_ids,
                )
                target_match = loss_candidate_index.eq(
                    target_index[:, None])
            target_position = target_match.to(
                torch.long).argmax(dim=-1)
            eligible = (
                candidate_count.ge(self.item_arch_loss_min_bucket)
                & loss_candidate_valid.sum(dim=-1).gt(1))
            if self.item_arch_loss_max_bucket > 0:
                eligible = (
                    eligible
                    & candidate_count.le(
                        self.item_arch_loss_max_bucket)
                )
            if not self._logged_item_arch_frontier:
                self.log(
                    "item loss frontier: bucket={}/{} eligible={}/{} "
                    "mean_eligible_bucket={:.2f}".format(
                        self.item_arch_loss_min_bucket,
                        self.item_arch_loss_max_bucket,
                        int(eligible.sum().item()),
                        eligible.numel(),
                        (
                            candidate_count[eligible]
                            .to(torch.float32).mean().item()
                            if eligible.any() else float("nan")
                        ),
                    ))
                self._logged_item_arch_frontier = True
            if eligible.any():
                direct_item_loss = self._item_arch_ranking_loss(
                    loss_candidate_scores[eligible],
                    loss_candidate_valid[eligible],
                    loss_candidate_index[eligible],
                    target_index[eligible],
                    target_position[eligible],
                    masked_indices[eligible],
                )
                if (self.item_arch_lattice_mode != "off"
                        and self.item_arch_lattice_w > 0):
                    direct_item_loss = (
                        direct_item_loss
                        + self.item_arch_lattice_w
                        * self._item_arch_lattice_loss(
                            loss_candidate_scores[eligible],
                            loss_candidate_valid[eligible],
                            loss_candidate_index[eligible],
                            target_index[eligible],
                            masked_indices[eligible],
                        )
                    )
                if self.item_arch_mode == "joint_energy":
                    if self.item_arch_residual_alpha is None:
                        raise RuntimeError(
                            "joint item residual scalar was not built")
                    # The direct term trains useful item representations from
                    # step one. The calibrated term trains the zero-start
                    # residual scale that will be used by CCPF completion,
                    # while preserving an exactly neutral epoch-0 decoder.
                    calibrated_item_loss = F.cross_entropy(
                        self._item_arch_masked_scale(
                            loss_candidate_scores[eligible],
                            loss_candidate_valid[eligible],
                            torch.tanh(
                                self.item_arch_residual_alpha),
                        ),
                        target_position[eligible],
                    )
                    item_loss = 0.5 * (
                        direct_item_loss + calibrated_item_loss)
                else:
                    item_loss = direct_item_loss
                top1_accuracy = loss_candidate_scores[
                    eligible].argmax(dim=-1).eq(
                        target_position[eligible]).to(
                            torch.float32).mean()
        if not self._logged_item_arch_candidates:
            coverage = (
                gold_coverage.to(torch.float32).mean().item()
                if gold_coverage is not None else float("nan"))
            self.log(
                "item architecture candidates: compatible_only={} "
                "mean_catalog_bucket={:.2f} mean_selected={:.2f} "
                "max_selected={} preinject_gold_coverage={:.4f}".format(
                    self.item_arch_compatible_only,
                    candidate_count.to(torch.float32).mean().item(),
                    selected_count.mean().item(),
                    int(selected_count.max().item()),
                    coverage,
                ))
            self._logged_item_arch_candidates = True

        fused_logits = token_logits
        gate = torch.zeros_like(
            token_logits[:, :, 0], dtype=torch.float32)
        if self.item_arch_mode == "epic" and masked_indices.any():
            marginal = self._item_arch_token_marginal(
                candidate_scores,
                candidate_index,
                candidate_valid,
                token_logits,
                masked_indices,
            )
            item_log_probability = marginal.clamp_min(1e-12).log()
            token_log_probability = F.log_softmax(
                token_logits.detach().to(torch.float32), dim=-1)
            residual = (
                item_log_probability - token_log_probability).clamp(
                    min=-self.item_arch_feedback_clip,
                    max=self.item_arch_feedback_clip,
                )
            gate = self._item_arch_gate_values(
                target_hidden,
                token_logits,
                candidate_scores,
                candidate_valid,
                candidate_count,
                masked_indices,
                decode,
            )
            fused_logits = (
                token_logits.to(torch.float32)
                + gate[:, :, None] * residual
            ).to(token_logits.dtype)
            if not self._logged_item_arch_gate:
                self.log(
                    "item architecture gate: mode={} decode={} "
                    "signed_mean={:.6f} abs_mean={:.6f} "
                    "max_abs={:.6f}".format(
                        self.item_arch_gate_mode,
                        decode,
                        gate.mean().item(),
                        gate.abs().mean().item(),
                        gate.abs().max().item(),
                    ))
                self._logged_item_arch_gate = True

        return ItemArchitectureOutput(
            logits=fused_logits,
            loss=item_loss,
            gate_mean=gate.abs().mean(),
            candidate_count=candidate_count.to(
                torch.float32).mean(),
            selected_count=selected_count.mean(),
            top1_accuracy=top1_accuracy,
        )

    def _item_arch_catalog_rerank(
        self,
        outputs: CausalLMOutputWithPast,
        input_ids: torch.Tensor,
        token_logits: torch.Tensor,
        catalog_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Add joint item energy before subset/CCPF catalog top-k pruning."""
        if self.item_arch_mode == "off":
            return catalog_scores
        if outputs.hidden_states is None:
            raise RuntimeError(
                "joint item reranking requires transformer hidden states")
        if (self.item_arch_mode == "joint_energy"
                and self.item_arch_residual_alpha is None):
            raise RuntimeError(
                "joint item residual scalar was not built")
        h = self.tokenizer.n_digit
        target_ids = input_ids[:, -h:]
        (
            candidate_index,
            candidate_valid,
            _,
            _,
        ) = self._item_arch_candidate_indices(
            token_logits,
            target_ids,
        )
        item_query = self._item_arch_query(
            outputs.hidden_states[-1][:, -h:, :],
            history_ids=input_ids[:, :-h],
        )
        item_scores = self._item_arch_candidate_scores(
            item_query,
            candidate_index,
            candidate_valid,
            active_heads=(
                target_ids.eq(self.mask_token_id)
                | target_ids.eq(0)
            ),
            history_ids=input_ids[:, :-h],
        )
        finite_scores = item_scores.masked_fill(
            ~candidate_valid, 0.0)
        count = candidate_valid.sum(
            dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
        mean = finite_scores.sum(dim=-1, keepdim=True) / count
        centered = (finite_scores - mean).masked_fill(
            ~candidate_valid, 0.0)
        variance = centered.square().sum(
            dim=-1, keepdim=True) / count
        normalized = centered / variance.sqrt().clamp_min(1e-6)
        decode_weight = (
            self.item_arch_feedback_strength
            if self.item_arch_decode_weight < 0
            else self.item_arch_decode_weight)
        if self.item_arch_mode == "joint_energy":
            residual_scale = (
                decode_weight * torch.tanh(
                    self.item_arch_residual_alpha))
        elif self.item_arch_inference_timing == "posthoc":
            # EPIC's same trained item scorer, applied once after token
            # generation rather than repeatedly inside the trajectory.
            residual_scale = torch.as_tensor(
                decode_weight, device=catalog_scores.device,
                dtype=torch.float32)
        else:
            return catalog_scores
        reranked = catalog_scores.clone()
        reranked.scatter_add_(
            1,
            candidate_index,
            residual_scale * normalized,
        )
        self._item_arch_catalog_rerank_calls += 1
        if not self._logged_item_arch_rerank:
            self.log(
                "item catalog rerank: mode={} decode_weight={:.4f} "
                "learned_scale={:.6f} mean_candidates={:.2f}".format(
                    self.item_arch_mode,
                    decode_weight,
                    residual_scale.item(),
                    candidate_valid.sum(
                        dim=-1).to(torch.float32).mean().item(),
                ))
            self._logged_item_arch_rerank = True
        return reranked

    def _item_arch_generation_logits(
        self,
        outputs: CausalLMOutputWithPast,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply trained EPIC feedback to the final target during decoding."""
        if self.item_arch_mode != "epic":
            return logits
        h = self.tokenizer.n_digit
        target_ids = input_ids[:, -h:]
        masked_indices = target_ids.eq(self.mask_token_id)
        if not masked_indices.any():
            return logits
        if self.item_arch_inference_timing != "inline":
            raise RuntimeError(
                "token-level item feedback is forbidden when "
                "item_arch_inference_timing=posthoc")
        self._item_arch_generation_feedback_calls += 1
        if outputs.hidden_states is None:
            raise RuntimeError(
                "EPIC decoding requires transformer hidden states")
        item_output = self._item_architecture_forward(
            outputs.hidden_states[-1][:, -h:, :],
            logits[:, -h:, :],
            target_ids,
            masked_indices,
            history_ids=input_ids[:, :-h],
            decode=True,
        )
        logits = logits.clone()
        logits[:, -h:, :] = item_output.logits
        return logits

    def _catalog_item_classification_loss(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Classify items only from corrupted SID positions.

        Visible target digits are deliberately excluded so the auxiliary
        objective cannot reduce to copying an unmasked code.
        """
        temperature = self.catalog_item_loss_temperature
        if self.catalog_item_loss_mode == "inbatch":
            unique_target, inverse = torch.unique(
                target_index, sorted=True, return_inverse=True)
            candidates = self.valid_item_tuples[unique_target]
            scores = self._catalog_tuple_scores(
                logits,
                candidates,
                position_mask=masked_indices,
                normalize_positions=True,
            ) / temperature
            return F.cross_entropy(scores, inverse)

        scores = self._catalog_tuple_scores(
            logits,
            position_mask=masked_indices,
            normalize_positions=True,
        ) / temperature
        if self.catalog_item_loss_mode in {
                "conditional", "conditional_hard", "stage_adaptive"
        }:
            # Visible SID digits are already part of the denoiser input.
            # Restrict competition to catalog items agreeing with that
            # evidence, then rank using only corrupted positions.
            target_tuples = self.valid_item_tuples[target_index]
            visible_match = self.valid_item_tuples[
                None, :, :].eq(target_tuples[:, None, :])
            compatible = (
                visible_match | masked_indices[:, None, :]
            ).all(dim=-1)
            candidate_count = compatible.sum(dim=-1)
            masked_count = masked_indices.sum(dim=-1)
            if self.catalog_item_loss_mode == "stage_adaptive":
                # The transition stage trains item discrimination only once
                # two or three SID digits must be recovered. Fully masked
                # targets have no visible evidence, so they receive a global
                # hard-negative objective instead. Averaging the two stage
                # means prevents the more frequent noise level from
                # dominating the auxiliary loss.
                transition_eligible = (
                    masked_count.ge(2)
                    & masked_count.le(3)
                    & masked_count.lt(self.tokenizer.n_digit)
                    & candidate_count.gt(1)
                )
                global_eligible = masked_count.eq(
                    self.tokenizer.n_digit)
                if not self._logged_catalog_item_stats:
                    self.log(
                        "stage-adaptive item loss: transition={}/{} "
                        "global={}/{} mean_candidates={:.2f} "
                        "mean_masked={:.2f}".format(
                            int(transition_eligible.sum().item()),
                            transition_eligible.numel(),
                            int(global_eligible.sum().item()),
                            global_eligible.numel(),
                            candidate_count.to(torch.float32).mean().item(),
                            masked_count.to(torch.float32).mean().item(),
                        ))
                    self._logged_catalog_item_stats = True

                stage_losses = []
                if transition_eligible.any():
                    transition_scores = scores[transition_eligible]
                    transition_scores = transition_scores.masked_fill(
                        ~compatible[transition_eligible], -torch.inf)
                    stage_losses.append(
                        self._focal_catalog_classification_loss(
                            transition_scores,
                            target_index[transition_eligible],
                        ))
                if global_eligible.any():
                    stage_losses.append(
                        self._hard_catalog_classification_loss(
                            scores[global_eligible],
                            target_index[global_eligible],
                        ))
                if not stage_losses:
                    return scores.sum() * 0.0
                return torch.stack(stage_losses).mean()

            eligible = (
                masked_count.ge(self.catalog_item_loss_min_masked)
                & candidate_count.gt(1)
            )
            if self.catalog_item_loss_max_masked > 0:
                eligible = (
                    eligible
                    & masked_count.le(self.catalog_item_loss_max_masked)
                )
            if not self._logged_catalog_item_stats:
                self.log(
                    "conditional item loss: eligible={}/{} "
                    "mean_candidates={:.2f} mean_masked={:.2f}".format(
                        int(eligible.sum().item()),
                        eligible.numel(),
                        candidate_count.to(torch.float32).mean().item(),
                        masked_count.to(torch.float32).mean().item(),
                    ))
                self._logged_catalog_item_stats = True
            if not eligible.any():
                return scores.sum() * 0.0
            scores = scores[eligible]
            target_index = target_index[eligible]
            compatible = compatible[eligible]
            scores = scores.masked_fill(~compatible, -torch.inf)
            if self.catalog_item_loss_mode == "conditional":
                return self._focal_catalog_classification_loss(
                    scores, target_index)

        if self.catalog_item_loss_mode == "full":
            return F.cross_entropy(scores, target_index)

        return self._hard_catalog_classification_loss(scores, target_index)

    def _focal_catalog_classification_loss(
        self,
        scores: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        """Mean catalog cross entropy with optional focal reweighting."""
        per_example = F.cross_entropy(
            scores, target_index, reduction="none")
        if self.catalog_item_loss_focal_gamma > 0:
            target_probability = torch.exp(-per_example)
            per_example = per_example * (
                1.0 - target_probability
            ).pow(self.catalog_item_loss_focal_gamma)
        return per_example.mean()

    def _hard_catalog_classification_loss(
        self,
        scores: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        """Cross entropy against the target and its highest-scoring negatives."""
        topk = min(
            self.catalog_item_loss_topk - 1,
            self.valid_item_tuples.shape[0] - 1,
        )
        negative_scores = scores.detach().clone()
        negative_scores.scatter_(
            1, target_index[:, None], -torch.inf)
        negative_index = torch.topk(
            negative_scores, k=topk, dim=-1).indices
        candidate_index = torch.cat(
            [target_index[:, None], negative_index], dim=1)
        candidate_scores = scores.gather(1, candidate_index)
        labels = torch.zeros(
            scores.shape[0],
            device=scores.device,
            dtype=torch.long,
        )
        return F.cross_entropy(candidate_scores, labels)

    def _catalog_decode_position_weights(
        self,
        logits: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return mean-one reliability weights for all-mask SID digits."""
        mode = self.catalog_decode_digit_weight
        if mode == "uniform":
            return None
        batch_size = logits.shape[0]
        information = self.catalog_information_gain[
            None, :].expand(batch_size, -1)
        if mode == "catalog_info":
            return information
        probability = F.softmax(logits.to(torch.float32), dim=-1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(
            dim=-1)
        confidence = (
            1.0 - entropy / math.log(logits.shape[-1])
        ).clamp_min(0.05)
        weights = (
            confidence
            if mode == "confidence"
            else confidence * information)
        return weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _normalize_catalog_fusion_scores(
        scores: torch.Tensor,
        mode: str,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize catalog scores, optionally over a per-query valid set.

        The default ``valid_mask=None`` path is intentionally identical to
        the original late-reranking implementation. Preselection fusion uses
        the mask so incompatible items remain at negative infinity under
        z-score and rank normalization instead of re-entering the catalog.
        """
        if valid_mask is not None:
            if valid_mask.shape != scores.shape:
                raise ValueError(
                    "catalog score valid_mask must match scores")
            valid_mask = valid_mask & torch.isfinite(scores)
        if mode == "none":
            return (
                scores
                if valid_mask is None
                else scores.masked_fill(~valid_mask, -torch.inf)
            )
        if mode == "zscore":
            if valid_mask is not None:
                count = valid_mask.sum(
                    dim=-1, keepdim=True).clamp_min(1)
                safe_scores = scores.masked_fill(~valid_mask, 0.0)
                mean = safe_scores.sum(
                    dim=-1, keepdim=True) / count.to(scores.dtype)
                centered = (scores - mean).masked_fill(~valid_mask, 0.0)
                variance = centered.square().sum(
                    dim=-1, keepdim=True) / count.to(scores.dtype)
                normalized = centered / variance.sqrt().clamp_min(1e-6)
                return normalized.masked_fill(
                    ~valid_mask, -torch.inf)
            return (
                (scores - scores.mean(dim=-1, keepdim=True))
                / scores.std(
                    dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            )
        ranked_scores = (
            scores
            if valid_mask is None
            else scores.masked_fill(~valid_mask, -torch.inf)
        )
        order = torch.argsort(ranked_scores, dim=-1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            scores.shape[-1],
            device=scores.device,
            dtype=order.dtype,
        )[None, :].expand_as(order)
        ranks.scatter_(1, order, rank_values)
        if valid_mask is None:
            denominator = max(1, scores.shape[-1] - 1)
            return -ranks.to(scores.dtype) / denominator
        denominator = (
            valid_mask.sum(dim=-1, keepdim=True) - 1
        ).clamp_min(1).to(scores.dtype)
        normalized = -ranks.to(scores.dtype) / denominator
        return normalized.masked_fill(~valid_mask, -torch.inf)

    def _subset_catalog_preselection_scores(
        self,
        item_scores: torch.Tensor,
        guide_logits: torch.Tensor,
        guide_position_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse both experts over every compatible completion before top-k."""
        compatible_items = torch.isfinite(item_scores)
        guide_scores = self._catalog_tuple_scores(
            guide_logits,
            position_weight=guide_position_weight,
        )
        completion_component = self._normalize_catalog_fusion_scores(
            item_scores,
            self.catalog_score_normalization,
            valid_mask=compatible_items,
        )
        guide_component = self._normalize_catalog_fusion_scores(
            guide_scores,
            self.catalog_score_normalization,
            valid_mask=compatible_items,
        )
        query_weight = self._catalog_decode_query_weights(
            guide_logits,
            completion_component,
            guide_component,
        )
        return completion_component + query_weight * guide_component

    def _catalog_decode_query_weights(
        self,
        guide_logits: torch.Tensor,
        beam_scores: torch.Tensor,
        guide_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Choose a catalog-fusion weight independently for each query.

        The fixed mode exactly preserves the previous decoder. Adaptive modes
        use label-free reliability signals available at inference time:
        confidence of all-mask SID digits, entropy/margin over generated
        candidates, or confident disagreement between the beam and catalog
        experts. Returned weights have shape ``[batch, 1]``.
        """
        if self.catalog_decode_weight_mode == "fixed":
            return guide_scores.new_full(
                (guide_scores.shape[0], 1),
                self.catalog_decode_weight,
            )

        probability = F.softmax(guide_logits.to(torch.float32), dim=-1)
        digit_entropy = -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        digit_reliability = (
            1.0 - digit_entropy / math.log(guide_logits.shape[-1])
        ).clamp(0.0, 1.0).mean(dim=-1)

        mode = self.catalog_decode_weight_mode
        if mode == "digit_confidence":
            reliability = digit_reliability
        elif mode == "candidate_entropy":
            candidate_probability = F.softmax(
                guide_scores.to(torch.float32), dim=-1)
            candidate_entropy = -(
                candidate_probability
                * candidate_probability.clamp_min(1e-12).log()
            ).sum(dim=-1)
            reliability = (
                1.0
                - candidate_entropy
                / math.log(max(2, guide_scores.shape[-1]))
            ).clamp(0.0, 1.0)
        elif mode == "candidate_margin":
            candidate_probability = F.softmax(
                guide_scores.to(torch.float32), dim=-1)
            top_values = torch.topk(
                candidate_probability,
                k=min(2, candidate_probability.shape[-1]),
                dim=-1,
            ).values
            reliability = (
                top_values[:, 0] - top_values[:, -1]
            ).clamp(0.0, 1.0)
        elif mode == "disagreement_confidence":
            beam_choice = beam_scores.argmax(dim=-1)
            guide_choice = guide_scores.argmax(dim=-1)
            disagreement = (beam_choice != guide_choice).to(
                digit_reliability.dtype)
            # Retain a weak catalog correction under agreement and intervene
            # more strongly only when a confident catalog expert disagrees.
            reliability = digit_reliability * (0.5 + 0.5 * disagreement)
        else:
            beam_probability = F.softmax(
                beam_scores.to(torch.float32), dim=-1)
            guide_probability = F.softmax(
                guide_scores.to(torch.float32), dim=-1)
            beam_choice = beam_scores.argmax(dim=-1)
            guide_choice = guide_scores.argmax(dim=-1)
            disagreement = (beam_choice != guide_choice).to(
                digit_reliability.dtype)
            if mode == "selective_disagreement":
                # A smooth dead zone preserves the iterative beam whenever
                # the catalog expert agrees or its SID confidence is weak.
                confidence_gate = torch.sigmoid(
                    (
                        digit_reliability
                        - self.catalog_decode_confidence_threshold
                    ) / self.catalog_decode_confidence_temperature
                )
                reliability = disagreement * confidence_gate
            else:
                row = torch.arange(
                    guide_scores.shape[0], device=guide_scores.device)
                # Symmetric preference reversal: both experts must prefer
                # their own top item before a large catalog intervention.
                guide_advantage = (
                    guide_probability[row, guide_choice]
                    - guide_probability[row, beam_choice]
                ).clamp_min(0.0)
                beam_advantage = (
                    beam_probability[row, beam_choice]
                    - beam_probability[row, guide_choice]
                ).clamp_min(0.0)
                conflict = 0.5 * (guide_advantage + beam_advantage)
                conflict_gate = torch.sigmoid(
                    (
                        conflict - self.catalog_decode_conflict_threshold
                    ) / self.catalog_decode_conflict_temperature
                )
                reliability = (
                    disagreement * digit_reliability * conflict_gate
                )

        weights = (
            self.catalog_decode_weight_min
            + (
                self.catalog_decode_weight_max
                - self.catalog_decode_weight_min
            ) * reliability
        ).to(guide_scores.dtype)[:, None]
        if not self._logged_catalog_query_weight:
            self.log(
                "adaptive catalog weights: mode={} min={:.4f} "
                "mean={:.4f} max={:.4f}".format(
                    mode,
                    weights.min().item(),
                    weights.mean().item(),
                    weights.max().item(),
                ))
            self._logged_catalog_query_weight = True
        return weights

    def get_transfer_index(self, input_ids, logits, num_transfer_tokens):
        mask_index = (input_ids == self.mask_token_id)

        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = p.gather(index=x0.unsqueeze(-1), dim=-1).squeeze(-1)

        x0 = torch.where(mask_index, x0, input_ids)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        _, transfer_index = torch.topk(confidence,
                                       k=num_transfer_tokens,
                                       dim=-1)

        return transfer_index

    def _catalog_reduction_transfer_index(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a confident digit that maximally shrinks its catalog bucket.

        For every masked SID digit, the model's highest-probability token
        supported by the current compatible catalog is assigned utility
        ``p(token) * log(current_bucket / resulting_bucket)``. The returned
        bucket sizes are diagnostics for the selected digit and token; beam
        expansion itself remains the existing tuple-constrained step.
        """
        h = self.tokenizer.n_digit
        target = input_ids[:, -h:]
        masked = target.eq(self.mask_token_id)
        if not masked.any(dim=-1).all():
            raise RuntimeError(
                "catalog-reduction transfer requires a masked SID digit")

        visible = ~masked
        compatible = (
            target[:, None, :].eq(self.valid_item_tuples[None, :, :])
            | ~visible[:, None, :]
        ).all(dim=-1)
        bucket_count = compatible.sum(dim=-1)
        if not bucket_count.gt(0).all():
            raise RuntimeError(
                "catalog-reduction transfer found an empty catalog bucket")

        target_probability = F.softmax(
            logits[:, -h:, :].to(torch.float32), dim=-1)
        utility = target_probability.new_full(
            (input_ids.shape[0], h), -torch.inf)
        resulting_bucket = torch.ones(
            (input_ids.shape[0], h),
            device=input_ids.device,
            dtype=torch.long,
        )
        compatible_weight = compatible.to(torch.float32)
        vocab_size = logits.shape[-1]
        for digit in range(h):
            if not masked[:, digit].any():
                continue
            token_count = torch.zeros(
                input_ids.shape[0],
                vocab_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
            catalog_token = self.valid_item_tuples[:, digit]
            token_count.scatter_add_(
                1,
                catalog_token[None, :].expand(input_ids.shape[0], -1),
                compatible_weight,
            )
            supported_probability = target_probability[:, digit, :].masked_fill(
                token_count.eq(0), -torch.inf)
            top_probability, top_token = supported_probability.max(dim=-1)
            top_bucket = token_count.gather(
                1, top_token[:, None]).squeeze(1).to(torch.long)
            digit_utility = (
                top_probability
                * torch.log(
                    bucket_count.to(torch.float32)
                    / top_bucket.clamp_min(1).to(torch.float32)
                )
            )
            utility[:, digit] = torch.where(
                masked[:, digit],
                digit_utility,
                utility[:, digit],
            )
            resulting_bucket[:, digit] = top_bucket

        selected_digit = utility.argmax(dim=-1)
        selected_bucket = resulting_bucket.gather(
            1, selected_digit[:, None]).squeeze(1)
        history_length = input_ids.shape[1] - h
        return (
            selected_digit + history_length,
            bucket_count,
            selected_bucket,
        )

    def generate(self, batch, n_return_sequences=1):
        if batch['split'] == 'val':
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.val_num_beams)
        else:
            outputs = self.beam_search(batch,
                                       n_return_sequences,
                                       num_beams=self.num_beams)

        return outputs

    def beam_search(self, batch, num_return_sequences=1, num_beams=1):
        assert num_beams >= num_return_sequences

        # Static evidence is local to one user batch/beam-search trajectory.
        # Reset before the all-mask forward so no query can leak across
        # evaluation batches.
        self._item_arch_static_query = None

        input_ids: torch.Tensor = self.item_id2tokens[batch['input_ids']]
        input_ids = input_ids.reshape((input_ids.shape[0], -1))
        attention_mask: torch.Tensor = input_ids != 0

        batch_size = input_ids.shape[0]
        n_digit = self.tokenizer.n_digit
        if self.denoising_trace_enabled:
            trace_labels = batch['labels']
            if (trace_labels.ndim >= 2
                    and trace_labels.shape[-1] == n_digit):
                self._trace_target_tuples = trace_labels.reshape(
                    -1, n_digit)
            else:
                self._trace_target_tuples = self.item_id2tokens[
                    trace_labels.reshape(-1)]
            self._trace_batch_id = self._denoising_trace_batch
            self._denoising_trace_batch += 1
        guide_logits = None
        guide_position_weight = None
        if self.catalog_decode_mode != "beam":
            guide_logits = self._all_mask_catalog_logits(
                input_ids, attention_mask)
            guide_position_weight = self._catalog_decode_position_weights(
                guide_logits)
            if self.catalog_decode_mode == "parallel":
                catalog_scores = self._catalog_tuple_scores(
                    guide_logits,
                    position_weight=guide_position_weight,
                )
                top_index = torch.topk(
                    catalog_scores,
                    k=num_return_sequences,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ).indices
                return self.valid_item_tuples[top_index]

        # Prepare beam search inputs
        input_ids, attention_mask, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
            input_ids, attention_mask, batch_size, num_beams)

        if self.prefix_resolved_decode:
            return self._prefix_resolved_beam_search(
                input_ids,
                attention_mask,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams,
                num_return_sequences,
            )
        if self.subset_resolved_decode:
            return self._subset_resolved_beam_search(
                input_ids,
                attention_mask,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams,
                num_return_sequences,
                guide_logits,
                guide_position_weight,
            )

        num_transfer_tokens = n_digit // self.gen_steps
        num_transfered = 0
        for i in range(self.gen_steps):

            if i == (self.gen_steps - 1):
                num_transfer_tokens = n_digit - num_transfered

            outputs = self.llada_forward(input_ids=input_ids,
                                         attention_mask=attention_mask.long())

            logits = outputs.logits
            logits = logits / self.temperature
            if (self.item_arch_inference_timing == "inline"
                    and i >= self.item_arch_feedback_start_step):
                logits = self._item_arch_generation_logits(
                    outputs, input_ids, logits)
            if self.catalog_decode_mode in {"guided", "guided_hybrid"}:
                guide_log_prob = F.log_softmax(
                    guide_logits, dim=-1)
                if guide_position_weight is not None:
                    guide_log_prob = (
                        guide_log_prob
                        * guide_position_weight[:, :, None])
                guide_log_prob = guide_log_prob.repeat_interleave(
                    num_beams, dim=0)
                logits = logits.clone()
                logits[:, -n_digit:, :] = (
                    logits[:, -n_digit:, :]
                    + self.catalog_decode_weight * guide_log_prob
                )

            transfer_index = self.get_transfer_index(input_ids, logits,
                                                     num_transfer_tokens)

            for j in range(num_transfer_tokens):
                input_ids, beam_scores = self.beam_search_step(
                    logits, input_ids, transfer_index[:, j], beam_scores,
                    beam_idx_offset, batch_size, num_beams)

            num_transfered += num_transfer_tokens

        label_ids = input_ids[:, -n_digit:].reshape(
            batch_size, num_beams, n_digit)
        if self.catalog_decode_mode in {"hybrid", "guided_hybrid"}:
            guide_log_prob = F.log_softmax(guide_logits, dim=-1)
            expanded_guide = guide_log_prob[:, None, :, :].expand(
                -1, num_beams, -1, -1)
            guide_scores = expanded_guide.gather(
                dim=-1,
                index=label_ids[:, :, :, None],
            ).squeeze(-1)
            if guide_position_weight is not None:
                guide_scores = (
                    guide_scores
                    * guide_position_weight[:, None, :])
            guide_scores = guide_scores.sum(dim=-1)
            beam_component = self._normalize_catalog_fusion_scores(
                beam_scores.reshape(batch_size, num_beams),
                self.catalog_score_normalization,
            )
            guide_component = self._normalize_catalog_fusion_scores(
                guide_scores,
                self.catalog_score_normalization,
            )
            query_weight = self._catalog_decode_query_weights(
                guide_logits,
                beam_component,
                guide_component,
            )
            combined_scores = (
                beam_component
                + query_weight * guide_component
            )
            top_beam = torch.topk(
                combined_scores,
                k=num_return_sequences,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            return label_ids.gather(
                1,
                top_beam[:, :, None].expand(-1, -1, n_digit),
            )
        return label_ids[:, :num_return_sequences, :]

    def _subset_resolved_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size: int,
        num_beams: int,
        num_return_sequences: int,
        guide_logits: Optional[torch.Tensor],
        guide_position_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Denoise an informative SID subset, then complete from the catalog.

        Unlike prefix completion, the visible positions are selected
        independently for every beam from the denoiser confidence. Exact
        tuple constraints keep each partial SID completable. The remaining
        positions are scored in one final conditional pass, and each catalog
        item keeps the best accumulated score over all compatible beams.
        """
        h = self.tokenizer.n_digit
        transfer_histogram = torch.zeros(
            h, device=input_ids.device, dtype=torch.long)
        predicted_reduction_sum = 0.0
        predicted_reduction_count = 0

        for trace_step in range(self.subset_resolved_tokens):
            outputs = self.llada_forward(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
            )
            logits = outputs.logits / self.temperature
            if (self.item_arch_inference_timing == "inline"
                    and trace_step >= self.item_arch_feedback_start_step):
                logits = self._item_arch_generation_logits(
                    outputs, input_ids, logits)
            self._record_denoising_trace(
                trace_step, input_ids, logits[:, -h:, :], beam_scores,
                batch_size, num_beams)
            if self.subset_transfer_policy == "confidence":
                transfer_index = self.get_transfer_index(
                    input_ids, logits, 1)[:, 0]
            elif self.subset_transfer_policy == "sequential":
                # Fixed left-to-right digit order, ignoring confidence and
                # catalog-reduction utility entirely -- a cheap proxy for
                # how the compatible-catalog/transition mechanisms behave
                # under an autoregressive-style resolution constraint,
                # without building a separate AR backbone.
                target = input_ids[:, -h:]
                masked = target.eq(self.mask_token_id)
                position = torch.arange(
                    h, device=input_ids.device)[None, :].expand(
                        input_ids.shape[0], -1)
                leftmost_digit = position.masked_fill(
                    ~masked, h).argmin(dim=-1)
                transfer_index = (
                    input_ids.shape[1] - h) + leftmost_digit
            else:
                active_beam = torch.isfinite(beam_scores)
                (
                    transfer_index,
                    source_bucket,
                    selected_bucket,
                ) = self._catalog_reduction_transfer_index(
                    input_ids, logits)
                selected_digit = transfer_index - (
                    input_ids.shape[1] - h)
                transfer_histogram += torch.bincount(
                    selected_digit[active_beam],
                    minlength=h,
                )
                predicted_reduction = (
                    1.0
                    - selected_bucket.to(torch.float32)
                    / source_bucket.clamp_min(1).to(torch.float32)
                )
                predicted_reduction_sum += float(
                    predicted_reduction[active_beam].sum().item())
                predicted_reduction_count += int(active_beam.sum().item())
            input_ids, beam_scores = self.beam_search_step(
                logits,
                input_ids,
                transfer_index,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams,
                force_tuple_constraint=True,
            )

        if (self.subset_transfer_policy == "catalog_reduction"
                and not self._logged_subset_transfer_stats):
            histogram = transfer_histogram.detach().cpu().tolist()
            mean_reduction = (
                predicted_reduction_sum
                / max(1, predicted_reduction_count)
            )
            self.log(
                "catalog-reduction transfer: digit_histogram={} "
                "mean_predicted_bucket_reduction={:.4f}".format(
                    histogram,
                    mean_reduction,
                ))
            self._logged_subset_transfer_stats = True

        if self.subset_adaptive_bucket_threshold > 0:
            outputs = self.llada_forward(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
            )
            logits = outputs.logits / self.temperature
            if self.item_arch_inference_timing == "inline":
                logits = self._item_arch_generation_logits(
                    outputs, input_ids, logits)
            input_ids, beam_scores = self._adaptive_subset_token_step(
                logits,
                input_ids,
                beam_scores,
            )

        outputs = self.llada_forward(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
        )
        completion_full_logits = outputs.logits / self.temperature
        if (self.item_arch_inference_timing == "inline"
                and self.subset_resolved_tokens
                >= self.item_arch_feedback_start_step):
            completion_full_logits = self._item_arch_generation_logits(
                outputs, input_ids, completion_full_logits)
        completion_logits = completion_full_logits[:, -h:, :]
        self._record_denoising_trace(
            self.subset_resolved_tokens, input_ids, completion_logits,
            beam_scores, batch_size, num_beams)
        target = input_ids[:, -h:]
        visible = target.ne(self.mask_token_id)
        compatible = (
            target[:, None, :].eq(self.valid_item_tuples[None, :, :])
            | ~visible[:, None, :]
        ).all(dim=-1)
        completion_scores = self._catalog_tuple_scores(
            completion_logits,
            position_mask=~visible,
        )
        completion_scores = completion_scores.masked_fill(
            ~compatible, -torch.inf)
        completion_scores = self._item_arch_catalog_rerank(
            outputs,
            input_ids,
            completion_logits,
            completion_scores,
        )
        completion_scores = (
            beam_scores[:, None]
            + self.subset_completion_score_weight * completion_scores
        )

        n_items = self.valid_item_tuples.shape[0]
        item_scores = completion_scores.reshape(
            batch_size, num_beams, n_items).amax(dim=1)
        finite_count = torch.isfinite(item_scores).sum(dim=-1)
        if finite_count.min().item() < num_return_sequences:
            raise RuntimeError(
                "subset-resolved decoding found fewer compatible catalog "
                "items than requested return sequences")
        if not self._logged_subset_completion_stats:
            compatible_count = compatible.sum(dim=-1).to(torch.float32)
            visible_count = visible.sum(dim=-1).to(torch.float32)
            self.log(
                "subset completion: mean_visible_tokens={:.2f} "
                "mean_candidates_per_beam={:.2f} "
                "max_candidates_per_beam={}".format(
                    visible_count.mean().item(),
                    compatible_count.mean().item(),
                    int(compatible_count.max().item()),
                ))
            self._logged_subset_completion_stats = True

        if (self.catalog_decode_mode == "hybrid"
                and self.subset_catalog_preselection_fusion):
            if guide_logits is None:
                raise RuntimeError(
                    "subset catalog preselection fusion requires "
                    "guide logits")
            combined_scores = self._subset_catalog_preselection_scores(
                item_scores,
                guide_logits,
                guide_position_weight,
            )
            top_item = torch.topk(
                combined_scores,
                k=num_return_sequences,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            if not self._logged_subset_preselection_stats:
                beam_top_item = torch.topk(
                    item_scores,
                    k=num_return_sequences,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ).indices
                rescued_item = ~(
                    top_item[:, :, None]
                    == beam_top_item[:, None, :]
                ).any(dim=-1)
                rescued_per_query = rescued_item.sum(
                    dim=-1).to(torch.float32)
                rank_changes = top_item.ne(
                    beam_top_item).sum(dim=-1).to(torch.float32)
                self.log(
                    "subset catalog preselection fusion: "
                    "mean_compatible_items={:.2f} "
                    "changed_set_queries={}/{} "
                    "mean_rescued_items={:.2f} "
                    "mean_rank_changes={:.2f}".format(
                        finite_count.to(torch.float32).mean().item(),
                        int(rescued_per_query.gt(0).sum().item()),
                        rescued_per_query.numel(),
                        rescued_per_query.mean().item(),
                        rank_changes.mean().item(),
                    ))
                self._logged_subset_preselection_stats = True
            return self.valid_item_tuples[top_item]

        candidate_beams = min(num_beams, int(finite_count.min().item()))
        selected_scores, selected_items = torch.topk(
            item_scores,
            k=candidate_beams,
            dim=-1,
            largest=True,
            sorted=True,
        )
        label_ids = self.valid_item_tuples[selected_items]
        if self.catalog_decode_mode == "hybrid":
            if guide_logits is None:
                raise RuntimeError(
                    "hybrid subset completion requires guide logits")
            guide_log_prob = F.log_softmax(guide_logits, dim=-1)
            expanded_guide = guide_log_prob[:, None, :, :].expand(
                -1, candidate_beams, -1, -1)
            guide_scores = expanded_guide.gather(
                dim=-1,
                index=label_ids[:, :, :, None],
            ).squeeze(-1)
            if guide_position_weight is not None:
                guide_scores = (
                    guide_scores
                    * guide_position_weight[:, None, :])
            guide_scores = guide_scores.sum(dim=-1)
            completion_component = self._normalize_catalog_fusion_scores(
                selected_scores,
                self.catalog_score_normalization,
            )
            guide_component = self._normalize_catalog_fusion_scores(
                guide_scores,
                self.catalog_score_normalization,
            )
            query_weight = self._catalog_decode_query_weights(
                guide_logits,
                completion_component,
                guide_component,
            )
            combined_scores = (
                completion_component + query_weight * guide_component)
            top_item = torch.topk(
                combined_scores,
                k=num_return_sequences,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            return label_ids.gather(
                1,
                top_item[:, :, None].expand(-1, -1, h),
            )
        return label_ids[:, :num_return_sequences, :]

    def _record_denoising_trace(
        self, step, input_ids, token_logits, beam_scores,
        batch_size, num_beams,
    ):
        """Record best-beam dynamics and gold survival over the full beam.

        This hook is deliberately evaluation-only.  It uses the exact
        catalog tuple likelihood after any active inline item message, masks
        incompatible tuples, and records target mass, entropy, and |C|.
        """
        if not self.denoising_trace_enabled:
            return
        with torch.no_grad():
            h = self.tokenizer.n_digit
            best = beam_scores.reshape(batch_size, num_beams).argmax(dim=1)
            row = (torch.arange(batch_size, device=input_ids.device)
                   * num_beams + best)
            target = input_ids[row, -h:]
            logits = token_logits[row]
            visible = target.ne(self.mask_token_id)
            compatible = (
                target[:, None, :].eq(self.valid_item_tuples[None, :, :])
                | ~visible[:, None, :]
            ).all(dim=-1)
            scores = self._catalog_tuple_scores(
                logits, position_mask=~visible).masked_fill(
                    ~compatible, -torch.inf)
            posterior = torch.softmax(scores.to(torch.float32), dim=-1)
            entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(-1)
            gold_tuple = self._trace_target_tuples
            gold_match = (self.valid_item_tuples[None, :, :]
                          .eq(gold_tuple[:, None, :]).all(-1))
            gold_found = gold_match.any(-1)
            gold_index = gold_match.to(torch.long).argmax(-1)
            true_mass = posterior.gather(1, gold_index[:, None]).squeeze(1)

            # Aggregate catalog scores over every active beam.  This exposes
            # whether the gold item is still recoverable at each denoising
            # step, rather than only describing the currently best path.
            all_target = input_ids[:, -h:]
            all_visible = all_target.ne(self.mask_token_id)
            all_compatible = (
                all_target[:, None, :].eq(
                    self.valid_item_tuples[None, :, :])
                | ~all_visible[:, None, :]
            ).all(dim=-1)
            active = torch.isfinite(beam_scores)
            all_scores = self._catalog_tuple_scores(
                token_logits, position_mask=~all_visible)
            all_scores = all_scores + beam_scores[:, None]
            all_scores = all_scores.masked_fill(
                ~all_compatible | ~active[:, None], -torch.inf)
            aggregate_scores = all_scores.reshape(
                batch_size, num_beams, -1).amax(dim=1)
            gold_score = aggregate_scores.gather(
                1, gold_index[:, None]).squeeze(1)
            gold_rank = (
                aggregate_scores.gt(gold_score[:, None]).sum(-1) + 1)
            gold_survives = all_compatible.reshape(
                batch_size, num_beams, -1).gather(
                    2,
                    gold_index[:, None, None].expand(
                        -1, num_beams, 1),
                ).squeeze(-1)
            gold_survives = (gold_survives & active.reshape(
                batch_size, num_beams)).any(-1)
            # A finite-looking ordinal among the remaining compatible items
            # is meaningless once the gold tuple has left every active beam.
            # Mark it below the complete catalog in that case and require
            # survival for candidate recall.
            gold_rank = torch.where(
                gold_survives & gold_found,
                gold_rank,
                torch.full_like(
                    gold_rank, self.valid_item_tuples.shape[0] + 1),
            )

            # Two deterministic compact checksums make the control/post-hoc
            # token-trajectory identity directly auditable without dumping
            # every beam tuple to a very large JSON file.
            flat_target = all_target.reshape(batch_size, -1).to(torch.int64)
            weights = torch.arange(
                1, flat_target.shape[1] + 1,
                device=flat_target.device, dtype=torch.int64)[None, :]
            beam_sig_1 = ((flat_target + 1) * weights).sum(-1)
            beam_sig_2 = ((flat_target + 1) * weights.square()).sum(-1)
            for i in range(batch_size):
                self.denoising_trace.append({
                    "batch": int(self._trace_batch_id),
                    "row": int(i),
                    "step": int(step),
                    "true_mass": float(true_mass[i].cpu()),
                    "entropy": float(entropy[i].cpu()),
                    "compatible_size": int(compatible[i].sum().cpu()),
                    "gold_found": bool(gold_found[i].cpu()),
                    "gold_survives": bool(gold_survives[i].cpu()),
                    "gold_rank": int(gold_rank[i].cpu()),
                    "gold_in_top128": bool(
                        gold_survives[i].cpu()
                        and gold_found[i].cpu()
                        and gold_rank[i].cpu() <= 128),
                    "active_beams": int(active.reshape(
                        batch_size, num_beams)[i].sum().cpu()),
                    "beam_signature": [
                        int(beam_sig_1[i].cpu()),
                        int(beam_sig_2[i].cpu()),
                    ],
                    "resolved_mask": [
                        bool(value) for value in visible[i].cpu().tolist()
                    ],
                })

    def _adaptive_subset_token_step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        beam_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve one more SID token only for still-ambiguous partial beams.

        Each existing partial beam is retained. Beams whose compatible
        catalog bucket exceeds the configured threshold take their most
        likely catalog-valid token; sufficiently resolved beams remain
        untouched. Exact completion then compares all paths after every path
        has accumulated a total score over the full SID.
        """
        h = self.tokenizer.n_digit
        target = input_ids[:, -h:]
        visible = target.ne(self.mask_token_id)
        compatible = (
            target[:, None, :].eq(self.valid_item_tuples[None, :, :])
            | ~visible[:, None, :]
        ).all(dim=-1)
        bucket_count = compatible.sum(dim=-1)
        continue_mask = bucket_count.gt(
            self.subset_adaptive_bucket_threshold)
        if not continue_mask.any():
            if not self._logged_subset_adaptive_stats:
                self.log(
                    "adaptive subset stopping: threshold={} "
                    "continued=0/{} mean_bucket_before={:.2f}".format(
                        self.subset_adaptive_bucket_threshold,
                        continue_mask.numel(),
                        bucket_count.to(torch.float32).mean().item(),
                    ))
                self._logged_subset_adaptive_stats = True
            return input_ids, beam_scores

        transfer_index = self.get_transfer_index(
            input_ids, logits, 1)[:, 0]
        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index[:, None, None].expand(
                -1, 1, vocab_size),
            dim=1,
        ).squeeze(1)
        next_token_scores = F.log_softmax(
            next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - h
        valid_tokens = self.posValidTokens[
            transfer_index - his_len]
        valid_tokens = valid_tokens.masked_fill(
            valid_tokens.eq(0), self.mask_token_id)
        next_token_scores = next_token_scores.gather(
            index=valid_tokens, dim=1)
        next_token_scores = next_token_scores.masked_fill(
            valid_tokens.eq(self.mask_token_id), -torch.inf)
        allowed = self._tuple_decode_allowed(
            input_ids, transfer_index, valid_tokens)
        next_token_scores = next_token_scores.masked_fill(
            ~allowed, -torch.inf)
        best_score, best_slot = next_token_scores.max(dim=-1)
        if not torch.isfinite(best_score[continue_mask]).all():
            raise RuntimeError(
                "adaptive subset stopping found no valid continuation")
        next_tokens = valid_tokens.gather(
            1, best_slot[:, None]).squeeze(1)

        updated_ids = input_ids.clone()
        row = torch.arange(
            input_ids.shape[0], device=input_ids.device)
        updated_ids[
            row[continue_mask],
            transfer_index[continue_mask],
        ] = next_tokens[continue_mask]
        updated_scores = (
            beam_scores
            + torch.where(
                continue_mask,
                best_score,
                torch.zeros_like(best_score),
            )
        )

        if not self._logged_subset_adaptive_stats:
            updated_target = updated_ids[:, -h:]
            updated_visible = updated_target.ne(self.mask_token_id)
            updated_compatible = (
                updated_target[:, None, :].eq(
                    self.valid_item_tuples[None, :, :])
                | ~updated_visible[:, None, :]
            ).all(dim=-1)
            updated_count = updated_compatible.sum(dim=-1)
            self.log(
                "adaptive subset stopping: threshold={} continued={}/{} "
                "mean_bucket_before={:.2f} mean_bucket_after={:.2f}".format(
                    self.subset_adaptive_bucket_threshold,
                    int(continue_mask.sum().item()),
                    continue_mask.numel(),
                    bucket_count.to(torch.float32).mean().item(),
                    updated_count.to(torch.float32).mean().item(),
                ))
            self._logged_subset_adaptive_stats = True
        return updated_ids, updated_scores

    def _all_mask_catalog_logits(
        self,
        history_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict every target SID position from history in one parallel pass."""
        batch_size = history_ids.shape[0]
        target_ids = torch.full(
            (batch_size, self.tokenizer.n_digit),
            self.mask_token_id,
            device=history_ids.device,
            dtype=torch.long,
        )
        target_attention = torch.ones_like(target_ids)
        all_ids = torch.cat([history_ids, target_ids], dim=-1)
        all_attention = torch.cat([
            history_attention_mask.long(),
            target_attention,
        ], dim=-1)
        outputs = self.llada_forward(
            input_ids=all_ids,
            attention_mask=all_attention,
        )
        logits = outputs.logits / self.temperature
        if self.item_arch_inference_timing == "inline":
            logits = self._item_arch_generation_logits(
                outputs, all_ids, logits)
        return logits[:, -self.tokenizer.n_digit:, :]

    def _prefix_resolved_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size: int,
        num_beams: int,
        num_return_sequences: int,
    ) -> torch.Tensor:
        """Decode a short prefix, then rank its catalog suffix in parallel."""
        h = self.tokenizer.n_digit
        his_len = input_ids.shape[1] - h

        # Exact catalog constraints make every partial prefix completable.
        for digit in range(self.soft_prefix_length):
            outputs = self.llada_forward(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
            )
            logits = outputs.logits / self.temperature
            if self.item_arch_inference_timing == "inline":
                logits = self._item_arch_generation_logits(
                    outputs, input_ids, logits)
            transfer_index = torch.full(
                (input_ids.shape[0], ),
                his_len + digit,
                device=input_ids.device,
                dtype=torch.long,
            )
            input_ids, beam_scores = self.beam_search_step(
                logits,
                input_ids,
                transfer_index,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams,
                force_tuple_constraint=True,
            )

        bucket_rows, found = self._lookup_prefix_buckets(
            input_ids[:, -h:])
        if not found.all():
            raise RuntimeError(
                "catalog-constrained prefix decoding produced an "
                "unknown prefix")

        soft_inputs = self._prefix_completion_embeddings(
            input_ids, bucket_rows)
        outputs = self.llada_forward(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
            inputs_embeds=soft_inputs,
        )
        completion_logits = outputs.logits / self.temperature
        if self.item_arch_inference_timing == "inline":
            completion_logits = self._item_arch_generation_logits(
                outputs, input_ids, completion_logits)
        suffix_logits = completion_logits[
            :, -h + self.soft_prefix_length:, :]
        suffix_log_prob = F.log_softmax(suffix_logits, dim=-1)

        member_index = self.prefix_bucket_members[bucket_rows]
        valid_member = member_index.ge(0)
        safe_member = member_index.clamp_min(0)
        candidate_tuples = self.valid_item_tuples[safe_member]
        candidate_suffix = candidate_tuples[
            :, :, self.soft_prefix_length:]
        expanded_log_prob = suffix_log_prob[:, None, :, :].expand(
            -1, candidate_suffix.shape[1], -1, -1)
        suffix_scores = expanded_log_prob.gather(
            dim=-1,
            index=candidate_suffix[:, :, :, None],
        ).squeeze(-1).mean(dim=-1)
        suffix_scores = suffix_scores.masked_fill(
            ~valid_member, -np.inf)
        completion_scores = (
            beam_scores[:, None]
            + self.prefix_completion_score_weight * suffix_scores
        )

        max_bucket = completion_scores.shape[1]
        completion_scores = completion_scores.reshape(
            batch_size, num_beams * max_bucket)
        beam_scores, top_index = torch.topk(
            completion_scores,
            k=num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )
        source_beam = torch.div(
            top_index, max_bucket, rounding_mode="floor")
        member_slot = top_index.remainder(max_bucket)
        batch_offset = (
            torch.arange(batch_size, device=input_ids.device)[:, None]
            * num_beams
        )
        source_row = (source_beam + batch_offset).reshape(-1)
        member_slot = member_slot.reshape(-1)
        chosen_bucket = bucket_rows[source_row]
        chosen_member = self.prefix_bucket_members[
            chosen_bucket, member_slot]
        if chosen_member.lt(0).any():
            raise RuntimeError(
                "invalid padded catalog completion selected")

        input_ids = input_ids[source_row].clone()
        input_ids[:, -h:] = self.valid_item_tuples[chosen_member]
        label_ids = input_ids[:, -h:].reshape(
            batch_size, num_beams, h)
        return label_ids[:, :num_return_sequences, :]

    def _prefix_completion_embeddings(
        self,
        input_ids: torch.Tensor,
        bucket_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a masked suffix as its catalog-posterior barycenter."""
        embeddings = self.llada.model.transformer.wte(input_ids).clone()
        if self.prefix_soft_decode_strength == 0.0:
            return embeddings
        member_index = self.prefix_bucket_members[bucket_rows]
        valid = member_index.ge(0)
        safe_member = member_index.clamp_min(0)
        candidates = self.valid_item_tuples[safe_member][
            :, :, self.soft_prefix_length:]
        candidate_embeds = self.llada.model.transformer.wte(candidates)
        barycenter = (
            candidate_embeds * valid[:, :, None, None]
        ).sum(dim=1) / valid.sum(dim=1)[:, None, None].clamp_min(1)
        suffix_slice = slice(
            input_ids.shape[1] - self.tokenizer.n_digit
            + self.soft_prefix_length,
            input_ids.shape[1],
        )
        hard_suffix = embeddings[:, suffix_slice, :]
        strength = self.prefix_soft_decode_strength
        embeddings[:, suffix_slice, :] = (
            (1.0 - strength) * hard_suffix
            + strength * barycenter.detach()
        )
        return embeddings

    def beam_search_step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size,
        num_beams,
        force_tuple_constraint: bool = False,
    ):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, vocab_size),
            dim=1).squeeze(1)

        # Calculate log softmax over the last dimension
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - self.tokenizer.n_digit
        valid_tokens = self.posValidTokens[transfer_index - his_len]

        valid_tokens = valid_tokens.masked_fill(valid_tokens == 0,
                                                self.mask_token_id)
        next_token_scores = next_token_scores.gather(index=valid_tokens, dim=1)
        next_token_scores = next_token_scores.masked_fill(
            valid_tokens == self.mask_token_id, -np.inf)
        if self.tuple_constrained_decode or force_tuple_constraint:
            allowed = self._tuple_decode_allowed(
                input_ids, transfer_index, valid_tokens)
            next_token_scores = next_token_scores.masked_fill(
                ~allowed, -np.inf)
        vocab_size = next_token_scores.shape[-1]

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
            next_token_scores)
        next_token_scores = next_token_scores.view(batch_size,
                                                   num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                    2 * num_beams,
                                                    dim=1,
                                                    largest=True,
                                                    sorted=True)

        next_indices = torch.div(next_tokens,
                                 vocab_size,
                                 rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        input_ids = input_ids[beam_idx + beam_idx_offset]
        transfer_index = transfer_index[beam_idx + beam_idx_offset]

        valid_tokens = self.posValidTokens[transfer_index - his_len]
        beam_next_tokens = valid_tokens.gather(
            index=beam_next_tokens.unsqueeze(1), dim=1).squeeze(1)

        x_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        input_ids[x_index, transfer_index] = beam_next_tokens

        return input_ids, beam_scores

    def _tuple_decode_allowed(
        self,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidates extending at least one valid catalog tuple."""
        h = self.tokenizer.n_digit
        target = input_ids[:, -h:]
        known = target.ne(self.mask_token_id)
        compatible = (
            target[:, None, :].eq(self.valid_item_tuples[None, :, :])
            | ~known[:, None, :]
        ).all(dim=-1)
        digit = transfer_index - (input_ids.shape[1] - h)
        tuple_token_at_digit = self.valid_item_tuples[:, digit].transpose(0, 1)
        counts = torch.zeros(
            input_ids.shape[0],
            self.tokenizer.vocab_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        counts.scatter_add_(
            1, tuple_token_at_digit, compatible.to(torch.float32))
        allowed = counts.gather(1, candidate_tokens).gt(0)
        # A fallback keeps decoding numerically safe if externally supplied
        # partial codes do not lie on the catalog manifold.
        fallback = ~allowed.any(dim=1)
        if fallback.any():
            allowed[fallback] = candidate_tokens[fallback].ne(
                self.mask_token_id)
        return allowed

    def prepare_beam_search_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size,
        num_beams,
    ):
        beam_label_ids = torch.ones(
            (batch_size * num_beams, self.tokenizer.n_digit),
            device=self.config['device'],
            dtype=torch.long) * self.mask_token_id
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        all_input_ids = torch.cat([input_ids, beam_label_ids], dim=-1)

        beam_label_attention_mask = torch.ones_like(
            beam_label_ids, device=self.config['device'], dtype=torch.long)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        all_attention_mask = torch.cat(
            [attention_mask, beam_label_attention_mask], dim=-1)

        beam_scores = torch.zeros((batch_size, num_beams),
                                  dtype=torch.float,
                                  device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams, ))

        beam_idx_offset = torch.arange(
            batch_size, device=self.config['device']).repeat_interleave(
                num_beams) * num_beams

        return all_input_ids, all_attention_mask, initial_beam_scores, beam_idx_offset

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
