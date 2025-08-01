"""
Continuous Thought Machine with Neuron-Level Normalization

This module extends the original CTM with biologically-inspired neuron-level normalization
to improve training stability, biological plausibility, and interpretability of neural synchrony.

The normalization is applied to post-activations at each internal tick using running
statistics maintained per neuron.
"""

import torch.nn as nn
import torch
import numpy as np
import math

from models.ctm import ContinuousThoughtMachine
from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)


class NormalizedContinuousThoughtMachine(ContinuousThoughtMachine):
    """
    Continuous Thought Machine with Neuron-Level Normalization.

    This extends the original CTM by adding biologically-inspired neuron-level normalization
    of post-activations at each internal tick. The normalization maintains running mean μ_i
    and variance σ_i² per neuron, updated using a decay factor α = 0.01.

    The normalized activations are computed as:
        ẑ_i^t = (z_i^t - μ_i) / √(σ_i² + ε)
    
    where ε = 10⁻⁵ for numerical stability.

    This normalization is applied to:
    - The history buffer Ẑ^t
    - Synchronization matrix S^t = Ẑ^t (Ẑ^t)^⊤
    - Inputs to the synapse model

    Args:
        Same as ContinuousThoughtMachine, with normalization enabled by default.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 normalization_decay=0.01,  # Decay factor for running statistics
                 normalization_epsilon=1e-5,  # Epsilon for numerical stability
                 ):
        # Call parent constructor with normalization enabled
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type=backbone_type,
            positional_embedding_type=positional_embedding_type,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            use_neuron_normalization=True,  # Enable normalization by default
            normalization_decay=normalization_decay,
            normalization_epsilon=normalization_epsilon,
        )


def create_ctm_with_normalization(config):
    """
    Factory function to create a CTM with neuron-level normalization.
    
    Args:
        config (dict): Configuration dictionary containing CTM parameters
        
    Returns:
        NormalizedContinuousThoughtMachine: CTM instance with normalization enabled
    """
    return NormalizedContinuousThoughtMachine(
        iterations=config.get('iterations', 10),
        d_model=config.get('d_model', 256),
        d_input=config.get('d_input', 128),
        heads=config.get('heads', 8),
        n_synch_out=config.get('n_synch_out', 64),
        n_synch_action=config.get('n_synch_action', 64),
        synapse_depth=config.get('synapse_depth', 3),
        memory_length=config.get('memory_length', 8),
        deep_nlms=config.get('deep_nlms', True),
        memory_hidden_dims=config.get('memory_hidden_dims', 64),
        do_layernorm_nlm=config.get('do_layernorm_nlm', False),
        backbone_type=config.get('backbone_type', 'resnet18-2'),
        positional_embedding_type=config.get('positional_embedding_type', 'learnable-fourier'),
        out_dims=config.get('out_dims', 4),
        prediction_reshaper=config.get('prediction_reshaper', [-1]),
        dropout=config.get('dropout', 0.1),
        dropout_nlm=config.get('dropout_nlm', None),
        neuron_select_type=config.get('neuron_select_type', 'random-pairing'),
        n_random_pairing_self=config.get('n_random_pairing_self', 0),
        normalization_decay=config.get('normalization_decay', 0.01),
        normalization_epsilon=config.get('normalization_epsilon', 1e-5),
    )


def create_baseline_ctm(config):
    """
    Factory function to create a baseline CTM without normalization.
    
    Args:
        config (dict): Configuration dictionary containing CTM parameters
        
    Returns:
        ContinuousThoughtMachine: Baseline CTM instance without normalization
    """
    return ContinuousThoughtMachine(
        iterations=config.get('iterations', 10),
        d_model=config.get('d_model', 256),
        d_input=config.get('d_input', 128),
        heads=config.get('heads', 8),
        n_synch_out=config.get('n_synch_out', 64),
        n_synch_action=config.get('n_synch_action', 64),
        synapse_depth=config.get('synapse_depth', 3),
        memory_length=config.get('memory_length', 8),
        deep_nlms=config.get('deep_nlms', True),
        memory_hidden_dims=config.get('memory_hidden_dims', 64),
        do_layernorm_nlm=config.get('do_layernorm_nlm', False),
        backbone_type=config.get('backbone_type', 'resnet18-2'),
        positional_embedding_type=config.get('positional_embedding_type', 'learnable-fourier'),
        out_dims=config.get('out_dims', 4),
        prediction_reshaper=config.get('prediction_reshaper', [-1]),
        dropout=config.get('dropout', 0.1),
        dropout_nlm=config.get('dropout_nlm', None),
        neuron_select_type=config.get('neuron_select_type', 'random-pairing'),
        n_random_pairing_self=config.get('n_random_pairing_self', 0),
        use_neuron_normalization=False,  # Disable normalization for baseline
    ) 