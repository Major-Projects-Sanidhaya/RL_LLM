"""
Hierarchical Policy Module (Advanced - Implement After Basics Work)

This module implements a two-level hierarchical policy for more sophisticated generation:
- High-level policy: Samples abstract "intentions" (latent plans)
- Low-level policy: Generates tokens conditioned on the current intention

How it works:
- State is encoded by shared transformer encoder
- High-level policy outputs Gaussian distribution over intention space
- Intention is sampled using reparameterization trick
- Low-level policy (transformer decoder) generates tokens conditioned on intention
- Enables longer-term planning and more coherent generation

Implementation approach:
- Shared state encoder processes token sequence
- High-level: State -> Mean & Std -> Sample intention (continuous vector)
- Low-level: State + Intention -> Decoder -> Next token distribution
- More complex than simple policy, use only after basic version works
- Inspired by hierarchical RL and options framework
"""

import torch
import torch.nn as nn
import math
from .networks import PositionalEncoding


class HierarchicalPolicy(nn.Module):
    """
    Two-level hierarchical policy for structured generation

    High-level policy: Generates abstract intentions (latent plans)
    Low-level policy: Generates tokens conditioned on those intentions

    This enables better long-term planning and coherence.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, intention_dim: int = 64,
                 num_layers: int = 4, nhead: int = 8):
        """
        Initialize hierarchical policy

        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimensionality of model embeddings
            intention_dim: Dimensionality of latent intention vector
            num_layers: Total transformer layers (split between encoder/decoder)
            nhead: Number of attention heads
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.intention_dim = intention_dim

        # Shared state encoder (processes current token sequence)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        # Use half the layers for encoding
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)

        # HIGH-LEVEL POLICY: State -> Intention distribution
        # Outputs parameters of Gaussian distribution over intentions
        self.intention_mean = nn.Linear(d_model, intention_dim)
        self.intention_logstd = nn.Linear(d_model, intention_dim)

        # LOW-LEVEL POLICY: State + Intention -> Token distribution
        # Transformer decoder that conditions on intention
        self.token_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Project intention to d_model for use in decoder
        self.intention_proj = nn.Linear(intention_dim, d_model)

        # Output head: projects decoder output to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Separate value heads for two-level hierarchy
        self.high_value = nn.Linear(d_model, 1)  # Value of intention
        self.low_value = nn.Linear(d_model, 1)  # Value of token given intention
    
    def encode_state(self, token_ids):
        """Encode state to hidden representation"""
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.state_encoder(x)
        return x
    
    def sample_intention(self, state_encoding):
        """
        Sample intention from high-level policy
        
        Args:
            state_encoding: [batch, seq_len, d_model]
        Returns:
            intention: [batch, intention_dim]
            intention_dist: Distribution object
        """
        # Pool state
        pooled = state_encoding.mean(dim=1)  # [batch, d_model]
        
        # Predict mean and log_std
        mean = self.intention_mean(pooled)
        logstd = self.intention_logstd(pooled)
        std = torch.exp(logstd)
        
        # Create distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Sample using reparameterization trick
        intention = dist.rsample()
        
        return intention, dist
    
    def forward(self, token_ids, intention=None, return_intention=False):
        """
        Forward pass
        
        Args:
            token_ids: [batch, seq_len]
            intention: [batch, intention_dim] or None
            return_intention: whether to return intention info
        """
        # Encode state
        state_encoding = self.encode_state(token_ids)
        
        # Sample intention if not provided
        if intention is None:
            intention, intention_dist = self.sample_intention(state_encoding)
        else:
            # Create dummy distribution for API consistency
            intention_dist = None
        
        # Project intention and expand to sequence length
        intention_encoded = self.intention_proj(intention)  # [batch, d_model]
        intention_expanded = intention_encoded.unsqueeze(1).expand(
            -1, state_encoding.size(1), -1
        )  # [batch, seq_len, d_model]
        
        # Decode tokens conditioned on intention
        # Use intention as memory in decoder
        decoded = self.token_decoder(
            state_encoding,
            intention_expanded
        )
        
        # Get logits for last position
        logits = self.output_layer(decoded[:, -1, :])
        
        if return_intention:
            # Compute values
            pooled = state_encoding.mean(dim=1)
            high_value = self.high_value(pooled)
            low_value = self.low_value(pooled)
            
            return {
                'logits': logits,
                'intention': intention,
                'intention_dist': intention_dist,
                'high_value': high_value,
                'low_value': low_value
            }
        else:
            return logits
    
    def get_action_distribution(self, token_ids, intention=None):
        """Get categorical distribution over actions"""
        logits = self.forward(token_ids, intention)
        return torch.distributions.Categorical(logits=logits)