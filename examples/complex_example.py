"""Complex VODE Example: Transformer-like Architecture

This example demonstrates VODE with a sophisticated Transformer-style architecture featuring:
- Multi-head self-attention mechanisms
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Encoder-decoder structure
- Deep nested module hierarchies

Usage:
    python complex_example.py

    # Or visualize with VODE CLI:
    vode --stage4 --depth 3 complex_example.py
"""

import torch
import torch.nn as nn
import math
from vode.capture.static_capture import capture_static_execution_graph
from vode.capture.dynamic_capture import capture_dynamic_execution_graph
from vode.visualize.graphviz_renderer import render_execution_graph, expand_to_depth


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple attention heads.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """Forward pass through multi-head attention."""
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and apply output projection
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_linear(context)

        return output


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    Implements: FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass through feed-forward network."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed-forward.

    Structure:
        x -> LayerNorm -> MultiHeadAttention -> Residual ->
        LayerNorm -> FeedForward -> Residual -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Forward pass through encoder layer."""
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout1(x)
        x = x + residual

        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual

        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and feed-forward.

    Structure:
        x -> LayerNorm -> SelfAttention -> Residual ->
        LayerNorm -> CrossAttention -> Residual ->
        LayerNorm -> FeedForward -> Residual -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """Forward pass through decoder layer."""
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = x + residual

        # Cross-attention with residual connection
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.dropout2(x)
        x = x + residual

        # Feed-forward with residual connection
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = x + residual

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Adds position information to input embeddings using sine and cosine functions.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder stack.

    Consists of multiple encoder layers stacked together.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Forward pass through encoder stack."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer decoder stack.

    Consists of multiple decoder layers stacked together.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """Forward pass through decoder stack."""
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ComplexTransformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks.

    Architecture:
        - Input embedding + positional encoding
        - Encoder stack (N layers)
        - Decoder stack (N layers)
        - Output projection

    This demonstrates a sophisticated architecture with:
        - Deep nesting (6+ levels)
        - Multiple attention mechanisms
        - Residual connections throughout
        - Layer normalization
        - Position-wise feed-forward networks
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder and decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass through the Transformer.

        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source mask (optional)
            tgt_mask: Target mask (optional)

        Returns:
            Output logits (batch_size, tgt_len, tgt_vocab_size)
        """
        # Encode source
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.src_pos_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode target
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output


def print_detailed_summary(model):
    """Print detailed model summary with hierarchy."""
    print("\nDetailed Model Summary:")
    print("=" * 80)

    total_params = 0
    module_counts = {}

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        module_counts[module_type] = module_counts.get(module_type, 0) + 1

        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                depth = name.count(".")
                indent = "  " * depth
                print(f"{indent}{name:50s} {module_type:20s} {params:>12,}")
                total_params += params

    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print("\nModule Type Counts:")
    for module_type, count in sorted(module_counts.items()):
        print(f"  {module_type:30s}: {count:>3}")


def analyze_hierarchy_depth(root):
    """Analyze and display hierarchy depth information."""
    print("\nHierarchy Depth Analysis:")
    print("-" * 60)

    def get_max_depth(node, current_depth=0):
        if not node.children:
            return current_depth
        return max(get_max_depth(child, current_depth + 1) for child in node.children)

    max_depth = get_max_depth(root)
    print(f"Maximum hierarchy depth: {max_depth}")

    # Show expansion at each depth
    for depth in range(min(max_depth + 1, 5)):
        expanded = expand_to_depth(root, max_depth=depth)
        print(f"\nDepth {depth}: {len(expanded)} nodes")

        # Sample some nodes
        sample_size = min(3, len(expanded))
        for i in range(sample_size):
            node = expanded[i]
            print(f"  [{i}] {node.operation.op_type:25s} - {node.name[:40]}")

        if len(expanded) > sample_size:
            print(f"  ... and {len(expanded) - sample_size} more nodes")


def main():
    """Main function demonstrating complex Transformer architecture."""
    print("=" * 80)
    print("VODE Complex Example: Transformer Architecture")
    print("=" * 80)

    # Model configuration
    config = {
        "src_vocab_size": 10000,
        "tgt_vocab_size": 10000,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 3,  # Reduced for faster execution
        "num_decoder_layers": 3,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_len": 100,
    }

    # Create model
    print("\n1. Creating Transformer model...")
    print(f"   Configuration:")
    for key, value in config.items():
        print(f"     {key:25s}: {value}")

    model = ComplexTransformer(**config)

    # Print detailed summary
    print_detailed_summary(model)

    # Static capture
    print("\n2. Static capture (structure only)...")
    static_root = capture_static_execution_graph(model)
    print(f"   Root node: {static_root.name}")
    print(f"   Number of direct children: {len(static_root.children)}")

    # Analyze hierarchy
    analyze_hierarchy_depth(static_root)

    # Dynamic capture
    print("\n3. Dynamic capture (with runtime data)...")
    src = torch.randint(0, config["src_vocab_size"], (2, 10))  # (batch, seq_len)
    tgt = torch.randint(0, config["tgt_vocab_size"], (2, 8))
    print(f"   Source shape: {tuple(src.shape)}")
    print(f"   Target shape: {tuple(tgt.shape)}")

    dynamic_root = capture_dynamic_execution_graph(model, src, tgt)
    print(f"   Captured {len(dynamic_root.children)} top-level modules")

    # Render at different depths
    print("\n4. Rendering visualizations...")

    depths_to_render = [0, 1, 2, 3]
    for depth in depths_to_render:
        print(f"   - Depth {depth}...")
        dot = render_execution_graph(static_root, max_depth=depth)
        filename = f"complex_depth{depth}.gv"
        with open(filename, "w") as f:
            f.write(dot.source)
        print(f"     Saved to: {filename}")

    # Dynamic visualization
    print("   - Dynamic (with shapes)...")
    dot_dynamic = render_execution_graph(dynamic_root, max_depth=1)
    with open("complex_dynamic.gv", "w") as f:
        f.write(dot_dynamic.source)
    print("     Saved to: complex_dynamic.gv")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-head self-attention mechanisms")
    print("  ✓ Encoder-decoder architecture")
    print("  ✓ Position-wise feed-forward networks")
    print("  ✓ Layer normalization and residual connections")
    print("  ✓ Deep nested hierarchies (6+ levels)")
    print("  ✓ ModuleList for repeated layers")
    print("  ✓ Complex dataflow with multiple paths")
    print("\nVisualization Tips:")
    print("  - Use depth 0-1 for high-level architecture overview")
    print("  - Use depth 2-3 to see encoder/decoder layer details")
    print("  - Use depth 4+ to see attention mechanism internals")
    print("\nTo visualize the .gv files:")
    print("  dot -Tpng complex_depth1.gv -o complex_depth1.png")
    print("  dot -Tsvg complex_depth2.gv -o complex_depth2.svg")


if __name__ == "__main__":
    main()
