"""Large model visualization example.

Demonstrates VODE on a realistic large model similar to the test case.
Shows best practices for visualizing complex architectures.

NOTE: Run from workspace root with vode in PYTHONPATH:
    cd /path/to/workspace
    python vode/examples/large_model_example.py
"""

import sys
from pathlib import Path

# Add vode to path (if not installed)
vode_path = Path(__file__).parent.parent.parent
if vode_path.exists():
    sys.path.insert(0, str(vode_path))

import torch
import torch.nn as nn
from vode.visualize import vode


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class LargeTransformer(nn.Module):
    """Large transformer model with many layers."""

    def __init__(self, vocab_size=10000, dim=512, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, dim))

        # Many transformer blocks (detected as loop)
        self.blocks = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_layers)])

        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.blocks(x)
        x = self.ln_final(x)
        x = self.head(x)
        return x


def main():
    """Demonstrate large model visualization strategies."""
    print("VODE Large Model Example")
    print("=" * 70)

    # Create large model
    model = LargeTransformer(vocab_size=10000, dim=512, num_layers=12)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {total_params:,}")

    # Strategy 1: Start with high-level overview
    print("\n" + "=" * 70)
    print("STRATEGY 1: High-level overview (depth 2)")
    print("=" * 70)
    print("\nUse low depth to understand overall architecture...")
    vode(model, mode="static", output="large_d2.svg", max_depth=2)
    print("Generated: large_d2.svg")
    print("  → Shows: embedding, blocks, ln_final, head")

    # Strategy 2: Medium detail
    print("\n" + "=" * 70)
    print("STRATEGY 2: Medium detail (depth 4)")
    print("=" * 70)
    print("\nIncrease depth to see block internals...")
    vode(model, mode="static", output="large_d4.svg", max_depth=4)
    print("Generated: large_d4.svg")
    print("  → Shows: individual transformer blocks with attention/MLP")

    # Strategy 3: Full detail (may be large)
    print("\n" + "=" * 70)
    print("STRATEGY 3: Full detail (no depth limit)")
    print("=" * 70)
    print("\nCapture complete hierarchy...")
    vode(model, mode="static", output="large_full.svg")
    print("Generated: large_full.svg")
    print("  → Shows: all modules including Linear layers in MLP")

    # Strategy 4: Dynamic capture with depth control
    print("\n" + "=" * 70)
    print("STRATEGY 4: Dynamic capture with tensor shapes")
    print("=" * 70)
    print("\nCapture runtime execution with sample input...")
    x = torch.randint(0, 10000, (2, 64))  # [batch=2, seq_len=64]
    vode(
        model,
        x,
        mode="dynamic",
        output="large_dynamic_d3.svg",
        max_depth=3,
        compute_stats=False,  # Disable stats for speed
    )
    print("Generated: large_dynamic_d3.svg")
    print("  → Shows: tensor shapes at each layer")

    # Strategy 5: Export to Graphviz for custom processing
    print("\n" + "=" * 70)
    print("STRATEGY 5: Export to Graphviz source")
    print("=" * 70)
    print("\nExport raw .gv file for custom styling...")
    vode(model, mode="static", output="large.gv", format="gv", max_depth=4)
    print("Generated: large.gv")
    print("  → Can be edited manually or processed by other tools")

    # Summary
    print("\n" + "=" * 70)
    print("BEST PRACTICES FOR LARGE MODELS")
    print("=" * 70)
    print(
        """
1. Start with depth 2-3 for overview
2. Increase depth incrementally to explore details
3. Use static mode for structure (faster, no input needed)
4. Use dynamic mode only when you need tensor shapes
5. Disable compute_stats for faster dynamic capture
6. Export to .gv for custom processing
7. Use collapse_loops=True to simplify Sequential/ModuleList

Generated files:
  - large_d2.svg (overview)
  - large_d4.svg (medium detail)
  - large_full.svg (complete)
  - large_dynamic_d3.svg (with tensor shapes)
  - large.gv (Graphviz source)
"""
    )


if __name__ == "__main__":
    main()
