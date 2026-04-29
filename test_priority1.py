#!/usr/bin/env python3
"""Quick test to verify Priority 1 implementation can initialize without errors."""

import os
import sys
import torch

# Set minimal config
os.environ["CROSS_LAYER_SHARING"] = "1"
os.environ["DELTA_RANK"] = "32"
os.environ["KFEC_ENABLED"] = "1"
os.environ["KFEC_RANK"] = "8"
os.environ["LQER_ENABLED"] = "0"
os.environ["VOCAB_SIZE"] = "8192"
os.environ["ITERATIONS"] = "10"
os.environ["MAX_WALLCLOCK_SECONDS"] = "0"

print("Testing Priority 1 implementation...")

try:
    # Import config
    print("1. Importing train_gpt...")
    import train_gpt
    print("   ✓ Import successful")

    # Check config loaded
    print("2. Checking configuration...")
    h = train_gpt.h
    assert h.cross_layer_sharing == True, "cross_layer_sharing not enabled"
    assert h.delta_rank == 32, f"delta_rank is {h.delta_rank}, expected 32"
    assert h.kfec_enabled == True, "kfec_enabled not enabled"
    assert h.kfec_rank == 8, f"kfec_rank is {h.kfec_rank}, expected 8"
    assert h.lqer_enabled == False, "lqer should be disabled"
    print("   ✓ Configuration correct")

    # Test model initialization
    print("3. Testing GPT model initialization...")
    model = train_gpt.GPT(h)
    print("   ✓ Model initialized")

    # Verify cross-layer sharing parameters exist
    print("4. Verifying cross-layer sharing parameters...")
    assert hasattr(model, "qo_base"), "Missing qo_base"
    assert hasattr(model, "qo_delta_U"), "Missing qo_delta_U"
    assert hasattr(model, "qo_delta_V"), "Missing qo_delta_V"
    assert hasattr(model, "kv_base"), "Missing kv_base"
    assert hasattr(model, "mlp_up_base"), "Missing mlp_up_base"
    assert hasattr(model, "mlp_down_base"), "Missing mlp_down_base"
    print("   ✓ All cross-layer parameters present")

    # Test _bank_weights computation
    print("5. Testing _bank_weights computation...")
    q_w, k_w, v_w, o_w, up_w, down_w = model._bank_weights(0)
    assert q_w.shape == (h.model_dim, h.model_dim), f"Wrong q_w shape: {q_w.shape}"
    print(f"   ✓ Weight computation works (q_w shape: {q_w.shape})")

    # Test forward pass
    print("6. Testing forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, h.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, h.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(input_ids)
    assert logits.shape == (batch_size, seq_len, h.vocab_size), f"Wrong logits shape: {logits.shape}"
    print(f"   ✓ Forward pass works (logits shape: {logits.shape})")

    # Test KFEC factorization
    print("7. Testing KFEC factorization...")
    E_test = torch.randn(512, 512)
    A, B, C, n_A, n_B = train_gpt._kronecker_factorize(E_test, rank=8, als_iters=2)
    print(f"   ✓ KFEC factorization works (A:{A.shape}, B:{B.shape}, C:{C.shape})")

    # Test KFEC packing
    print("8. Testing KFEC packing...")
    qA, sA, qB, sB, qC, sC = train_gpt._kfec_pack(A, B, C, bits=4)
    print(f"   ✓ KFEC packing works")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nPriority 1 implementation is ready to run on RunPod.")
    print("\nEstimated parameter reduction:")

    # Calculate parameter savings
    if h.cross_layer_sharing:
        # Original
        orig_qo = 2 * h.num_layers * h.model_dim * h.model_dim
        orig_kv = 2 * h.num_layers * (h.num_kv_heads * (h.model_dim // h.num_heads)) * h.model_dim
        orig_mlp_up = h.num_layers * int(h.mlp_mult * h.model_dim) * h.model_dim
        orig_mlp_down = h.num_layers * h.model_dim * int(h.mlp_mult * h.model_dim)
        orig_total = orig_qo + orig_kv + orig_mlp_up + orig_mlp_down

        # New
        kv_dim = h.num_kv_heads * (h.model_dim // h.num_heads)
        hidden_dim = int(h.mlp_mult * h.model_dim)
        new_base = (2 * h.model_dim * h.model_dim +
                    2 * kv_dim * h.model_dim +
                    hidden_dim * h.model_dim +
                    h.model_dim * hidden_dim)
        new_deltas = (2 * 2 * h.num_layers * h.model_dim * h.delta_rank +
                      2 * 2 * h.num_layers * kv_dim * h.delta_rank +
                      2 * h.num_layers * hidden_dim * h.delta_rank +
                      2 * h.num_layers * h.model_dim * h.delta_rank)
        new_total = new_base + new_deltas

        reduction_pct = (1 - new_total / orig_total) * 100
        print(f"  Original weight banks: {orig_total/1e6:.2f}M parameters")
        print(f"  With cross-layer sharing: {new_total/1e6:.2f}M parameters")
        print(f"  Reduction: {reduction_pct:.1f}%")

    sys.exit(0)

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
