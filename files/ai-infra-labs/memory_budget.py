"""Paper estimates, not measured peak memory. Python 3.10+, standard library."""
import argparse
import json


def estimate(parameters, layers, kv_heads, head_dim, tokens, batch):
    if min(parameters, layers, kv_heads, head_dim, tokens, batch) <= 0:
        raise ValueError("All inputs must be positive")
    gib = 1024 ** 3
    return {
        "bf16_weights_GiB": parameters * 2 / gib,
        "ideal_4bit_weights_GiB": parameters * 0.5 / gib,
        "example_adam_static_16_bytes_per_parameter_GiB": parameters * 16 / gib,
        "bf16_kv_cache_GiB": 2 * layers * kv_heads * head_dim * tokens * batch * 2 / gib,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=int, default=7_000_000_000)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    values = estimate(args.parameters, args.layers, args.kv_heads,
                      args.head_dim, args.tokens, args.batch)
    # Independent hand calculation: 32 layers, 8 KV heads, 4096 tokens = 0.5 GiB.
    assert estimate(1, 32, 8, 128, 4096, 1)["bf16_kv_cache_GiB"] == 0.5
    print(json.dumps(values, indent=2))
    print("Excludes activations, quantization metadata, workspaces and fragmentation.")
