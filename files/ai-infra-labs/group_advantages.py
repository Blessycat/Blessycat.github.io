"""Within-prompt population-standard-deviation normalization for teaching."""
import math
import statistics


def advantages(rewards, epsilon=1e-8):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Need at least two finite rewards")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    return [(reward - mean) / (std + epsilon) for reward in rewards]


if __name__ == "__main__":
    for rewards in ([1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 2, 100]):
        values = advantages(rewards)
        assert abs(sum(values)) < 1e-8
        print(f"rewards={rewards}, advantages={[round(x, 4) for x in values]}")
    assert advantages([1, 1, 1, 1]) == [0, 0, 0, 0]
    assert advantages([0, 0, 1, 1])[0] < 0
    print("This checks normalization only, not a full GRPO objective or implementation.")
