"""Two-action DPO with a uniform reference. No language model is trained."""
import math


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def loss(theta, beta=0.5):
    # Policy logits are [theta, 0]; reference log-odds is zero.
    return softplus(-beta * theta)


def gradient(theta, beta=0.5):
    z = beta * theta
    # Stable -beta * sigmoid(-z).
    return -beta * math.exp(-softplus(z))


if __name__ == "__main__":
    assert abs(loss(0) - math.log(2)) < 1e-12
    for theta in [-3.0, 0.0, 3.0]:
        h = 1e-5
        numerical = (loss(theta + h) - loss(theta - h)) / (2 * h)
        assert abs(numerical - gradient(theta)) < 1e-8
    theta = 0.0
    initial = loss(theta)
    for step in range(20):
        theta -= 0.5 * gradient(theta)
    probability = math.exp(-softplus(-theta))
    assert loss(theta) < initial and probability > 0.5
    print(f"initial_loss={initial:.6f}, final_loss={loss(theta):.6f}")
    print(f"chosen_probability={probability:.6f}; finite-difference checks passed")
