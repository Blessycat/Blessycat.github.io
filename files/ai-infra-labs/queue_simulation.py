"""Deterministic single-server FCFS queue; not a GPU serving benchmark."""
import math
import statistics


def simulate(interval, service=0.1, count=100):
    if interval <= 0 or service <= 0 or count < 1:
        raise ValueError("positive interval, service, and count required")
    finish = 0.0
    latency = []
    for index in range(count):
        arrival = index * interval
        finish = max(arrival, finish) + service
        latency.append(finish - arrival)
    ordered = sorted(latency)
    return {
        "offered_requests_per_second": 1 / interval,
        "mean_latency_s": statistics.mean(latency),
        "p95_latency_s": ordered[math.ceil(0.95 * count) - 1],
        "fraction_within_0.2s": sum(x <= 0.2 + 1e-9 for x in latency) / count,
        "completed_requests_per_second_including_drain": count / finish,
    }


if __name__ == "__main__":
    assert abs(simulate(0.2)["mean_latency_s"] - 0.1) < 1e-10
    assert simulate(0.05)["p95_latency_s"] > 1.0
    for interval in [0.2, 0.12, 0.1, 0.08, 0.05]:
        print(simulate(interval))
    print("No batching, randomness, TTFT, token streaming or hardware effects are modeled.")
