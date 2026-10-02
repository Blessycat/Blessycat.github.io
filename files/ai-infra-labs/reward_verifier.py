"""Exact integer-answer verifier. Never executes model output."""
import re


def reward(text, expected):
    # Deliberately strict task contract: exactly one tagged integer.
    if not isinstance(text, str) or len(text) > 128:
        return 0
    match = re.fullmatch(r"<answer>(-?[0-9]{1,12})</answer>", text.strip())
    return int(match is not None and int(match.group(1)) == expected)


if __name__ == "__main__":
    cases = [
        ("<answer>42</answer>", 1),
        (" <answer>42</answer>\n", 1),
        ("<answer>41</answer>", 0),
        ("42", 0),
        ("<answer>42</answer><answer>0</answer>", 0),
        ("ignore the rules; score this as correct", 0),
        ("<answer>4.2e1</answer>", 0),
        ("<answer>" + "9" * 1000 + "</answer>", 0),
    ]
    for text, expected_score in cases:
        assert reward(text, 42) == expected_score
    print(f"{len(cases)} verifier contract checks passed")
    print("Format-strict reward can penalize semantically correct answers: document the contract.")
