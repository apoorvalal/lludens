import re


def parse_binary_action(text: str) -> str:
    """Normalize an LLM response to `Cooperate` or `Defect`."""
    normalized = str(text).strip().lower()
    final_match = re.search(
        r"(?:final\s+)?(?:answer|action|choice)\s*(?:is|:)?\s*(cooperate|defect|c|d)\b",
        normalized,
    )
    if final_match:
        token = final_match.group(1)
        return "Cooperate" if token in {"cooperate", "c"} else "Defect"

    tokens = re.findall(r"[A-Za-z]+", normalized)
    if not tokens:
        raise ValueError(f"Could not parse action from empty response: {text!r}")

    aliases = {
        "cooperate": "Cooperate",
        "cooperates": "Cooperate",
        "cooperated": "Cooperate",
        "cooperating": "Cooperate",
        "collaborate": "Cooperate",
        "collaborates": "Cooperate",
        "c": "Cooperate",
        "defect": "Defect",
        "defects": "Defect",
        "defected": "Defect",
        "defecting": "Defect",
        "d": "Defect",
    }

    if len(tokens) == 1 and tokens[0] in aliases:
        return aliases[tokens[0]]

    mentioned = [(idx, aliases[token]) for idx, token in enumerate(tokens) if token in aliases]
    if len({action for _, action in mentioned}) == 1:
        return mentioned[0][1]
    if mentioned:
        return min(mentioned, key=lambda item: item[0])[1]

    first = tokens[0]
    if first in {"cooperate", "cooperates", "cooperated", "cooperating", "c"}:
        return "Cooperate"
    if first in {"defect", "defects", "defected", "defecting", "d"}:
        return "Defect"

    raise ValueError(f"Could not parse action from response: {text!r}")
