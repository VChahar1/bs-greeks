"""Save an actual comparison run with explicit option inputs."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.compare import compare


def main() -> None:
    """Write computed analytic, finite-difference and autograd sensitivities."""
    root = Path(__file__).resolve().parents[1]
    parameters = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    result = compare(**parameters)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/comparison.json").write_text(
        json.dumps({"parameters": parameters, "results": result}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
