"""Side-by-side comparison of all three methods.

This is the "cross-validation" step: if all three methods agree to high
precision on Black-Scholes (where we know the analytical answer), we
can trust the numerical methods on models where no analytical answer exists.

Run `python -m src.compare` for the comparison table.
"""

from __future__ import annotations

from src.analytical import all_greeks
from src.finitediff import all_greeks_fd
from src.pathwise import all_greeks_autograd


def compare(S: float, K: float, T: float, r: float, sigma: float,
            option_type: str = "call") -> dict:
    """Run all three methods and return a comparison dict."""
    ana = all_greeks(S, K, T, r, sigma, option_type)
    fd = all_greeks_fd(S, K, T, r, sigma, option_type)
    ad = all_greeks_autograd(S, K, T, r, sigma, option_type)

    results = {}
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        results[greek] = {
            "analytical": ana[greek],
            "finite_diff": fd[greek],
            "autograd": ad[greek],
            "fd_error": abs(fd[greek] - ana[greek]),
            "ad_error": abs(ad[greek] - ana[greek]),
        }
    return results


def _demo() -> None:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    print(f"Three-method comparison: ATM call (S={S}, K={K}, T={T}, r={r}, σ={sigma})")
    print("=" * 80)
    print(f"{'Greek':>8s}  {'Analytical':>12s}  {'Finite Diff':>12s}  "
          f"{'Autograd':>12s}  {'FD Error':>10s}  {'AD Error':>10s}")
    print("-" * 80)

    results = compare(S, K, T, r, sigma, "call")
    for greek, vals in results.items():
        print(f"{greek:>8s}  {vals['analytical']:>12.6f}  {vals['finite_diff']:>12.6f}  "
              f"{vals['autograd']:>12.6f}  {vals['fd_error']:>10.2e}  {vals['ad_error']:>10.2e}")

    print()
    print("FD error reflects truncation from the finite bump size.")
    print("AD error should be near machine epsilon (~1e-15) since autograd is exact.")


if __name__ == "__main__":
    _demo()