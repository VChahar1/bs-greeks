"""Black-Scholes Greeks via finite differences.

For each Greek, we bump the relevant input by ±epsilon, re-price, and
take the central-difference ratio. This is the universal method that
works for any pricing model, not just Black-Scholes.

The key tradeoff: epsilon too large → truncation error (the finite-
difference approximation of the derivative is inaccurate). Epsilon too
small → floating-point cancellation (subtracting two nearly-equal prices
loses significant digits). The sweet spot is typically epsilon ≈ 1e-4
to 1e-2 times the input value, depending on the Greek.

Run `python -m src.finitediff` for a demo.
"""

from __future__ import annotations

from src.analytical import bs_price


def delta_fd(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", eps: float = 0.01) -> float:
    """Delta via central finite difference: (P(S+ε) - P(S-ε)) / (2ε)."""
    p_up = bs_price(S + eps, K, T, r, sigma, option_type)
    p_dn = bs_price(S - eps, K, T, r, sigma, option_type)
    return (p_up - p_dn) / (2 * eps)


def gamma_fd(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", eps: float = 0.01) -> float:
    """Gamma via central second difference: (P(S+ε) - 2P(S) + P(S-ε)) / ε²."""
    p_up = bs_price(S + eps, K, T, r, sigma, option_type)
    p_mid = bs_price(S, K, T, r, sigma, option_type)
    p_dn = bs_price(S - eps, K, T, r, sigma, option_type)
    return (p_up - 2 * p_mid + p_dn) / (eps ** 2)


def vega_fd(S: float, K: float, T: float, r: float, sigma: float,
            option_type: str = "call", eps: float = 0.001) -> float:
    """Vega via central difference in sigma."""
    p_up = bs_price(S, K, T, r, sigma + eps, option_type)
    p_dn = bs_price(S, K, T, r, sigma - eps, option_type)
    return (p_up - p_dn) / (2 * eps)


def theta_fd(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", eps: float = 1 / 252) -> float:
    """Theta via forward difference in T.

    Uses forward difference (not central) because T-eps might be negative
    for near-expiry options. The negative sign matches the convention that
    theta represents value lost per unit time.
    """
    p_now = bs_price(S, K, T, r, sigma, option_type)
    p_later = bs_price(S, K, T + eps, r, sigma, option_type)
    return -(p_later - p_now) / eps


def rho_fd(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", eps: float = 0.0001) -> float:
    """Rho via central difference in r."""
    p_up = bs_price(S, K, T, r + eps, sigma, option_type)
    p_dn = bs_price(S, K, T, r - eps, sigma, option_type)
    return (p_up - p_dn) / (2 * eps)


def all_greeks_fd(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = "call") -> dict:
    """Compute all five Greeks via finite differences."""
    return {
        "delta": delta_fd(S, K, T, r, sigma, option_type),
        "gamma": gamma_fd(S, K, T, r, sigma, option_type),
        "vega": vega_fd(S, K, T, r, sigma, option_type),
        "theta": theta_fd(S, K, T, r, sigma, option_type),
        "rho": rho_fd(S, K, T, r, sigma, option_type),
    }


def _demo() -> None:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(f"Finite-difference Greeks for ATM call (S={S}, K={K}, T={T}, r={r}, σ={sigma})")
    print("-" * 50)
    for name, value in all_greeks_fd(S, K, T, r, sigma, "call").items():
        print(f"  {name:>8s}: {value:>10.6f}")


if __name__ == "__main__":
    _demo()