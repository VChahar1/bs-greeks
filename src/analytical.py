"Black-Scholes Greeks via closed-form formulas."

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2, the standard BS intermediate quantities."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Black-Scholes price. Included here so analytical.py is self-contained."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Delta = dP/dS."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma = d²P/dS². Same for calls and puts."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega = dP/dσ. Same for calls and puts."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(S * np.sqrt(T) * norm.pdf(d1))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Theta = -dP/dT (negative because options lose value as T decreases).

    Convention: theta is per year. Divide by 252 for per-trading-day theta.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return float(term1 + term2)


def rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Rho = dP/dr."""
    _, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))


def all_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """Compute all five Greeks at once."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho": rho(S, K, T, r, sigma, option_type),
    }


def _demo() -> None:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(f"Analytical Greeks for ATM call (S={S}, K={K}, T={T}, r={r}, σ={sigma})")
    print("-" * 50)
    for name, value in all_greeks(S, K, T, r, sigma, "call").items():
        print(f"  {name:>8s}: {value:>10.6f}")


if __name__ == "__main__":
    _demo()
