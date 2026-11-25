"Black-Scholes Greeks via PyTorch autograd (pathwise differentiation)."

from __future__ import annotations

import torch
import numpy as np


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using torch.erfc, which is differentiable."""
    return 0.5 * torch.erfc(-x / np.sqrt(2))


def _standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, differentiable."""
    return torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def bs_price_torch(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    option_type: str = "call",
) -> torch.Tensor:
    """Black-Scholes price using torch tensors (differentiable)."""
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    if option_type == "call":
        return S * _standard_normal_cdf(d1) - K * torch.exp(
            -r * T
        ) * _standard_normal_cdf(d2)
    return K * torch.exp(-r * T) * _standard_normal_cdf(-d2) - S * _standard_normal_cdf(
        -d1
    )


def all_greeks_autograd(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    "Compute all five Greeks via PyTorch autograd."
    S_t = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    K_t = torch.tensor(K, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64, requires_grad=True)
    r_t = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    sig_t = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)

    price = bs_price_torch(S_t, K_t, T_t, r_t, sig_t, option_type)
    price.backward()

    return {
        "delta": float(S_t.grad),
        "gamma": _compute_gamma_autograd(S, K, T, r, sigma, option_type),
        "vega": float(sig_t.grad),
        "theta": -float(T_t.grad),
        "rho": float(r_t.grad),
    }


def _compute_gamma_autograd(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    "Gamma = d²P/dS² via autograd."
    S_t = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    K_t = torch.tensor(K, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    r_t = torch.tensor(r, dtype=torch.float64)
    sig_t = torch.tensor(sigma, dtype=torch.float64)

    price = bs_price_torch(S_t, K_t, T_t, r_t, sig_t, option_type)

    delta_val = torch.autograd.grad(price, S_t, create_graph=True)[0]

    gamma_val = torch.autograd.grad(delta_val, S_t)[0]

    return float(gamma_val)


def _demo() -> None:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(
        f"Pathwise (autograd) Greeks for ATM call (S={S}, K={K}, T={T}, r={r}, σ={sigma})"
    )
    print("-" * 50)
    for name, value in all_greeks_autograd(S, K, T, r, sigma, "call").items():
        print(f"  {name:>8s}: {value:>10.6f}")


if __name__ == "__main__":
    _demo()
