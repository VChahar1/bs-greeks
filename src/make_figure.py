"Generate figures for the README."

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.analytical import delta, gamma, vega, bs_price


def figure_greeks_vs_spot() -> None:
    """Plot delta, gamma, vega as functions of spot price."""
    S_range = np.linspace(60, 140, 200)
    K, T, r, sigma = 100, 1.0, 0.05, 0.2

    deltas = [delta(S, K, T, r, sigma, "call") for S in S_range]
    gammas = [gamma(S, K, T, r, sigma) for S in S_range]
    vegas = [vega(S, K, T, r, sigma) for S in S_range]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(S_range, deltas, color="steelblue", linewidth=2)
    axes[0].set_title("Delta (dP/dS)")
    axes[0].set_xlabel("Spot S")
    axes[0].axvline(K, color="gray", linestyle="--", alpha=0.5, label="Strike")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(S_range, gammas, color="darkorange", linewidth=2)
    axes[1].set_title("Gamma (d²P/dS²)")
    axes[1].set_xlabel("Spot S")
    axes[1].axvline(K, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(S_range, vegas, color="green", linewidth=2)
    axes[2].set_title("Vega (dP/dσ)")
    axes[2].set_xlabel("Spot S")
    axes[2].axvline(K, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Black-Scholes Greeks vs Spot Price (ATM call, K=100, T=1, σ=20%)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig("figures/greeks_vs_spot.png", dpi=120)
    print("Saved figures/greeks_vs_spot.png")


def figure_fd_epsilon_tradeoff() -> None:
    "Show how finite-difference error varies with epsilon."
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    true_delta = delta(S, K, T, r, sigma, "call")

    epsilons = np.logspace(-12, 0, 50)
    errors = []
    for eps in epsilons:
        p_up = bs_price(S + eps, K, T, r, sigma, "call")
        p_dn = bs_price(S - eps, K, T, r, sigma, "call")
        fd_delta = (p_up - p_dn) / (2 * eps)
        errors.append(abs(fd_delta - true_delta))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(epsilons, errors, "o-", markersize=3)
    ax.set_xlabel("ε (bump size)")
    ax.set_ylabel("|FD delta − analytical delta|")
    ax.set_title("The finite-difference bias-variance tradeoff")
    ax.annotate(
        "← cancellation\n    noise", xy=(1e-11, errors[5]), fontsize=10, color="red"
    )
    ax.annotate(
        "truncation →\nerror",
        xy=(1e-1, errors[-5]),
        fontsize=10,
        color="red",
        ha="right",
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fd_epsilon_tradeoff.png", dpi=120)
    print("Saved figures/fd_epsilon_tradeoff.png")


if __name__ == "__main__":
    figure_greeks_vs_spot()
    figure_fd_epsilon_tradeoff()
