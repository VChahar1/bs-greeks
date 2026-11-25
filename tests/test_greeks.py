"Tests for the Greeks implementations."

from __future__ import annotations

import pytest

from src.analytical import all_greeks, delta, gamma, vega, theta, rho, bs_price
from src.finitediff import all_greeks_fd
from src.pathwise import all_greeks_autograd


PARAMS = (100, 100, 1.0, 0.05, 0.2)


def test_call_delta_between_0_and_1():
    d = delta(*PARAMS, "call")
    assert 0 < d < 1


def test_put_delta_between_neg1_and_0():
    d = delta(*PARAMS, "put")
    assert -1 < d < 0


def test_call_put_delta_differ_by_one():
    """Put-call parity implies: delta_call - delta_put = 1."""
    dc = delta(*PARAMS, "call")
    dp = delta(*PARAMS, "put")
    assert abs((dc - dp) - 1.0) < 1e-10


def test_gamma_positive():
    assert gamma(*PARAMS) > 0


def test_vega_positive():
    assert vega(*PARAMS) > 0


def test_theta_negative_for_call():
    """ATM call theta is negative (option loses value with time)."""
    assert theta(*PARAMS, "call") < 0


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_fd_matches_analytical(greek):
    ana = all_greeks(*PARAMS, "call")
    fd = all_greeks_fd(*PARAMS, "call")

    tol = 0.01 if greek == "theta" else 0.001
    assert abs(fd[greek] - ana[greek]) < tol, (
        f"{greek}: FD={fd[greek]:.6f} vs analytical={ana[greek]:.6f}"
    )


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_autograd_matches_analytical(greek):
    ana = all_greeks(*PARAMS, "call")
    ad = all_greeks_autograd(*PARAMS, "call")

    tol = 1e-6
    assert abs(ad[greek] - ana[greek]) < tol, (
        f"{greek}: autograd={ad[greek]:.6f} vs analytical={ana[greek]:.6f}"
    )


def test_deep_itm_call_delta_near_one():
    d = delta(200, 100, 1.0, 0.05, 0.2, "call")
    assert d > 0.99


def test_deep_otm_call_delta_near_zero():
    d = delta(50, 100, 1.0, 0.05, 0.2, "call")
    assert d < 0.01


def test_put_greeks_fd_match():
    ana = all_greeks(*PARAMS, "put")
    fd = all_greeks_fd(*PARAMS, "put")
    for greek in ["delta", "vega", "rho"]:
        assert abs(fd[greek] - ana[greek]) < 0.001


def test_put_greeks_autograd_match():
    ana = all_greeks(*PARAMS, "put")
    ad = all_greeks_autograd(*PARAMS, "put")
    for greek in ["delta", "vega", "rho"]:
        assert abs(ad[greek] - ana[greek]) < 1e-6
