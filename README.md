## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.compare        # side-by-side comparison table
python -m src.make_figure    # regenerate figures
pytest -v                    # ~20 tests
```

## Notes and extensions

This implementation covers vanilla European options under flat Black-Scholes. Natural extensions: Greeks under Heston or local vol (where only FD and autograd methods apply), cross-gamma (d²P/dSdσ), higher-order Greeks (speed = dGamma/dS, vanna = dDelta/dσ), and the "bumped Greeks" approach used in regulatory risk calculations (where specific bump sizes are mandated by Basel rules).

The autograd approach extends naturally to Monte Carlo Greeks via pathwise differentiation: implement the path simulation in PyTorch, compute the payoff, and `.backward()` gives Greeks for every path simultaneously. This is dramatically more efficient than bumping-and-repricing for MC models with many risk factors.