"""JZG equation of state transcription and the command line interface."""

from __future__ import annotations

import pytest

from pymd.cli import main
from pymd.eos import NIST_MC, jzg_energy, jzg_pressure


@pytest.mark.parametrize("row", NIST_MC)
def test_jzg_reproduces_nist_mc(row):
    """Independent check of the 32 transcribed MBWR coefficients against NIST SRSW MC data."""
    t, rho, u, _, p, _ = row
    assert jzg_energy(rho, t) == pytest.approx(u, abs=0.02)
    assert jzg_pressure(rho, t) == pytest.approx(p, abs=0.03)


def test_jzg_ideal_gas_limit():
    assert jzg_pressure(1e-6, 2.0) == pytest.approx(2e-6, rel=1e-3)
    assert abs(jzg_energy(1e-6, 2.0)) < 1e-4


def test_cli_backends_and_run(capsys):
    assert main(["backends"]) == 0
    assert "numpy" in capsys.readouterr().out
    assert main(["run", "--backend", "numpy", "--cells", "4", "--steps", "20", "--thermo",
                 "10"]) == 0
    out = capsys.readouterr().out
    assert "E_tot" in out and len(out.strip().splitlines()) == 5
