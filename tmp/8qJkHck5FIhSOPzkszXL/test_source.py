# test_source.py

import pytest
import numpy as np
from scipy.integrate import odeint
from source import derivatives

def test_odeint():
    t = np.linspace(0, 100, 1000)
    y0 = np.array([1000, 1, 0])  # 1000 susceptible, 1 infected, 0 recovered
    solution = odeint(derivatives, y0, t)
    assert np.allclose(solution[-1, :], [0, 0, 1000])  # assert all variables reach to steady state

def test_infection_rate():
    t = np.linspace(0, 100, 1000)
    y0 = np.array([1000, 1, 0])  # 1000 susceptible, 1 infected, 0 recovered
    # Change birth_rate to 0 and vaccine_rate to 1 to see if infection_rate dominates
    solution = odeint(derivatives, y0, t, args=(1,1))
    assert np.allclose(solution[-1, :], [0, 0, 1000])  # assert all variables reach to steady state