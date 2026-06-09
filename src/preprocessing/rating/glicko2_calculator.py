import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterable, List
import numpy as np
import pandas as pd

GLICKO2_SCALE = 173.7178

@dataclass
class PlayerState:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06
    last_date: Optional[pd.Timestamp] = None
    matches: int = 0

def to_glicko2_scale(rating: float, rd: float) -> tuple[float, float]:
    mu = (rating - 1500.0) / GLICKO2_SCALE
    phi = rd / GLICKO2_SCALE
    return mu, phi


def from_glicko2_scale(mu: float, phi: float) -> tuple[float, float]:
    rating = 1500.0 + GLICKO2_SCALE * mu
    rd = GLICKO2_SCALE * phi
    return rating, rd

def g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))

def E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))


def volatility_update(phi: float, sigma: float, delta: float, v: float, tau: float, eps: float = 1e-6) -> float:
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA = f(A)
    fB = f(B)

    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)

        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA /= 2.0

        B = C
        fB = fC

    return math.exp(A / 2.0)

def age_player_rd(player: PlayerState, current_date: pd.Timestamp, period_days: int = 7) -> None:
    if player.last_date is None:
        return

    current_date = pd.to_datetime(current_date, format="%Y%m%d")
    last_date = pd.to_datetime(player.last_date, format="%Y%m%d")
    days_inactive = (current_date - last_date).days
    if days_inactive <= 0:
        return
    periods = math.ceil(days_inactive / period_days)
    mu, phi = to_glicko2_scale(player.rating, player.rd)
    phi = math.sqrt(phi * phi + periods * (player.vol ** 2))

    player.rating, player.rd = from_glicko2_scale(mu, phi)

def update_player_vs_one_opponent(player: PlayerState, opp: PlayerState, score: float, tau: float = 0.5) -> tuple[float, float]:
    mu, phi = to_glicko2_scale(player.rating, player.rd)
    mu_j, phi_j = to_glicko2_scale(opp.rating, opp.rd)

    g_j = g(phi_j)
    E_j = E(mu, mu_j, phi_j)

    v = 1.0 / (g_j * g_j * E_j * (1.0 - E_j))
    delta = v * g_j * (score - E_j)

    sigma_prime = volatility_update(phi=phi, sigma=player.vol, delta=delta, v=v, tau=tau)
    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * g_j * (score - E_j)

    new_rating, new_rd = from_glicko2_scale(mu_prime, phi_prime)
    rating_delta = new_rating - player.rating

    player.rating = new_rating
    player.rd = new_rd
    player.vol = sigma_prime
    player.matches += 1

    return E_j, rating_delta