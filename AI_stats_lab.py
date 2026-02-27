"""
Prob and Stats Lab – Discrete Probability Distributions
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    # ----- Theoretical -----
    P_A = 4 / 52
    P_B = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = P_A * P_B_given_A

    # Independence check (not independent)
    # P(A∩B) != P(A)*P(B)

    # ----- Simulation -----
    rng = np.random.default_rng(seed=42)
    n = 200_000
    deck = np.arange(52)

    A_count = 0
    A_and_B_count = 0

    for _ in range(n):
        draw = rng.choice(deck, size=2, replace=False)
        first_ace = draw[0] < 4   # Treat first 4 cards as Aces
        second_ace = draw[1] < 4
        if first_ace:
            A_count += 1
            if second_ace:
                A_and_B_count += 1

    empirical_P_A = A_count / n
    empirical_P_B_given_A = A_and_B_count / A_count
    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error,
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    # Theoretical probabilities
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    # Simulation
    rng = np.random.default_rng(seed=42)
    n = 100_000
    samples = rng.binomial(n=1, p=p, size=n)
    empirical_P_X_1 = np.mean(samples == 1)

    # Absolute error
    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error,
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    # Theoretical probabilities
    def comb(n, k):
        return math.comb(n, k)

    P0 = comb(n, 0) * (p ** 0) * ((1 - p) ** (n - 0))
    P2 = comb(n, 2) * (p ** 2) * ((1 - p) ** (n - 2))
    P_ge_1 = 1 - P0

    # Simulation
    rng = np.random.default_rng(seed=42)
    trials = 100_000
    samples = rng.binomial(n=n, p=p, size=trials)
    empirical_P_ge_1 = np.mean(samples >= 1)

    # Absolute error
    absolute_error = abs(P_ge_1 - empirical_P_ge_1)

    return (
        P0,
        P2,
        P_ge_1,
        empirical_P_ge_1,
        absolute_error,
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    p = 1 / 6

    # Theoretical
    P1 = (5 / 6) ** (1 - 1) * p
    P3 = (5 / 6) ** (3 - 1) * p
    P_gt_4 = (5 / 6) ** 4

    # Simulation
    rng = np.random.default_rng(seed=42)
    n = 200_000
    counts = []
    for _ in range(n):
        count = 1
        while rng.integers(1, 7) != 6:
            count += 1
        counts.append(count)

    counts = np.array(counts)
    empirical_P_gt_4 = np.mean(counts > 4)
    absolute_error = abs(P_gt_4 - empirical_P_gt_4)

    return (
        P1,
        P3,
        P_gt_4,
        empirical_P_gt_4,
        absolute_error,
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    # Theoretical
    P0 = math.exp(-lam) * (lam ** 0) / math.factorial(0)
    P15 = math.exp(-lam) * (lam ** 15) / math.factorial(15)
    # For P(X >= 18), use 1 - CDF(17)
    cdf_17 = sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(0, 18))
    P_ge_18 = 1 - cdf_17

    # Simulation
    rng = np.random.default_rng(seed=42)
    n = 100_000
    samples = rng.poisson(lam=lam, size=n)
    empirical_P_ge_18 = np.mean(samples >= 18)

    # Absolute error
    absolute_error = abs(P_ge_18 - empirical_P_ge_18)

    return (
        P0,
        P15,
        P_ge_18,
        empirical_P_ge_18,
        absolute_error,
    )
