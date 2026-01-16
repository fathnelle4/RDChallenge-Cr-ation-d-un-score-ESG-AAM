#%%
# =============================================================================
# SETUP
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

from tools import (
    compute_esg_score,
    compute_temperature,
    compute_tracking_error,
    compute_turnover,
    portfolio_with_drift
)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

DATA_PATH = "data/"

metadata   = pd.read_parquet(DATA_PATH + "metadata.parquet")
prices     = pd.read_parquet(DATA_PATH + "price.parquet")
universe   = pd.read_parquet(DATA_PATH + "universe.parquet")
esg_score  = pd.read_parquet(DATA_PATH + "esg_score.parquet")
itr        = pd.read_parquet(DATA_PATH + "itr.parquet")

# Vérifications
assert prices.columns.equals(universe.columns)
assert esg_score.columns.equals(universe.columns)
assert itr.columns.equals(universe.columns)

# =============================================================================
# DIAGNOSTICS – INDICE PARENT
# =============================================================================

# Nombre de constituants dans le temps
(universe > 0).sum(axis=1).plot(
    figsize=(10, 4),
    title="Nombre de constituants – Indice parent"
)
plt.show()

# Répartition sectorielle à la dernière date
sector_map = metadata.set_index("ID")["SECTOR"]
date = universe.index[-1]

(
    universe.loc[date]
    .groupby(sector_map)
    .sum()
    .plot(kind="bar", figsize=(10, 4),
          title=f"Répartition sectorielle – Indice parent ({date.date()})")
)
plt.show()

# =============================================================================
# PERFORMANCE DE L’INDICE PARENT (OFFICIELLE)
# =============================================================================

parent_index, parent_daily_weights = portfolio_with_drift(
    weights=universe,
    prices=prices
)

parent_returns = parent_index.pct_change().dropna()

(parent_index * 100).plot(
    figsize=(12, 5),
    title="Indice parent MSCI (base 100)"
)
plt.show()

# =============================================================================
# STATISTIQUES DE PERFORMANCE
# =============================================================================

def perf_stats(returns):
    return pd.Series({
        "Rendement annualisé": returns.mean() * 252,
        "Volatilité annualisée": returns.std() * np.sqrt(252),
        "Sharpe (rf=0)": (returns.mean() / returns.std()) * np.sqrt(252),
        "Max Drawdown": (
            (1 + returns).cumprod()
            / (1 + returns).cumprod().cummax() - 1
        ).min()
    })

perf_stats(parent_returns)

# =============================================================================
# PROFIL ESG & CLIMAT – INDICE PARENT
# =============================================================================

date = universe.index[-1]
w_parent = universe.loc[date].fillna(0)

esg_parent = compute_esg_score(esg_score, date, w_parent)
itr_parent = compute_temperature(itr, date, w_parent)

print("ESG parent :", esg_parent)
print("Température implicite parent :", itr_parent)

# Distributions pondérées
(esg_score.loc[esg_score.index < date].iloc[-1] * w_parent).dropna().hist(bins=30)
plt.title("Distribution des scores ESG pondérés – Parent")
plt.show()

(itr.loc[itr.index < date].iloc[-1] * w_parent).dropna().hist(bins=30)
plt.title("Distribution des températures implicites pondérées – Parent")
plt.show()

# =============================================================================
# ÉTAPE 2 – CONSTRUCTION DU MSCI_ESG (EXCLUSION NAÏVE)
# =============================================================================

def apply_esg_exclusion(
    weights: pd.Series,
    esg_scores: pd.Series,
    sector_map: pd.Series,
    exclusion_threshold: float = 0.30
) -> pd.Series:
    """
    Exclude the worst ESG-weighted companies per sector.
    """
    new_weights = weights.copy()

    for sector in sector_map.unique():
        ids_sector = sector_map[sector_map == sector].index
        w_sector = weights.loc[ids_sector]
        w_sector = w_sector[w_sector > 0]

        if w_sector.empty:
            continue

        esg_sector = esg_scores.loc[w_sector.index]

        df = pd.DataFrame({
            "weight": w_sector,
            "esg": esg_sector
        }).sort_values("esg")  # worst first

        df["cum_weight"] = df["weight"].cumsum()
        excluded = df[
            df["cum_weight"] <= exclusion_threshold * df["weight"].sum()
        ].index

        new_weights.loc[excluded] = 0.0

    if new_weights.sum() > 0:
        new_weights /= new_weights.sum()

    return new_weights

# Poids ESG dans le temps
esg_weights = pd.DataFrame(
    index=universe.index,
    columns=universe.columns,
    dtype=float
)

for date in universe.index:
    w_parent = universe.loc[date].fillna(0)
    available_esg_dates = esg_score.index[esg_score.index < date]

    if len(available_esg_dates) == 0:
        continue

    esg_t = esg_score.loc[available_esg_dates[-1]]

    esg_weights.loc[date] = apply_esg_exclusion(
        weights=w_parent,
        esg_scores=esg_t,
        sector_map=sector_map
    )

# =============================================================================
# DIAGNOSTICS MSCI_ESG
# =============================================================================

pd.DataFrame({
    "Parent": (universe > 0).sum(axis=1),
    "MSCI_ESG": (esg_weights > 0).sum(axis=1)
}).plot(
    figsize=(10, 4),
    title="Nombre de titres – Parent vs MSCI_ESG"
)
plt.show()

sector_parent = universe.loc[date].groupby(sector_map).sum()
sector_esg = esg_weights.loc[date].groupby(sector_map).sum()

pd.DataFrame({
    "Parent": sector_parent,
    "MSCI_ESG": sector_esg
}).plot(
    kind="bar",
    figsize=(12, 4),
    title="Répartition sectorielle – Parent vs MSCI_ESG"
)
plt.show()

# =============================================================================
# ÉTAPE 3 – PERFORMANCE MSCI_ESG (SANS OPTIMISATION)
# =============================================================================

msci_esg_index, esg_daily_weights = portfolio_with_drift(
    weights=esg_weights,
    prices=prices
)

msci_esg_returns = msci_esg_index.pct_change().dropna()

(pd.DataFrame({
    "MSCI Parent": parent_index,
    "MSCI ESG (naïf)": msci_esg_index
}) * 100).plot(
    figsize=(12, 5),
    title="Performance cumulée – Parent vs MSCI_ESG"
)
plt.show()

# =============================================================================
# TRACKING ERROR & TURNOVER
# =============================================================================

tracking_error = compute_tracking_error(
    msci_esg_returns,
    parent_returns
)

turnovers = [
    compute_turnover(
        esg_daily_weights.iloc[i-1],
        esg_daily_weights.iloc[i]
    )
    for i in range(1, len(esg_daily_weights))
]

print("Tracking Error annualisée :", tracking_error)
print("Turnover moyen :", np.mean(turnovers))


#%%
