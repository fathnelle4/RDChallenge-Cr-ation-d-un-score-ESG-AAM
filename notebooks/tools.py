import pandas as pd
import numpy as np
import datetime

def compute_temperature(itr: pd.DataFrame, date: pd.Timestamp, weights: pd.Series) -> float:
    """
    Compute the temperature factor for a given date.

    Parameters:
    itr (pd.DataFrame): DataFrame containing the ITR values with dates as index and asset IDs as columns.
    date (pd.Timestamp): The date for which to compute the temperature factor.
    weights (pd.Series): Series containing the weights for each asset ID.

    Returns:
    
    float: The temperature factor for the specified date.
    """

    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
        date = pd.Timestamp(date)

    # Find the previous date with available ITR data
    prev_date = itr.index[itr.index < date].max()
    if pd.isna(prev_date):
        raise ValueError("No previous date with available ITR data found.")
    
    # Get the ITR values for the previous date
    itr_values = itr.loc[prev_date]

    # Compute the weighted average ITR
    temperature = (itr_values * weights).sum() / weights.sum()
    return temperature

def compute_esg_score(esg_score: pd.DataFrame, date: pd.Timestamp, weights: pd.Series) -> float:
    """
    Compute the ESG score factor for a given date.

    Parameters:
    esg_score (pd.DataFrame): DataFrame containing the ESG scores with dates as index and asset IDs as columns.
    date (pd.Timestamp): The date for which to compute the ESG score factor.
    weights (pd.Series): Series containing the weights for each asset ID.

    Returns:
    float: The ESG score factor for the specified date.
    """

    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
        date = pd.Timestamp(date)

    # Find the previous date with available ESG score data
    prev_date = esg_score.index[esg_score.index < date].max()
    if pd.isna(prev_date):
        raise ValueError("No previous date with available ESG score data found.")
    
    # Get the ESG scores for the previous date
    esg_values = esg_score.loc[prev_date]

    # Compute the weighted average ESG score
    esg_factor = (esg_values * weights).sum() / weights.sum()
    return esg_factor

def compute_turnover(prev_weights: pd.Series, current_weights: pd.Series) -> float:
    """
    Compute the turnover between two weight distributions.

    Parameters:
    prev_weights (pd.Series): Series containing the previous weights for each asset ID.
    current_weights (pd.Series): Series containing the current weights for each asset ID.

    Returns:
    float: The turnover value.
    """

    # Align the indices of both weight series
    all_ids = prev_weights.index.union(current_weights.index)
    prev_aligned = prev_weights.reindex(all_ids, fill_value=0)
    current_aligned = current_weights.reindex(all_ids, fill_value=0)

    # Compute turnover
    turnover = (prev_aligned - current_aligned).abs().sum() / 2
    return turnover

def compute_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, period: int = 252) -> float:
    """
    Compute the tracking error between portfolio returns and benchmark returns.

    Parameters:
    portfolio_returns (pd.Series): Series containing the portfolio returns.
    benchmark_returns (pd.Series): Series containing the benchmark returns.

    Returns:
    float: The tracking error value.
    """

    # Align the indices of both return series
    aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join="inner")

    # Compute tracking error
    diff = aligned_portfolio - aligned_benchmark
    tracking_error = diff.std(ddof=1) * np.sqrt(period)
    
    return tracking_error

def portfolio_with_drift(weights, prices):
    """
    Calculate portfolio returns with weight drift.
    Rebalances only on dates where weights are defined in the original weights DataFrame.

    Parameters:
    weights (pd.DataFrame): DataFrame containing the target weights with dates as index and asset IDs as columns.
    prices (pd.DataFrame): DataFrame containing the asset prices with dates as index and asset IDs as columns.

    Returns:
    pd.Series: Series containing the portfolio value over time.
    """
    returns = prices.pct_change(fill_method=None).fillna(0)
    
    # Forward fill weights to all dates (but only rebalance on original dates)
    daily_weights = weights.reindex(prices.index)
    
    # Initialize portfolio value
    portfolio_value = pd.Series(1.0, index=prices.index)
    current_weights = pd.Series(0.0, index=prices.columns)
    
    for i, date in enumerate(prices.index):
        if i == 0:
            # Initialize at first date
            if date in weights.index:
                current_weights = weights.loc[date].fillna(0)
        else:
            # Apply drift from previous day's returns
            prev_returns = returns.iloc[i]
            current_weights = current_weights * (1 + prev_returns)
            
            # Renormalize to maintain sum = 1 (optional, depends on your needs)
            weight_sum = current_weights.sum()
            if weight_sum > 0:
                current_weights = current_weights / weight_sum
            
            # Rebalance if this is a rebalancing date
            if date in weights.index:
                current_weights = weights.loc[date].fillna(0)
            
            # Store current weights
            daily_weights.iloc[i] = current_weights

            # Calculate portfolio return
            portfolio_return = (current_weights * prev_returns).sum()
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
    
    return portfolio_value, daily_weights

def portfolio_without_drift(weights, prices):
    """
    Calculate portfolio returns without weight drift.
    Rebalances on every date to the specified weights
    
    Parameters:
    weights (pd.DataFrame): DataFrame containing the target weights with dates as index and asset IDs as columns.
    prices (pd.DataFrame): DataFrame containing the asset prices with dates as index and asset IDs as columns.

    Returns:
    pd.Series: Series containing the portfolio value over time.
    """
    returns = prices.pct_change(fill_method=None).fillna(0)
    weights_filled = weights.reindex_like(prices).ffill()

    portfolio_value = (returns * weights_filled).sum(axis=1).add(1).cumprod()
    return portfolio_value, weights_filled