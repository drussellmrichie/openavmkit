"""
Kolbe, et al.
-----------
Implementation of the Kolbe et al. (2023) paper.

**Experimental and WIP - not yet ready for production use.**
"""

from scipy.spatial import cKDTree
from scipy.special import comb

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import trange

from hilbertcurve.hilbertcurve import HilbertCurve

from openavmkit.data import (
    SalesUniversePair,
    get_hydrated_sales_from_sup,
    get_sale_field,
)
from openavmkit.utilities.data import div_df_z_safe
from openavmkit.utilities.settings import area_unit


def difference_weights(m: int) -> np.ndarray:
    """
    Return Δₘ weights that satisfy Σw = 0 and ‖w‖₂ = 1.

    For m=10 returns the asymptotically optimal weights tabulated in Hall, Kay &
    Titterington (1990, *Biometrika* 77, Table 1), as cited and used by Kolbe et al.
    (2012).  These achieve ~95 % efficiency relative to the fully-optimal estimator
    (Yatchew, 1997).

    For other values of m falls back to binomial difference coefficients
    ``(-1)^s C(m,s)`` normalised to unit L2 norm, which also satisfy the two
    constraints but are not asymptotically optimal.

    Parameters
    ----------
    m : int
        Difference order (number of lags; returned array has length m+1).

    Returns
    -------
    np.ndarray
        Array of length ``m+1`` with ``d[s]`` being the weight applied to the
        observation ``s`` lags back.  Satisfies Σd = 0 and ‖d‖₂ = 1.
    """
    # Hall et al. (1990) Table 1 optimal weights, keyed by difference order m.
    # Weights are ordered d_0, d_1, …, d_m  (d_s applied to y_{i-s}).
    _HALL_WEIGHTS: dict[int, list[float]] = {
        10: [0.1995,  0.0539,  0.0104, -0.0140, -0.0325,
             0.8510, -0.2384, -0.2079, -0.1882, -0.1830, -0.2507],
    }
    if m in _HALL_WEIGHTS:
        w = np.array(_HALL_WEIGHTS[m], dtype=float)
        # Re-normalise to correct for rounding in the tabulated values.
        return w / np.linalg.norm(w)
    # Fallback: binomial weights (-1)^s C(m,s), valid but not optimal.
    w = np.array([(-1) ** s * comb(m, s) for s in range(m + 1)], dtype=float)
    return w / np.linalg.norm(w)


def hilbert_order(lat: np.ndarray, lon: np.ndarray, n_bits: int = 16) -> np.ndarray:
    """
    Return indices that sort (lat, lon) via a Hilbert curve.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitude values.
    lon : np.ndarray
        Array of longitude values.
    n_bits : int, optional
        Number of bits for Hilbert curve resolution. Defaults to 16.

    Returns
    -------
    np.ndarray
        Array of indices that sort points along the Hilbert curve.
    """

    if lat.size != lon.size:
        raise ValueError("lat and lon must have the same length")

    # Scale each axis to the integer grid [0, 2ⁿ_bits‑1]
    lat_scaled = (
        (lat - lat.min()) / (lat.max() - lat.min()) * (2**n_bits - 1)
    ).astype(int)
    lon_scaled = (
        (lon - lon.min()) / (lon.max() - lon.min()) * (2**n_bits - 1)
    ).astype(int)

    coords_int = np.stack((lat_scaled, lon_scaled), axis=1).tolist()

    hc = HilbertCurve(n_bits, 2)
    dist_1d = hc.distances_from_points(coords_int, match_type=True)

    return np.argsort(dist_1d)


def adaptive_weights_smoothing(
    resid: np.ndarray,
    coords: np.ndarray,
    *,
    k_neighbors: int,
    h0: float = 500.0,
    n_iter: int = 6,
    alpha: float = 0.6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform adaptive weights smoothing on residuals using spatial coordinates.

    This function computes smoothing weights for each observation based on its residual value
    and the distances to its k nearest neighbors. The initial bandwidth `h0` is iteratively
    updated over `n_iter` iterations according to the adaptive smoothing parameter `alpha`.

    Parameters
    ----------
    resid : np.ndarray
        Array of residual values for each observation (shape `(n,)`).
    coords : np.ndarray
        Array of spatial coordinates for each observation (shape `(n, 2)`).
    k_neighbors : int
        Number of nearest neighbors to consider when computing local weights.
    h0 : float, optional
        Initial bandwidth for smoothing (distance threshold), by default 500.0.
    n_iter : int, optional
        Number of iterations to perform for adaptive bandwidth adjustment, by default 6.
    alpha : float, optional
        Adaptation rate for updating the bandwidth at each iteration, by default 0.6.
    verbose : bool, optional
        If True, print progress information during iterations, by default False.

    Raises
    ------
    ValueError
        If `k_neighbors` is not a positive integer.

    Returns
    -------
    np.ndarray
        Array of smoothed weights (shape `(n,)`), normalized so that they sum to one.
    """

    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    tree = cKDTree(coords)
    dists, neigh_idx = tree.query(coords, k=k_neighbors)

    finite = resid[~np.isnan(resid)]
    sigma = np.nanmedian(np.abs(finite - np.nanmedian(finite))) / 0.6745
    sigma = max(float(sigma), 1e-8)

    a_hat = np.where(np.isnan(resid), float(np.nanmedian(finite)), resid)
    h = np.full_like(resid, h0, dtype=float)

    for _ in trange(n_iter, disable=not verbose, desc="AWS k‑NN"):
        new_vals = np.empty_like(a_hat)
        for i in range(resid.size):
            idx = neigh_idx[i]
            mask = ~np.isnan(resid[idx])
            if not mask.any():
                new_vals[i] = a_hat[i]
                continue
            d = dists[i][mask]
            r_j = resid[idx][mask]
            a_j = a_hat[idx][mask]

            K_dist = np.exp(-(d**2) / (2.0 * h[i] ** 2))
            K_sim = np.exp(-((a_hat[i] - a_j) ** 2) / (2.0 * (alpha * sigma) ** 2))
            w = K_dist * K_sim
            w_sum = w.sum()
            new_vals[i] = a_hat[i] if w_sum == 0 else np.dot(w, r_j) / w_sum

            flatness = np.mean(np.abs(a_hat[i] - a_j)) if w_sum else 0.0
            h[i] = np.clip(h[i] * (1 - 0.5 * flatness / (alpha * sigma)), h0 * 0.3, h0)
        a_hat = new_vals
    return a_hat


def kolbe_et_al_estimate(
    sup: SalesUniversePair,
    bldg_fields: list[str],
    model_group: str,
    settings: dict,
    params: dict = None,
    verbose: bool = False,
) -> tuple[sm.regression.linear_model.RegressionResults, pd.Series, pd.Series]:
    """
    Estimate adaptive weights smoothing (AWS) residuals using the Kolbe et al. (2023) method for a given model group.

    Parameters
    ----------
    sup : SalesUniversePair
        Sales and universe data pair containing sales data and universe information.
    bldg_fields : list[str]
        List of building fields to include in the estimation.
    model_group : str
        The model group to filter the sales and universe data.
    settings : dict
        Settings dictionary containing configuration parameters.
    params : dict, optional
        Dictionary of parameters for the estimation, including:
    verbose : bool, optional
        If True, print progress information during iterations, by default False.

    Returns
    -------
    tuple[sm.regression.linear_model.RegressionResults, pd.Series, pd.Series]
        A tuple containing:
        - Regression results from the OLS model.
        - ``land_ratio``: Series (indexed by parcel key) of exp(AWS-smoothed
          log-residual), i.e. price/predicted_structure_price.  Values ≥ 1
          indicate positive land value; values < 1 are clipped to 0 by callers.
        - ``struct_psf``: Series (indexed by parcel key) of exp(z′β̂ + const),
          i.e. OLS-predicted building contribution in $/sqft of land area.
    """
    if params is None:
        params = {}
    
    unit = area_unit(settings)
    
    k_neighbors = params.get("k_neighbors", 60)
    diff_order = params.get("diff_order", 10)
    h0 = params.get("pilot_bandwidth", 600.0)
    n_iter = params.get("n_iter", 4)

    df_sales = get_hydrated_sales_from_sup(sup)
    df_univ = sup.universe

    # Select the model group:
    df_sales = df_sales[df_sales["model_group"].eq(model_group)].copy()
    df_univ = df_univ[df_univ["model_group"].eq(model_group)].copy()

    sale_field = get_sale_field(settings)
    
    unit = area_unit(settings)
    
    # ensure we have the columns we need:
    necessary_cols = [
        sale_field,
        "latitude",
        "longitude",
        f"land_area_{unit}",
    ] + bldg_fields
    for field in necessary_cols:
        if field not in df_sales.columns:
            raise ValueError(f"Sales dataframe must have a '{field}' field")
        if field not in df_univ.columns:
            if field == sale_field:
                df_univ[field] = None
            else:
                raise ValueError(f"Universe dataframe must have a '{field}' field")

    # ------------------------------------------
    # 0. Construct the DataFrame
    # ------------------------------------------

    # Get only the fields we care about:
    df_univ = df_univ[["key"] + necessary_cols]

    # DF base is the sales dataframe
    df = df_sales[["key", "key_sale", "sale_date"] + necessary_cols].copy()

    # Determine which keys are not in sales but are in univ:
    df_univ_to_add = df_univ[~df_univ["key"].isin(df["key"])].copy()

    df_univ_to_add["key_sale"] = None
    df_univ_to_add["sale_date"] = None

    # Add the missing rows from df_univ_to_add to df:
    df = pd.concat([df, df_univ_to_add], ignore_index=True)
    df = df[~df["latitude"].isna() & ~df["longitude"].isna()]

    # ----------------------------------------------
    # 1. Convert to price-per-area and building-per-area
    # ----------------------------------------------

    # Log-space specification: p = log(price / land_area).
    # Using log prices rather than raw prices/area avoids systematic negative
    # land residuals in dense urban markets where OLS-predicted building
    # contributions routinely exceed total price for row houses.  The residual
    # (log_p − z′β̂) equals log(price/structure), so exp(a_hat) − 1 scaled by
    # the predicted structure price per sqft gives land value per sqft.
    _raw_p = div_df_z_safe(df, sale_field, f"land_area_{unit}").to_numpy(
        dtype=float, na_value=np.nan
    )
    df["p"] = np.where(_raw_p > 0, np.log(_raw_p), np.nan)
    p_area_cols: list[str] = []
    for col in bldg_fields:
        p_area = f"{col}_per_land_{unit}"
        df[p_area] = div_df_z_safe(df, col, f"land_area_{unit}")
        p_area_cols.append(p_area)

    # ---------------------------------------------
    # 2. Spatial ordering
    # ---------------------------------------------

    # Order the FULL dataset (sales + unsold universe) by Hilbert curve for AWS smoothing.
    order_full = hilbert_order(df["latitude"].values, df["longitude"].values)
    df = df.iloc[order_full].reset_index(drop=True)

    # For the OLS step we need sales-only: unsold universe rows have p=NaN and
    # break differencing windows (a window of diff_order+1 consecutive rows must
    # all be sold for the differenced value to be non-NaN).  With diff_order=10
    # and ~13% of rows sold this yields essentially zero valid OLS rows.
    # Fix: Hilbert-order the transacted rows separately for the OLS fit.
    # Residuals (land price = actual price/area − building contribution) can only
    # be computed for transacted properties; AWS smoothing then infers land price
    # for the full universe from those transacted residuals.
    df_sales_ols = df[df["p"].notna()].copy().reset_index(drop=True)
    order_sales = hilbert_order(df_sales_ols["latitude"].values, df_sales_ols["longitude"].values)
    df_sales_ols = df_sales_ols.iloc[order_sales].reset_index(drop=True)

    # ----------------------------------------------
    # 3. Higher-order differences (sales-only ordering)
    # ----------------------------------------------

    d = difference_weights(diff_order)

    def diff_series(s: pd.Series) -> pd.Series:
        X = np.column_stack([s.shift(k) for k in range(diff_order + 1)])
        return pd.Series(X[diff_order:] @ d, index=s.index[diff_order:])

    y_d = diff_series(df_sales_ols["p"])
    X_d = pd.DataFrame({c: diff_series(df_sales_ols[c]) for c in p_area_cols})
    X_d = sm.add_constant(X_d, has_constant='add')

    # All rows are sold so NaNs arise only at the diff_order burn-in edges.
    valid = y_d.notna() & X_d.notna().all(axis=1)
    y_d = y_d[valid]
    X_d = X_d.loc[valid]

    ols_res = sm.OLS(y_d, X_d, hasconst=True).fit(cov_type="HC1")

    # ----------------------------------------------
    # 4. AWS residual smoothing
    # ----------------------------------------------

    # Compute residuals for all rows in the full Hilbert-ordered df.
    # In log-space:
    #   residual = log(price/area) − (z′β̂ + const)
    #            = log(price / predicted_structure_price)
    #            = log(land_ratio)          where land_ratio ≥ 1 means land ≥ 0.
    # For unsold parcels: p is NaN → residual is NaN.
    # AWS smoothing initialises NaN residuals from the spatial median of sold
    # residuals and then smooths over all parcel coordinates, producing a
    # log-ratio field for the full universe.  Callers exponentiate to recover
    # the ratio, then multiply by exp(z′β̂) to obtain land price per sqft.
    resid = df["p"].iloc[diff_order:] - (
        df.loc[diff_order:, p_area_cols] @ ols_res.params[p_area_cols]
        + ols_res.params["const"]
    )
    resid_np = resid.to_numpy(dtype=float)  # ensure np.nan not pd.NA

    # convert lat/lon to planar metres (equirectangular)
    R = 6_371_000.0
    lat_rad = np.radians(df["latitude"].iloc[diff_order:].values)
    lon_rad = np.radians(df["longitude"].iloc[diff_order:].values)
    x = R * lon_rad * np.cos(lat_rad.mean())
    y = R * lat_rad
    coords_m = np.column_stack([x, y])

    a_hat = adaptive_weights_smoothing(
        resid_np,
        coords_m,
        k_neighbors=k_neighbors,
        h0=h0,
        n_iter=n_iter,
        verbose=verbose,
    )

    # Re-index by parcel key so callers can join back to sup.universe.
    # Parcels with multiple sales appear multiple times; take the mean.
    _keys = df["key"].iloc[diff_order:].values

    # a_hat is in log-ratio space; exponentiate to get the ratio
    # (price / predicted_structure_price).  Values ≥ 1 → positive land value.
    _ratio_keyed = pd.Series(np.exp(a_hat), index=_keys, name="land_ratio")
    _ratio_keyed = _ratio_keyed.groupby(level=0).mean()

    # Predicted structure price per sqft (in original $) for every parcel row.
    # exp(z′β̂ + const) = predicted price/area attributable to the building.
    _struct_log = (
        df.loc[diff_order:, p_area_cols].to_numpy(dtype=float) @ ols_res.params[p_area_cols].to_numpy(dtype=float)
        + float(ols_res.params["const"])
    )
    _struct_keyed = pd.Series(np.exp(_struct_log), index=_keys, name="struct_psf")
    _struct_keyed = _struct_keyed.groupby(level=0).mean()

    return ols_res, _ratio_keyed, _struct_keyed
