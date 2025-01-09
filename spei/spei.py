import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Tuple, Literal
from calendar import monthrange as _monthrange
import scipy.stats as stats


def convert_np_datetime_to_datetime(np_date: np.datetime64) -> datetime:
    """
    Convert numpy datetime64 to Python datetime object.

    Parameters:
    -----------
        np_date (np.datetime64): Numpy datetime to convert

    Returns:
    --------
        datetime: Converted Python datetime object
    """
    year, month, day = np.asarray(
        np_date.astype("datetime64[D]").astype(str).split("-")
    ).astype(int)
    return datetime(year, month, day)


def monthrange(date: Union[datetime, np.datetime64]) -> int:
    """
    Get the number of days in the month for the given date.

    Parameters:
    -----------
        date (Union[datetime, np.datetime64]): Input date

    Returns:
    --------
        int: Number of days in the month
    """
    if isinstance(date, np.datetime64):
        date = convert_np_datetime_to_datetime(date)
    return _monthrange(date.year, date.month)[1]


def days_passed_in_year(date: Union[datetime, np.datetime64]) -> int:
    """
    Calculate the number of days passed in the year up to the given date.

    Parameters:
    -----------
        date (Union[datetime, np.datetime64]): Input date

    Returns:
    --------
        int: Number of days passed in the year
    """
    if isinstance(date, np.datetime64):
        date = convert_np_datetime_to_datetime(date)

    # Calculate days in previous months
    previous_months_days = sum(
        _monthrange(date.year, month)[1] for month in range(1, date.month)
    )

    return previous_months_days + date.day


def calculate_solar_angles(
    msum_array: np.ndarray, lat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate solar angles and related parameters for PET calculation.

    Parameters:
    -----------
    msum_array : np.ndarray
        Array of day numbers in year for each time point
    lat : float
        Latitude in degrees

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        omega: Sunset hour angle
        N: Mean daily daylight hours
    """
    # converting latitude from degrees to radian
    tan_lat = np.tan(np.deg2rad(lat))

    # Solar declination angle (Delta)
    delta = 0.4093 * np.sin(((2 * np.pi * msum_array) / 365) - 1.405)

    # Calculate sunset hour angle
    tan_delta = np.tan(delta)
    tan_lat_delta = tan_lat * tan_delta
    tan_lat_delta[tan_lat_delta < -1] = -1
    tan_lat_delta[tan_lat_delta > 1] = 1
    omega = np.arccos(-tan_lat_delta)

    # Calculate mean daily daylight hours
    N = (24 / np.pi) * omega

    return omega, N


def calculate_heat_index(temperatures: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate heat index and its related exponent.

    Parameters:
    -----------
    temperatures : pd.Series
        Temperature time series with datetime index

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        I: Heat index array
        m: Empirical exponent
    """

    temperatures = temperatures.clip(lower=0)
    i = (temperatures / 5) ** 1.514
    I = i.groupby(i.index.year).sum()
    # Calculate empirical exponent
    m = 6.75e-07 * I**3 - 7.71e-05 * I**2 + 0.01792 * I + 0.492

    # Mapping values to match the monthly time resolution
    I = temperatures.index.year.map(I).values
    m = temperatures.index.year.map(m).values
    return I, m


def _calculate_pet(tas: np.ndarray, time: np.ndarray, lat: float) -> np.ndarray:
    """
    Calculate Potential Evapotranspiration (PET) using temperature data.
    NOTE: this function only works for 1D arrays
    Parameters
    ----------
    tas : np.ndarray
        Temperature data in Celsius degrees.
    time : np.ndarray
        Array of datetime values corresponding to temperature data.
    lat : np.ndarray or float
        Latitude(s) in degrees. Can be a scalar or an array.

    Returns
    -------
    np.ndarray
        Calculated PET values.

    Notes
    -----
    This implementation uses the mean instead of the sum for the heat index
    calculation as it provides better alignment with validation datasets.
    """

    # Calculate day-of-year and month length arrays
    msum_array = np.asarray([days_passed_in_year(date) for date in time])
    mlen_array = np.asarray([monthrange(date) for date in time])

    # Calculate solar parameters
    omega, N = calculate_solar_angles(msum_array, lat)

    # Calculate monthly correction factor (K)
    K = (N / 12) * (mlen_array / 30)

    # Convert temperature array to pandas Series for time-based operations
    tas_series = pd.Series(data=tas, index=time)
    # month_indices = tas_series.index.month

    # Calculate heat index and empirical exponent
    I, m = calculate_heat_index(tas_series)

    # Calculate final PET
    pet = np.zeros_like(tas)
    for index, _I in enumerate(I):
        if _I != 0:
            pet[index] = 16 * K[index] * (10 * tas[index] / _I) ** m[index]
    return pet


def calculate_pet(
    tas: np.ndarray, time: np.ndarray, lat: np.ndarray, axis=0
) -> np.ndarray:
    """
    Calculate Potential Evapotranspiration (PET) using temperature data.

    Parameters
    ----------
    tas : np.ndarray
        Temperature data in Celsius degrees.
    time : np.ndarray
        Array of datetime values corresponding to temperature data.
    lat : np.ndarray or float
        Latitude(s) in degrees. Can be a scalar or an array.
    axis : int, optional
        Axis along which to calculate PET (default is 0).

    Returns
    -------
    np.ndarray
        Calculated PET values.

    Notes
    -----
    For multiple values of latitude, the latitude axis needs to be the second
    axis (i.e lat_axis = 1)

    This implementation uses the mean instead of the sum for the heat index
    calculation as it provides better alignment with validation datasets.
    """
    pet = np.full_like(tas, fill_value=np.nan)  # Initialize PET array with NaNs.

    if np.isscalar(lat):  # Single latitude case
        pet = np.apply_along_axis(
            lambda arr: _calculate_pet(tas=arr, time=time, lat=lat), axis=0, arr=tas
        )
    else:  # Multiple latitudes case
        for i, _lat in enumerate(lat):
            pet[:, i] = np.apply_along_axis(
                lambda arr: _calculate_pet(tas=arr, time=time, lat=_lat),
                axis=axis,
                arr=tas[:, i],
            )

    return pet


def rolling_mean(array, window_size, axis=0):
    """
    Calculates the rolling mean of an array over a specified window size.

    Parameters
    ----------
    array : np.ndarray
        Input array for which the rolling mean is to be calculated.
    window_size : int
        The size of the rolling window. Must be greater than 0.
    axis : int, optional
        Axis along which the rolling mean is calculated. Default is 0.

    Returns
    -------
    np.ndarray
        Array containing the rolling mean values with the same shape as the input, padded with NaNs.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    def _rolling_mean_1d(arr, win_size):
        if win_size > len(arr):
            raise ValueError("Window size must not be larger than the input array.")

        arr = np.array(arr, dtype=float)
        kernel = np.ones(win_size) / win_size
        rolling_means = np.convolve(arr, kernel, mode="valid")

        # Pad to maintain the original length
        pad_width = (win_size - 1) // 2
        return np.pad(
            rolling_means,
            (pad_width, len(arr) - len(rolling_means) - pad_width),
            mode="constant",
            constant_values=np.nan,
        )

    return np.apply_along_axis(
        lambda arr: _rolling_mean_1d(arr, window_size), axis=axis, arr=array
    )


def fit_distribution(
    data, distribution: Literal["Gamma", "Log-Normal", "Pearson III"] = "Gamma"
):
    """
    Fits a Pearson Type III distribution to the data, excluding NaNs.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    distribution: str
        One of the available distribtion: Gamma, Log-Normal or Person III
    Returns
    -------
    tuple
        Parameters of the fitted Pearson Type III distribution.
    """
    valid_data = data[~np.isnan(data)]  # Exclude NaNs
    match distribution:
        case "Gamma":
            return stats.gamma.fit(valid_data)
        case "Log-Normal":
            return stats.fisk.fit(valid_data)
        case "Pearson III":
            return stats.pearson3.fit(valid_data)
        case _:
            raise ValueError(
                f"The distribution {distribution} is not valid. Please chose one Gamma, Log-Normal or Pearson III "
            )


def _calculate_spei(
    data: np.ndarray,
    time_scale: int = 6,
    distribution: Literal["Gamma", "Log-Normal", "Pearson III"] = "Gamma",
) -> np.ndarray:
    """
    Calculates the Standardized Precipitation Evapotranspiration Index (SPEI).
    NOTE: this function only works for 1D arrays

    Parameters
    ----------
    data : np.ndarray
        Input data array (difference between precipitation and PET).
    time_scale : int
        Time scale for rolling mean and SPEI calculation.

    Returns
    -------
    np.ndarray
        Array of SPEI values with the same shape as the input data.
    """
    # Calculate rolling mean
    smoothed_data = rolling_mean(data, time_scale, axis=0)

    # Reshape to group by month (assuming monthly data)
    reshaped_data = smoothed_data.reshape(-1, 12)

    # Fit Pearson III distribution and calculate probabilities
    distribution_params = np.apply_along_axis(
        lambda arr: fit_distribution(data=arr,distribution=distribution), axis=0, arr=reshaped_data
    )
    
    match distribution:
        case "Gamma":
            probabilities = stats.gamma.cdf(reshaped_data, *distribution_params)
        case "Log-Normal":
            probabilities = stats.fisk.cdf(reshaped_data, *distribution_params)
        case "Pearson III":
            probabilities = stats.pearson3.cdf(reshaped_data, *distribution_params)
        case _:
            raise ValueError(
                f"The distribution {distribution} is not valid. Please chose one Gamma, Log-Normal or Pearson III "
            )

    # Convert probabilities to SPEI values using the standard normal distribution
    spei = stats.norm.ppf(probabilities)

    return spei.flatten()


def calculate_spei(
    data: np.ndarray,
    time_scale: int = 6,
    axis=0,
    distribution: Literal["Gamma", "Log-Normal", "Pearson III"] = "Gamma",
) -> np.ndarray:
    """
    Calculates the Standardized Precipitation Evapotranspiration Index (SPEI).

    Parameters
    ----------
    data : np.ndarray
        Input data array (difference between precipitation and PET).
    time_scale : int
        Time scale for rolling mean and SPEI calculation.

    Returns
    -------
    np.ndarray
        Array of SPEI values with the same shape as the input data.

    NOTES
    ------
    Usually the time axis is 0, manually set it otherwise
    """
    spei = np.full_like(data, fill_value=np.nan)  # Initialize PET array with NaNs.

    spei = np.apply_along_axis(
        lambda arr: _calculate_spei(
            data=arr, time_scale=time_scale, distribution=distribution
        ),
        axis=axis,
        arr=data,
    )
    return spei
