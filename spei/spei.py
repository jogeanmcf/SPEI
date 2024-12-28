import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange
import scipy.stats as stats


def sum_12(arr: np.ndarray) -> np.ndarray:
    """"""

    num_to_add = 12 - len(arr) % 12
    if num_to_add == 12:
        num_to_add = 0
    if arr.ndim > 1:
        raise ValueError("Array must be 1D")
    arr = np.concatenate([arr, np.zeros(num_to_add)])
    return np.outer(
        arr.reshape(-1, 12).sum(axis=1), np.ones(12)
    ).flatten()  # TODO: right now this works only for arry lenth multiples of 12


def NDM(date: datetime):
    """
    Calculate the number of days in a given month (NDM).

    Parameters:
    year (int): The year.
    month (int): The month (1 for January, 2 for February, etc.).

    Returns:
    int: The number of days in the specified month.
    """
    # monthrange returns a tuple (weekday of the first day, number of days in the month)
    return monthrange(date.year, date.month)[1]


def average_julian_day(date: datetime):
    """
    Calculate the average Julian day for a given month.

    Parameters:
    year (int): The year.
    month (int): The month (1 for January, 2 for February, etc.).
    """
    # Calculate the first and last day of the month
    first_day = datetime(date.year, date.month, 1)
    if date.month == 12:
        # Handle December as the last month of the year
        last_day = datetime(date.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(date.year, date.month + 1, 1) - timedelta(days=1)

    # Calculate the Julian day for the first and last days
    julian_first = first_day.timetuple().tm_yday
    julian_last = last_day.timetuple().tm_yday

    # Return the average
    return (julian_first + julian_last) / 2


def m_coeficient(I):
    return 6.75e-07 * np.power(I, 3) - 7.71e-05 * np.power(I, 2) + 0.01792 * I + 0.492


def i_coefficient(tas: np.ndarray):  # TODO: document this funciton
    return np.power(tas / 5, 1.514)


def calculate_pet(
    tas: np.ndarray, date: np.ndarray[datetime], lat: np.ndarray
) -> np.ndarray:
    """
    Calculate the potential evapotranspiration for a given temperature, month and latitude

    :param tas: temperature in degrees Celsius
    :param date: date corresponding to the temperature measurements
    :param lat: latitude associated with the temperature local measurements
    """
    J = np.asarray([average_julian_day(d) for d in date])
    delta = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    omega = np.arccos(-np.outer(np.tan(np.deg2rad(lat)), np.tan(delta)))
    N = 24 / np.pi * omega
    _NDM = np.asarray([NDM(d) for d in date])
    K = N / 12 * (_NDM / 12)
    i = i_coefficient(tas)
    I = sum_12(
        i
    )  # TODO: I is a heat index, which is calculated as the sum of 12 monthly index values in the original paper]
    del i  # freeing memory
    del N  # freeing memory
    return (16 * K * np.power(10 * tas / I, m_coeficient(I))).flatten()


def calculate_spei(
    pr: np.ndarray,
    tas: np.ndarray,
    date: np.ndarray[datetime],
    lat: np.ndarray,
):
    """
    Calculate the Standardized Precipitation Evapotranspiration Index (SPEI).

    Parameters
    ----------
    pr : array_like
        Time series of precipitation values.
    tas : array_like
        temperatur in degrees Celsius.
    date : array_like
        Time series of datetime values corresponding with the pr and tas measures.
    lat : array_like
    distribution : str, optional
        Distribution to fit to the data. Default is 'Gamma'.
    periodicity : str, optional
        Time periodicity of the data. Default is 'monthly'.

    Returns
    -------
    spei : array_like
        Standardized Precipitation Evapotranspiration Index.
    """
    pet = calculate_pet(tas, date, lat)
    d = pr - pet
    shape, loc, scale = stats.gamma.fit(d)
    probabilityes = stats.gamma.cdf(d, shape, loc, scale)
    spei = stats.norm.ppf(probabilityes)
    return spei
