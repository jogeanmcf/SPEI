import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange


def NDM(year, month):
    """
    Calculate the number of days in a given month (NDM).

    Parameters:
    year (int): The year.
    month (int): The month (1 for January, 2 for February, etc.).

    Returns:
    int: The number of days in the specified month.
    """
    # monthrange returns a tuple (weekday of the first day, number of days in the month)
    return monthrange(year, month)[1]


def average_julian_day(year, month):
    """
    Calculate the average Julian day for a given month.

    Parameters:
    year (int): The year.
    month (int): The month (1 for January, 2 for February, etc.).
    """
    # Calculate the first and last day of the month
    first_day = datetime(year, month, 1)
    if month == 12:
        # Handle December as the last month of the year
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)

    # Calculate the Julian day for the first and last days
    julian_first = first_day.timetuple().tm_yday
    julian_last = last_day.timetuple().tm_yday

    # Return the average
    return (julian_first + julian_last) / 2


def m_coeficient(I):
    return 6.75e-07 * np.power(I, 3) - 7.71e-05 * np.power(I, 2) + 0.01792 * I + 0.492


def i(T):  # TODO: Rename this function to something more descriptive
    return np.power(T / 5, 1.514)


def calculate_pet(
    T: np.ndarray, year: int, month: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Calculate the potential evapotranspiration for a given temperature, month and latitude

    :param T: temperature in degrees Celsius
    :param month: month of the year
    :param phi: latitude in radians
    """
    J = average_julian_day(year, month)
    delta = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    omega = np.arccos(-np.tan(phi) * np.tan(delta))
    N = 24 / np.pi * omega
    K = N / 12 * (NDM(month) / 12)
    values = i(T)
    I = np.mean(values)
    return 16 * K * np.power(10 * T / I, m_coeficient(I))


def spei(
    precip: np.ndarray,
    pet: np.ndarray,
    scale: int = 1,
    distribution="Gamma",
    periodicity="monthly",
):
    """
    Calculate the Standardized Precipitation Evapotranspiration Index (SPEI).

    Parameters
    ----------
    precip : array_like
        Time series of precipitation values.
    pet : array_like
        Time series of potential evapotranspiration values.
    scale : int, optional
        Time scale for aggregating data. Default is 1.
    distribution : str, optional
        Distribution to fit to the data. Default is 'Gamma'.
    periodicity : str, optional
        Time periodicity of the data. Default is 'monthly'.

    Returns
    -------
    spei : array_like
        Standardized Precipitation Evapotranspiration Index.
    """
    # Check if the input data is valid
    if not isinstance(precip, (list, np.ndarray)):
        raise ValueError("precip must be a list or numpy array")
    if not isinstance(pet, (list, np.ndarray)):
        raise ValueError("pet must be a list or numpy array")

    # Calculate SPEI
    spei = np.zeros(precip.shape)
    # TODO: Implement the SPEI calculation

    return spei
