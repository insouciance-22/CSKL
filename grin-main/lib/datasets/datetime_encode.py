import numpy as np
import pandas as pd
from typing import Union, List


def ensure_list(value: any) -> List:
    """Ensure that the input is a list."""
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    else:
        return [value]


def time_unit_to_nanoseconds(time_unit: str) -> int:
    """Convert time unit to nanoseconds."""
    valid_units = ['year', 'week', 'day', 'hour', 'minute', 'second']
    if time_unit not in valid_units:
        raise ValueError(f"Invalid time unit: {time_unit}")

    if time_unit == 'year':
        return 365.2425 * 24 * 60 * 60 * 10 ** 9
    elif time_unit == 'week':
        time_unit = 'W'

    return pd.Timedelta('1' + time_unit).value


def datetime_encoded(df: pd.DataFrame, units: Union[str, List[str]]) -> pd.DataFrame:
    """Transform dataset's temporal index into covariates using sinusoidal transformations.
    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.
        units (str or list): Temporal units to be encoded (e.g., 'day', 'week', 'hour').

    Returns:
        pd.DataFrame: DataFrame containing the sin/cos encoded features for each unit.
    """
    # Ensure the units is a list (to handle both single unit and list of units)
    units = ensure_list(units)

    # Ensure the DataFrame index is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be of datetime type.")

    index_nano = df.index.view(np.int64)  # Convert datetime index to nanoseconds
    datetime_features = {}

    for unit in units:
        nano_unit = time_unit_to_nanoseconds(unit)  # Convert the time unit to nanoseconds
        nano_sec = index_nano * (2 * np.pi / nano_unit)  # Scaling factor based on unit

        # Create sin/cos transformations for each time unit
        datetime_features[f'{unit}_sin'] = np.sin(nano_sec)
        datetime_features[f'{unit}_cos'] = np.cos(nano_sec)

    # Return the new DataFrame with the encoded features
    return pd.DataFrame(datetime_features, index=df.index, dtype=np.float32)


# # Example usage:
# # Let's say df has a datetime index
# date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
# df = pd.DataFrame({'value': np.random.randn(len(date_rng))}, index=date_rng)
#
# encoded_df = datetime_encoded(df, ['day', 'hour'])
# print(encoded_df.head())
