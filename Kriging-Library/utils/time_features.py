import pandas as pd
import numpy as np


def time_encoder(length, step_interval, enc_type='timeofday', scale=False):
    # For INCREASE
    time_steps = np.arange(length)
    if enc_type == 'timeofday':
        minutes_per_day = 24 * 60
        steps_per_day = minutes_per_day / step_interval
        time_of_day_in_minutes = time_steps % steps_per_day
        if scale:
            time_of_day_in_minutes = time_of_day_in_minutes / np.max(time_of_day_in_minutes)
        return time_of_day_in_minutes
    elif enc_type == 'dayofweek':
        minutes_per_week = 7 * 24 * 60
        steps_per_week = minutes_per_week / step_interval
        day_of_week_in_days = time_steps % steps_per_week
        return day_of_week_in_days
    elif enc_type == 'timeofhour':
        minutes_per_hour = 60
        steps_per_hour = minutes_per_hour / step_interval
        time_of_hour_in_minutes = time_steps % steps_per_hour
        return time_of_hour_in_minutes
    elif enc_type == 'hourofday':
        hours_per_day = 24
        steps_per_day = hours_per_day / step_interval
        time_of_day_in_hours = time_steps % steps_per_day
        return time_of_day_in_hours
    else:
        raise ValueError("Invalid interval! Supported values are 'timeofday', 'dayofweek', 'hourofday' and 'timeofhour'.")


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    n = int(24 * 60 / 15)
    a = time_encoder(50, 15)
    print(a)
    print(a.shape)
    b = torch.from_numpy(a.astype('float32'))
    b = F.one_hot(b.to(torch.int64), num_classes=n)
    print(b.shape)
    c = np.eye(n)[a.astype(int)]
    print(c.shape)
