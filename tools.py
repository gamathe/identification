import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import *

from math import sqrt, atan, log, exp, sin, cos, tan, trunc, isclose, tanh

import datetime
from datetime import timezone
from datetime import time
from datetime import timedelta

import calendar

import pytz
from functools import reduce


def azimuth_norm(az):
    # azimuth comprised between -360° and 360°
    az = az - 360 * trunc(az/360)
    # azimuth comprised between -180° and 180°"
    if (abs(az) > 180) :
        if (az > 180) : 
            az = az - 360
        else:
            az = az + 360
    return az


def failure_durations(dfinit, lstcol):

    df = dfinit.copy()

    # Create boolean masks m1 and m2 such that 
    # m1 represents the condition where the difference between successive dates is 15 min 
    # m2 represent the condition where the given column does contain NaN values
    
    s   = pd.to_datetime(df.index.to_series(), format='%Y-%m-%d %H:%M:%S')
    s   = s[~s.index.duplicated(keep='first')]  # Suppress duplicates
    m1  = s.diff().view('int64').eq(15 * 60 * 10**9) # time difference in pandas is in nanosecond resolution
    m2  = df[lstcol].isna().any(axis=1)
    
    # count contiguous NaN values and get the start and final date of the NaN contiguous periods
    out = s[m2].groupby([(~m1).cumsum(), (~m2).cumsum()]).agg(['first', 'last', 'count']).reset_index(drop=True)
    out['failure_hours'] = out['count'] / 4 
    out = out.drop(columns=['count'])
    out = out.sort_values(by = 'failure_hours', ascending = True)
    out['failure_hours'].value_counts().sort_index().head(15)

    return out
    

def no_failure_durations(dfinit, lstcol):

    df = dfinit.copy()

    # Create boolean masks m1 and m2 such that 
    # m1 represents the condition where the difference between successive dates is 15 min 
    # m2 represent the condition where the given column does not contain NaN values
    s   = pd.to_datetime(df.index.to_series(), format='%Y-%m-%d %H:%M:%S')
    s   = s[~s.index.duplicated(keep='first')]  # Suppress duplicates
    m1  = s.diff().view('int64').eq(15 * 60 * 10**9) # time difference in pandas is in nanosecond resolution
    m2  = df[lstcol].notna().all(axis=1)
    
    # count contiguous NaN values and get the start and final date of the NaN contiguous periods
    out = s[m2].groupby([(~m1).cumsum(), (~m2).cumsum()]).agg(['first', 'last', 'count']).reset_index(drop=True)
    out['no_failure_days'] = out['count'] / 4 / 24
    out = out.drop(columns=['count'])

    return out
    

def interpolate_list(df_init, lstcol, failure_max_h):

    df = df_init.copy()

    # Ensure datetime index
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    df = df.set_index('date_utc')
    df['date_utc'] = df.index

    # Time difference in minutes (for detecting 15-min steps)
    dt = df.index.to_series().diff().dt.total_seconds().div(60)

    # A new segment starts whenever time diff != 15 min
    new_segment = dt.ne(15).cumsum()

    for col in lstcol:

        # Mask where col is NaN
        na_mask = df[col].isna()

        # Identify NaN segments by grouping on:
        #  - segment breaks (new_segment)
        #  - non-NaN breaks (~na_mask).cumsum()
        seg_id = (new_segment + (~na_mask).cumsum())

        # Compute segment lengths in hours
        seg_sizes = (
            na_mask.groupby(seg_id)
                   .sum()                # number of NaN points
                   .mul(15/60)           # convert #points → hours (each = 0.25h)
        )

        # Segments allowed to interpolate
        allowed_segments = seg_sizes[seg_sizes <= failure_max_h].index

        # Global mask of "interpolable" NaN values
        allowed_mask = na_mask & seg_id.isin(allowed_segments)

        # Interpolate only these NaN values
        # Trick: temporarily replace *only allowed* NaN with real NaN, block others using placeholders
        s = df[col].copy()
        s.loc[~allowed_mask & na_mask] = np.inf  # placeholder impossible value
        s = s.replace(np.inf, np.nan).interpolate(limit_area="inside")

        # Assign back only the interpolated NaNs
        df[col] = df[col].where(~allowed_mask, s)

    return df
