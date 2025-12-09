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


# MONTH BY MONTH

def configure_ticks(ax, start, stop, freq_major = None, freq_minor = None):
    """Set x-axis major and minor ticks with formatting."""

    if freq_major is None:
        freq_major = '1D'  # default
    if freq_minor is None:
        freq_minor = '1H'  # default
        
    major_ticks = pd.date_range(start=start.normalize(), end=stop.normalize(), freq=freq_major)
    minor_ticks = pd.date_range(start=start, end=stop, freq= freq_minor)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    plt.xticks(major_ticks, rotation=90)  # Ensure ticks are shown


def plot_temperature(ax, day, df, curves=None, colors=None, linestyles = None ):
    """
    Plot temperature lines with selectable curves.

    Parameters:
        ax : matplotlib axis
        day : array-like, x-axis values
        df  : DataFrame, containing temperature columns
        curves : list of str, column names to plot (default all available)
        colors : list of str, matplotlib colors for each curve (optional)
    """
    ax.set_ylabel('Temperature (°C)', color='k', fontsize=14)
    
    if curves is None:
        curves = ["t_i"]  # default columns
    if colors is None:
        # default matplotlib tab colors (repeated if more curves)
        default_colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
        colors = default_colors[:len(curves)]
    if linestyles is None:
        # default matplotlib linestyles (repeated if more curves)
        default_linestyles = ['solid', 'solid', 'solid', 'solid', 'solid']
        linestyles = default_linestyles[:len(curves)]
    
    for i, col in enumerate(curves):
        color = colors[i] if i < len(colors) else None
        linestyle = linestyles[i] if i < len(colors) else None
        ax.plot(day, df[col], color=color, linestyle =linestyle, label=col)
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14)
    ax.grid(True, which='major', linestyle='dotted', alpha=0.8)
    ax.legend()


def plot_power(ax, day, df, curves=None, colors=None, linestyles = None, rolling_span = None):
    """
    Plot power lines (rolling) with selectable curves.

    Parameters:
        ax : matplotlib axis
        day : array-like, x-axis values
        df  : DataFrame, containing temperature columns
        curves : list of str, column names to plot (default all available)
        colors : list of str, matplotlib colors for each curve (optional)
    """
    ax.set_ylabel('Power (W)', color='k', fontsize=14)

    if curves is None:
        curves = ["Q_dot_g"]  # default columns
    if colors is None:
        # default matplotlib tab colors (repeated if more curves)
        default_colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple']
        colors = default_colors[:len(curves)]
    if linestyles is None:
        # default matplotlib linestyles (repeated if more curves)
        default_linestyles = ['solid', 'solid', 'solid', 'solid', 'solid']
        linestyles = default_linestyles[:len(curves)]
    if rolling_span is None:
        rolling_span = 4
    
    for i, col in enumerate(curves):
        color = colors[i] if i < len(colors) else None
        linestyle = linestyles[i] if i < len(colors) else None
        ax.plot(day, df[col].rolling(rolling_span, center=True, min_periods=1).mean(), color=color, linestyle =linestyle, label=col)
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14)
    ax.grid(True, which='major', linestyle='dotted', alpha=0.8)
    ax.legend()



# DAILY CONSUMPTIONS VALIDATION

def plot_daily_data(data, title, ax, y_max, lst_cols, E_kWh_day):
    """
    Plot daily measured and calculated energy consumption with weekend shading 
    and tolerance-based color coding.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame indexed by datetime, containing daily values for consumption.
    title : str
        Title of the plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object where the plot will be drawn.
    y_max : float
        Maximum value for the y-axis.
    colnan : str
        Column name for the NaN-filled measured data.
    colmeas : str
        Column name for the measured energy consumption.
    colcalc : str
        Column name for the calculated energy consumption.

    Notes
    -----
    - Weekends (Saturday and Sunday) are shaded in grey.
    - Gaps in data are handled by inserting NaNs so lines break cleanly.
    - Measured values are classified as:
        * within tolerance (green),
        * high consumption (red),
        * low consumption (blue).
    - All points are placed at integer x-positions corresponding to days.
    """
    
    labels = data.index.strftime('%d-%m')  # Format datetime index as 'day-month'

    colnan, colmeas, colcalc = lst_cols

    nans = data[colnan].values
    meas = data[colmeas].values
    calc = data[colcalc].values

    tolerance = 0.20 * E_kWh_day

    x = np.arange(len(data))

    mask = (nans == meas)
    
    # Identify indices where mask is True
    valid_indices = np.where(mask)[0]
    
    # Detect gaps in consecutive days
    gaps = np.where(np.diff(valid_indices) > 1)[0]  # Indices where gaps occur
    
    # Prepare x_valid and y_valid with NaN insertions
    x_valid  = x[valid_indices].tolist()
    y1_valid = nans[valid_indices].tolist()
    y2_valid = calc[valid_indices].tolist()    
    
    # Insert NaN values at gaps to break the line
    for gap_index in reversed(gaps):  # Reverse to avoid index shifting issues
        x_valid.insert(gap_index + 1, np.nan)
        y1_valid.insert(gap_index + 1, np.nan)
        y2_valid.insert(gap_index + 1, np.nan)
    
    # Convert back to numpy arrays
    x_valid = np.array(x_valid)
    y1_valid = np.array(y1_valid) # Measured values
    y2_valid = np.array(y2_valid) # Calculated values

    high_cond = (y1_valid > y2_valid + tolerance)
    low_cond  = (y1_valid < y2_valid - tolerance)
    valid_cond = (y1_valid >= y2_valid - tolerance) & (y1_valid <= y2_valid + tolerance)

    # Shade weekends 
    we = np.where(data.index.weekday >= 5)[0]
    blocks = np.split(we, np.where(np.diff(we) != 1)[0] + 1)
    
    for block in blocks:
        if len(block) > 0:
            start = block[0] - 0.5
            end   = block[-1] + 0.5   # +1 ensures Sunday is included
            ax.axvspan(
                start, end,
                color='tab:gray',
                alpha=0.2
            )

    # Shade leave days 
    leave = np.where((data.leave > 0) | (data.leave_fwb > 0))[0]
    blocks = np.split(leave, np.where(np.diff(leave) != 1)[0] + 1)
    
    for block in blocks:
        if len(block) > 0:
            start = block[0] - 0.5
            end   = block[-1] + 0.5   # +1 ensures last leave day is included
            ax.axvspan(
                start, end,
                color='tab:blue',
                alpha=0.1
            )

    ax.plot(x, calc, label='Calculated EC', color='C0', linewidth=2)
    ax.fill_between(x_valid, y2_valid - tolerance, y2_valid + tolerance, alpha=0.2)

    # Split valid_cond into True high and low subsets ---
    x_valid_true = x_valid[valid_cond]
    y1_valid_true = y1_valid[valid_cond]

    x_valid_high = x_valid[high_cond]
    y1_valid_high = y1_valid[high_cond]

    x_valid_low = x_valid[low_cond]
    y1_valid_low = y1_valid[low_cond]

    # Plot calculated EC points with conditional colors
    ax.plot(x_valid_true, y1_valid_true, marker='o', linestyle='None', 
            color='C2', label='Measured EC (within tolerance)')
    ax.plot(x_valid_high, y1_valid_high, marker='o', linestyle='None', 
            color='C3', label='Measured EC (high consumtion)')
    ax.plot(x_valid_low, y1_valid_low, marker='o', linestyle='None', 
            color='C0', label='Measured EC (low consumption)')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy consumption [kWh]')
    ax.set_title(title,  x=0.40, y=1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=6)  # Add labels and rotate for better readability
    ax.set_ylim(0, y_max)  # Set the same y-axis limit for all charts
    ax.legend()
    ax.grid(True, which='both', linestyle='dotted', linewidth=0.5)  # Add gridlines


# DISCOMFORT

def plot_deviation_histograms(
    time_step_s,
    dataframes,
    colti,
    legends,
    days=('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'),
    bins=np.arange(0.5, 7 + 1, 0.25),
    xticks=np.arange(0.5, 8, 1),
    histtype=('stepfilled',),
    colors=('tab:blue',),
    linestyles=('dotted',),
    figsize=(20,3)
):
    """
    Plot cumulative histograms of deviation signals for 1 or 2 dataframes.

    Parameters
    ----------
    time_step_s : sampling time in seconds
    dataframes : list
        [dfr1] or [dr1, dfr1]
    days : list/tuple of day names to plot.
    bins : array-like
    xticks : array-like
    histtype, colors, linestyles : list/tuple
        Styling for each dataframe
    figsize : tuple
    """

    # --- Unpack dataframes -----------------------------------------------------
    if len(dataframes) not in (1, 2):
        raise ValueError("dataframes must be [dfr1] or [dfr1, dfr2]")

    dfr1 = dataframes[0]
    dfr2 = dataframes[1] if len(dataframes) == 2 else None

    # --- Compute df deviation --------------------------------------------------
    mask_dfr1 = (dfr1['leave_fwb'] == 0) & (dfr1['f_occ'] > 0) & \
              (dfr1[colti[0]] <= dfr1["t_set_default"] - 0.5)
    dfr1["Dti_dev_1"] = (dfr1["t_set_default"] - dfr1[colti[0]]).where(mask_dfr1, np.nan)

    # --- Compute dfr deviation (if exists) ------------------------------------
    if dfr2 is not None:
        mask_dfr2 = (dfr2['leave_fwb'] == 0) & (dfr2['f_occ'] > 0) & \
                   (dfr2[colti[1]] <= dfr2["t_set_default"] - 0.5)
        dfr2["Dti_dev_2"] = (dfr2["t_set_default"] - dfr2[colti[1]]).where(mask_dfr2, np.nan)

        # dfr2 only allowed where dfr1 is NOT NaN
        valid_mask = ~dfr1["Dti_dev_1"].isna()
        dfr2["Dti_dev_2"] = dfr2["Dti_dev_2"].where(valid_mask, np.nan)

    # --- Plot setup ------------------------------------------------------------
    ncols = min(5, len(days))
    fig, axs = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    axs[0].set_ylabel('Number of hours')

    for j, day in enumerate(days[:ncols]):

        # DF deviation
        dfr1_day = dfr1.loc[dfr1["dayname"] == day, "Dti_dev_1"].dropna()

        # DFR2 deviation (if second dataframe)
        if dfr2 is not None:
            dfr2_day = dfr2.loc[dfr2["dayname"] == day, "Dti_dev_2"].dropna()

        ax = axs[j]

        # --- Plot first dataframe (dfr1) ----------------------------------------
        if len(dfr1_day) > 0:
            w_dfr1 = np.ones(len(dfr1_day)) * time_step_s / 3600
            ax.hist(
                dfr1_day,
                bins=bins,
                cumulative=-1,
                weights=w_dfr1,
                histtype=histtype[0],
                edgecolor=colors[0],
                facecolor=colors[0] if histtype[0] == "stepfilled" else None,
                linestyle=linestyles[0] if histtype[0] == "step" else None,
                linewidth=2 if histtype[0] == "step" else None,
                alpha=0.45 if histtype[0] == "stepfilled" else 1,
                align='mid',
                label='Deviation ' + legends[0],
            )

        # --- Plot second dataframe (dfr2) --------------------------------------
        if dfr2 is not None and len(dfr2_day) > 0:
            w_dfr2 = np.ones(len(dfr2_day)) * time_step_s / 3600
            ax.hist(
                dfr2_day,
                bins=bins,
                cumulative=-1,
                weights=w_dfr2,
                histtype=histtype[1],
                edgecolor=colors[1],
                facecolor=colors[1] if histtype[1] == "stepfilled" else None,
                linestyle=linestyles[1] if histtype[1] == "step" else None,
                linewidth=2 if histtype[1] == "step" else None,
                alpha=0.45 if histtype[1] == "stepfilled" else 1,
                align='mid',
                label='Deviation ' + legends[1],
            )

        ax.set_title(day)
        ax.set_xlim(0.25, None)
        ax.set_xticks(xticks)
        ax.grid(True, linestyle='dotted')
        ax.legend()

    fig.suptitle('Cumulated hours of temperature deviation from setpoint')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()



