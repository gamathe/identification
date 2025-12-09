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


def dTiz_t_3R2C_Q(Uz, tauz, Tz, i2, T0, ti, dti_t, read_dti) :
    U0, U1, U2   = Uz
    tau1, tau2 = tauz
    T1, T2       = Tz

    if ~read_dti :
        dT1_t = - 1/tau1 * (T1 - T0) + 1/tau1/U1 * U2 * (T2 - T1)
        dT2_t = - 1/tau2 * (T2 - T1) + 1/tau2/U2 * U0 * (T0 - T2) + i2/tau2/U2
        Q_dot_t = - U1 * (T1 - T0) - U0 * (T2 - T0)

    else :
        dT1_t = - 1/tau1 * (T1 - T0) + 1/tau1/U1 * U2 * (ti - T1) 
        dT2_t =   dti_t
        Q_dot_t = - U1 * (T1 - T0) + U2 * (ti - T1) + tau2 * U2 * dti_t - i2

    Q_dot_r = tau2 * U2 * dti_t + U2 * (ti - T1) - U0 * (T0 - ti) - i2
    Q_dot_m = tau1 * U1 * dT1_t + tau2 * U2 * dT2_t
    
    return dT1_t, dT2_t, Q_dot_r, Q_dot_t, Q_dot_m


def ffit_identification(x, df, df_selected, time_step_s, ts, t_e_air, t_i, dti_t, read_dti, \
                        Iz, Q_dot_g, Q_dot_g_m1, flag_error):

    U0, U1, U2,  tau1, tau2, Asol, fQ = np.square(x)

    Uz = np.array([U0, U1, U2]); tauz = np.array([tau1, tau2])

    fQ = tanh(2 * fQ)  # comprised between 0 and 1

    Qc = Q_dot_g * fQ + Q_dot_g_m1 * (1-fQ)

    HTC = U0 + 1/(1/U1+1/U2) 

    dfrs = []
    RMS_list  = []

    ndays_init = 8

    for idx, row in df_selected.iterrows():

        i1 = row["first"]                                        # start of selected period
        i2 = row["last"]  - pd.Timedelta(minutes=15)             # end of selected period

        ind_start = df.index.get_loc(i1) 
        ind_stop  = df.index.get_loc(i2)

        ind_init = max(0, ind_start - 10 * 24 * 4)
        
        # ---- Step 0: Initialize the temperatures ----
        
        T1_0 = np.nanmean(t_i[ind_init:ind_start])
        T2_0 = t_i[ind_start]
    
        Ti0 = [T1_0, T2_0]

        # ---- Step 1: Solve the temperatures ----

        def model_dTi_t(t, Ti):
            T1, T2 = Ti
            
            Tz = np.array([T1, T2])
            
            ind = int((t - ts[0]) / time_step_s)
            if ind > ind_stop : ind = ind_stop
    
            T0 = t_e_air[ind]
            ti_val = t_i[ind]
            dti_val = dti_t[ind]

            initialize = (ind - ind_start < 24 * 4 * ndays_init)
            read = initialize | read_dti[ind]

            Q_dot_i = 0
            Q_dot_s = Asol * Iz[ind]
            Q_dot_h = Qc[ind]
    
            i2 = Q_dot_h + Q_dot_s + Q_dot_i

            dT1_t, dT2_t, Q_dot_r, Q_dot_t, Q_dot_m = dTiz_t_3R2C_Q(Uz, tauz, Tz, i2, T0, ti_val, dti_val, read)

            return [dT1_t, dT2_t], Q_dot_i, Q_dot_s, Q_dot_h, Q_dot_r, Q_dot_t,  Q_dot_m
    
        def dTi_t(t, Ti):
            ret = model_dTi_t(t, Ti) # only dTi_T is returned for ordinary differential integral
            return ret[0]

        T_array = solve_ivp(
            dTi_t,
            (ts[ind_start], ts[ind_stop + 1]),
            Ti0,
            t_eval=ts[ind_start:ind_stop + 1],
            max_step = 3600,
            method="LSODA")
 
        Tsol = T_array.y.T

        # Use the corresponding timestamps from the index
        start_time = df.index[ind_start] 
        end_time = df.index[ind_stop] 
        
        # Generate the date range between start_time and end_time
        d = pd.date_range(start=start_time, end=end_time, freq='15min')

        dfr = pd.DataFrame(Tsol, columns=['T1', 'T2'], index=d)

        # ---- Step 2: Compute fluxes in post-processing ----

        lst_flows = ["Q_dot_i", "Q_dot_s", "Q_dot_h", "Q_dot_r", "Q_dot_t", "Q_dot_m"]
        
        dfr[lst_flows] = np.asarray([model_dTi_t(ts[ind_start + tt], Tsol[tt])[1:len(lst_flows) + 1] \
                                     for tt in range(Tsol.shape[0])])

        # ---- Step 3: Assemble DataFrame ----
  
        dfr = pd.concat([dfr, df.iloc[ind_start : ind_stop + 1]], axis=1)
        dfr = dfr.iloc[ndays_init*24*4:].copy()

        dfrs.append(dfr)

        # ---- Step 4: RMS ----
        
        dfr["err"] = (dfr["T2"] - dfr["t_i"]).where(~pd.isna(dfr["t_i_not_interpolated"]), 0) * (1 + flag_error)

        chi_2 = sum(dfr["err"] * dfr["err"]) / (len(dfr['t_i']) - dfr['t_i_not_interpolated'].isna().sum())

        RMS = sqrt(chi_2)

        RMS_list.append(RMS)

    dfr = pd.concat(dfrs)
    
    chi_2 = sum(dfr["err"] * dfr["err"]) / (len(dfr['t_i']) - dfr['t_i_not_interpolated'].isna().sum())

    RMS = sqrt(chi_2)
    
    try: 
        print(round(RMS,2),  \
        "HTCs :", int(HTC),  \
        " var %: ", *((np.trunc(100*np.array(np.square(x))/np.array(np.square(ffit_identification.x1)))).astype(int)))
    except:
        print("...")

    ffit_identification.xold = x
    
    return dfr, dfrs, RMS_list, RMS, Tsol



def ffit_simulation(U0, U1, U2, tau1, tau2, Asol, df, time_step_s, ts, \
                    month, t_e_air, t_i, dti_t, t_set_smoothed, dead_band, Q_dot_i_occ, Iz, \
                    Q_dot_n_z, Xhte, Xh):
    
    ind_start = 0
    ind_stop = len(df) - 1
    
    df['Xte'] = [Xhte(tt, 10, 0.5) for tt in t_e_air]
    Xte  = df['Xte'].values
    
    def model_dTi_t(t, Ti):
        T1, T2 = Ti
        
        Tz = np.array([T1, T2])
        
        ind = int((t - ts[0]) / time_step_s)
        if ind > ind_stop : ind = ind_stop
    
        T0 = t_e_air[ind] 

        try:
            ti_val = float(t_i[ind])
        except:
            ti_val = np.nan
        
        try:
            dti_val = float(dt_i_t[ind])
        except:
            dti_val = np.nan
    
        initialize =  (ind - ind_start < 24 * 4 * ndays_init)
        read = initialize & ~np.isnan(ti_val) & ~np.isnan(dti_val)
    
        Q_dot_i = Q_dot_i_occ[ind] 
        Q_dot_s = Asol * Iz[ind]
    
        if (month[ind] == 7) | (month[ind] == 8)   :
            Q_dot_h = 0
        else :
            Q_dot_h  = Xh(T2, t_set_smoothed[ind], dead_band[ind]) * Xte[ind] * Q_dot_n_z
    
        i2 = Q_dot_h + Q_dot_s + Q_dot_i
    
        dT1_t, dT2_t, Q_dot_r, Q_dot_t, Q_dot_m = dTiz_t_3R2C_Q(Uz, tauz, Tz, i2, T0, ti_val, dti_val, read)
            
        return [dT1_t, dT2_t], Q_dot_i, Q_dot_s, Q_dot_h, Q_dot_r, Q_dot_t, Q_dot_m
    
    def dTi_t(t, Ti):
        ret = model_dTi_t(t, Ti) # only dTi_T is returned for ordinary differential integral
        return ret[0]
    
    Uz = np.array([U0, U1, U2]); tauz = np.array([tau1, tau2])
    
    ind_init = max(0, ind_start - 10 * 24 * 4)
    
    ndays_init = 4
    
    # ---- Step 0: Initialize the temperatures ----
    T1_0 = t_i[ind_start]
    T2_0 = t_i[ind_start]
    
    Ti0 = [T1_0, T2_0]
    
    # ---- Step 1: Solve the temperatures ----
    T_array = solve_ivp(
        dTi_t,
        (ts[ind_start], ts[ind_stop]),
        Ti0,
        t_eval=ts[ind_start:ind_stop],
        max_step = 3600,
        method="LSODA"
    )
    Tsol = T_array.y.T
    
    # Use the corresponding timestamps from the index
    start_time = df.index[ind_start] 
    end_time   = df.index[ind_stop]
    
    # Generate the date range between start_time and end_time
    d = pd.date_range(start=start_time, end=end_time - pd.Timedelta(minutes=15), freq='15min')
    
    dfr = pd.DataFrame(Tsol, columns=['T1', 'T2'], index=d)
    
    # ---- Step 2: Compute fluxes in post-processing ----
    lst_flows = ["Q_dot_i", "Q_dot_s", "Q_dot_h", "Q_dot_r", "Q_dot_t", "Q_dot_m"]
    
    dfr[lst_flows] = np.asarray([model_dTi_t(ts[ind_start + tt], Tsol[tt])[1:len(lst_flows) + 1] \
                                 for tt in range(Tsol.shape[0])])
    
    # ---- Step 3: Assemble DataFrame ----
    dfr = pd.concat([dfr, df.iloc[ind_start:ind_stop]], axis=1)
    dfr = dfr.iloc[ndays_init*24*4:].copy()
    
    # ---- Step 4: RMS ----
    dfr["err"] = (dfr["T2"] - dfr["t_i"]).where(~pd.isna(dfr["t_i"]), 0)
    chi2 = sum(dfr["err"] * dfr["err"]) / (len(dfr["t_i"]) - dfr["t_i"].isna().sum())
    RMS = sqrt(chi2)

    return dfr, RMS



def ffit_optimization(x, U0, U1, U2, tau1, tau2, Asol, df, time_step_s, ts, \
                    month, t_e_air, t_i, dti_t, t_set_default, dead_band, Q_dot_i_occ, Iz, \
                    dow_next_occ, nhours_before_occ, t_set_occ, Dtset_occ, Dte_base, \
                    Q_dot_n_z, Xhte, Xh, Xrh, aod, bod, f_mult_conso):

    amo, bmo = np.square(x)

    a_param = np.where(dow_next_occ == 0 , amo, aod)
    
    nhours_reheat_te = np.where(dow_next_occ == 0, bmo, bod) * Dte_base
    nhours_remaining_fot_ti = np.clip(nhours_before_occ - nhours_reheat_te, 0, None)
    
    HTC = U0 + 1/(1/U1+1/U2) 

    ind_start = 0
    ind_stop = len(df) - 1
    
    Xte =  [Xhte(tt, 10, 0.5) for tt in t_e_air]

    def model_dTi_t(t, Ti):
        T1, T2 = Ti
        
        Tz = np.array([T1, T2])
        
        ind = int((t - ts[0]) / time_step_s)
        if ind > ind_stop : ind = ind_stop
    
        T0 = t_e_air[ind]
        
        try:
            ti_val = float(t_i[ind])
        except:
            ti_val = np.nan
        
        try:
            dti_val = float(dt_i_t[ind])
        except:
            dti_val = np.nan
    
        initialize =  (ind - ind_start < 24 * 4 * ndays_init)
        read = initialize & ~np.isnan(ti_val) & ~np.isnan(dti_val)
    
        Q_dot_i = Q_dot_i_occ[ind]
        Q_dot_s = Asol * Iz[ind]
    
        if (month[ind] == 7) | (month[ind] == 8)   :
            Q_dot_h = 0
        else :
            nhours_reheat = nhours_reheat_te[ind] + a_param[ind] * (t_set_occ[ind] - T2)
            Xrheat  = Xrh(nhours_remaining_fot_ti[ind], a_param[ind], T2, t_set_occ[ind] - 0.5)
            # fprop   = ((nhours_reheat - nhours_before_occ[ind]) / 4) ** 4 
            tset    = t_set_default[ind] + Dtset_occ[ind] * Xrheat #* fprop
            Q_dot_h = Xh(T2, tset, dead_band[ind]) * Xte[ind] * Q_dot_n_z 
    
        i2 = Q_dot_h + Q_dot_s + Q_dot_i
    
        dT1_t, dT2_t, Q_dot_r, Q_dot_t, Q_dot_m = dTiz_t_3R2C_Q(Uz, tauz, Tz, i2, T0, ti_val, dti_val, read)
            
        return [dT1_t, dT2_t], Q_dot_i, Q_dot_s, Q_dot_h, Q_dot_r, Q_dot_t, Q_dot_m
    
    def dTi_t(t, Ti):
        ret = model_dTi_t(t, Ti) # only dTi_T is returned for ordinary differential integral
        return ret[0]
    
    Uz = np.array([U0, U1, U2]); tauz = np.array([tau1, tau2])
    
    ind_init = max(0, ind_start - 10 * 24 * 4)
    
    ndays_init = 4
    
    # ---- Step 0: Initialize the temperatures ----
    T1_0 = t_i[ind_start]
    T2_0 = t_i[ind_start]
    
    Ti0 = [T1_0, T2_0]
    
    # ---- Step 1: Solve the temperatures ----
    
    T_array = solve_ivp(
        dTi_t,
        (ts[ind_start], ts[ind_stop]),
        Ti0,
        t_eval=ts[ind_start:ind_stop],
        max_step = 3600,
        method="LSODA"
    )
    Tsol = T_array.y.T
    
    # Use the corresponding timestamps from the index
    start_time = df.index[ind_start] 
    end_time   = df.index[ind_stop]
    
    # Generate the date range between start_time and end_time
    d = pd.date_range(start=start_time, end=end_time - pd.Timedelta(minutes=15), freq='15min')
    
    dfr = pd.DataFrame(Tsol, columns=['T1', 'T2'], index=d)
    
    # ---- Step 2: Compute fluxes in post-processing ----
    
    lst_flows = ["Q_dot_i", "Q_dot_s", "Q_dot_h", "Q_dot_r", "Q_dot_t", "Q_dot_m"]
    
    dfr[lst_flows] = np.asarray([model_dTi_t(ts[ind_start + tt], Tsol[tt])[1:len(lst_flows) + 1] \
                                 for tt in range(Tsol.shape[0])])
    
    # ---- Step 3: Assemble DataFrame ----
    
    dfr = pd.concat([dfr, df.iloc[ind_start:ind_stop]], axis=1)
    dfr = dfr.iloc[ndays_init*24*4:].copy()
    
    # ---- Step 4: RMS ----

    dfr['ti_deviation'] = \
    (dfr['t_set_default'] - dfr['T2']).where((dfr['T2'] < dfr['t_set_default'] - 0.5) & (dfr['f_occ'] == 1) , np.nan)
    chi2_ti = \
    (dfr["ti_deviation"] * dfr["ti_deviation"]).sum() / (len(dfr['ti_deviation']) - dfr['ti_deviation'].isna().sum()) 

    dfr['dt_consumption'] = f_mult_conso * dfr["Q_dot_h"] / HTC / 30 
    chi2_cons = \
    (dfr["dt_consumption"] * dfr["dt_consumption"]).sum() / (len(dfr['dt_consumption']) - dfr['dt_consumption'].isna().sum()) 
       
    RMS = sqrt(chi2_ti + chi2_cons)

    print("RMS = ", round(RMS,2), \
          ", amo :", round(amo, 2), ", bmo :", round(bmo, 2), \
          ", RMS_ti :", round(sqrt(chi2_ti), 2), ", RMS_conso :", round(sqrt(chi2_cons), 2),  \
          " var %: ", *((np.trunc(100*np.array(np.square(x))/np.array(np.square(ffit_optimization.x1)))).astype(int)))
    
    return dfr, RMS



def check_balance(dfr) :
    
    dfr["Qh_kWh"] = dfr["Q_dot_h"].cumsum() / 1000 / 4
    dfr["Qi_kWh"] = dfr["Q_dot_i"].cumsum() / 1000 / 4
    dfr["Qs_kWh"] = dfr["Q_dot_s"].cumsum() / 1000 / 4
    dfr["Qt_kWh"] = dfr["Q_dot_t"].cumsum() / 1000 / 4
    dfr["Qm_kWh"] = dfr["Q_dot_m"].cumsum() / 1000 / 4
    dfr["Qr_kWh"] = dfr["Q_dot_r"].cumsum() / 1000 / 4
    
    balance = pd.DataFrame({
        "occupancy": pd.Series(dtype='int'),
        "solar": pd.Series(dtype='int'),
        "heating": pd.Series(dtype='int'),
        "transmission": pd.Series(dtype='int'),
        "thermal mass": pd.Series(dtype='int'),
        "unbalance": pd.Series(dtype='int'),
        "residuals": pd.Series(dtype='int')
    })
    
    
    dfrb = dfr.copy()
    QI_kWh = dfrb["Qi_kWh"].values[-1] - dfrb["Qi_kWh"].values[0]
    QS_kWh = dfrb["Qs_kWh"].values[-1] - dfrb["Qs_kWh"].values[0]
    QH_kWh = dfrb["Qh_kWh"].values[-1] - dfrb["Qh_kWh"].values[0]
    QT_kWh = dfrb["Qt_kWh"].values[-1] - dfrb["Qt_kWh"].values[0]
    QM_kWh = dfrb["Qm_kWh"].values[-1] - dfrb["Qm_kWh"].values[0]
    QR_kWh = dfrb["Qr_kWh"].values[-1] - dfrb["Qr_kWh"].values[0]
    
    QU_kWh = QI_kWh + QS_kWh + QH_kWh + QT_kWh - QM_kWh 
    
    balance = pd.concat([balance, \
               pd.DataFrame({"occupancy": QI_kWh, "solar": QS_kWh, "heating": QH_kWh, \
               "transmission": QT_kWh,  "thermal mass": QM_kWh,  \
                "unbalance": QU_kWh, "residuals": QR_kWh}, index=[0])])

    return balance


def smooth_long_plateaus(arr, smooth_len = 4):
    # Progressive increase of t_set during the restart
    arr = np.array(arr, dtype=float)
    n = len(arr)
    result = arr.copy()
    
    i = 0
    while i < n-1:
        # Check for rising edge
        if arr[i+1] > arr[i]:
            # Count the length of the plateau of the higher value
            plateau_val = arr[i+1]
            plateau_start = i+1
            plateau_len = 1
            while plateau_start + plateau_len < n and arr[plateau_start + plateau_len] == plateau_val:
                plateau_len += 1
            
            # If plateau is longer than smooth_len, smooth the first `smooth_len` steps
            if plateau_len > smooth_len:
                start_val = arr[i]
                end_val = plateau_val
                # Interpolate over the next `smooth_len` steps
                interp_steps = min(smooth_len, n - i - 1)
                result[i+1:i+1+interp_steps] = np.linspace(start_val, end_val, interp_steps+1)[1:]
            
            # Move i past the plateau
            i = plateau_start + plateau_len - 1
        else:
            i += 1
            
    return result