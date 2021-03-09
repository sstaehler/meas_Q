#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''
import numpy as np
from matplotlib import pyplot as plt
from obspy.taup.helper_classes import TauModelError


def get_dist(model, tSmP, phase_list, depth, plot=False):
    from scipy.optimize import newton

    # Reasonable guess 1
    dist0 = tSmP / 6.5
    try:
        dist = newton(func=get_TSmP, fprime=get_SSmP,
                      x0=dist0, args=(model, tSmP, phase_list, plot, depth),
                      maxiter=10)
    except RuntimeError:
        dist = None
    if dist == None:
        # Reasonable guess 2
        dist0 = tSmP / 8.
        try:
            dist = newton(func=get_TSmP, fprime=get_SSmP,
                          x0=dist0,
                          args=(model, tSmP, phase_list, plot, depth),
                          maxiter=10)
        except RuntimeError:
            dist = None

    if plot:
        dists = np.arange(start=10, stop=40, step=0.3)
        plt.axhline(tSmP)
        #plot_TTcurve(model, dists)
        #plt.axhline(tSmP)
        #plt.ylim(0, 300)
        plt.show()
    return dist


def plot_TTcurve(model, dists, depth=40):
    times_P = []
    times_S = []
    ps_P = []
    ps_S = []
    dists_P = []
    dists_S = []
    tdiffs = []
    ddiffs = []
    for dist in dists:
        times_P_this = []
        arrs = model.get_travel_times(source_depth_in_km=depth,
                                      distance_in_degree=dist,
                                      phase_list='P')
        for arr in arrs:
            times_P_this.append(arr.time)
            times_P.append(arr.time)
            ps_P.append(np.deg2rad(arr.ray_param))
            dists_P.append(dist)

        arrs = model.get_travel_times(source_depth_in_km=depth,
                                      distance_in_degree=dist,
                                      phase_list='S')
        for arr in arrs:
            #for time in times_P_this:
            tdiffs.append(arr.time - times_P_this[0])
            ddiffs.append(dist)
            times_S.append(arr.time)
            ps_S.append(np.deg2rad(arr.ray_param))
            dists_S.append(dist)

    times_P = np.asarray(times_P)
    times_S = np.asarray(times_S)
    dists_P = np.asarray(dists_P)
    dists_S = np.asarray(dists_S)
    ps_P = np.asarray(ps_P)
    ps_S = np.asarray(ps_S)

    # sort = np.argsort(ps_P)
    # plt.plot(times_P[sort], dists_P[sort], 'o', ls='dashed')
    # sort = np.argsort(ps_S)
    # plt.plot(times_S[sort], dists_S[sort], 'o', ls='dashed')
    fig, ax = plt.subplots(nrows=3, ncols=1,
                           sharex=True,
                           figsize=(5, 10))
    ax[2].plot(dists_P, times_P, 'x', label='P')
    ax[2].plot(dists_S, times_S, 'o', label='S')
    ax[1].plot(dists_P, ps_P, 'x', label='P')
    ax[1].plot(dists_S, ps_S, 'o', label='S')
    ax[0].plot(ddiffs, tdiffs, 'x')

    return fig, ax


def get_TSmP(distance, model, tmeas, phase_list, plot, depth, return_abs=False):
    if len(phase_list) != 2:
        raise ValueError('Only two phases allowed')
    tP = None
    tS = None
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=distance,
                                          phase_list=phase_list)
    except (ValueError, TauModelError):
        pass
    else:
        for arr in arrivals:
            if arr.name == phase_list[0] and tP is None:
                tP = arr.time
            elif arr.name == phase_list[1] and tS is None:
                tS = arr.time
    if tP is None or tS is None:
        if plot:
            plt.plot(distance, -1000, 'o')
        return -1000.
    else:
        if plot:
            plt.plot(distance, tS - tP, 'o')
        if return_abs:
            return (tS) - tmeas
        else:
            return (tS - tP) - tmeas


def get_SSmP(distance, model, tmeas, phase_list, plot, depth):
    if len(phase_list) != 2:
        raise ValueError('Only two phases allowed')
    sP = None
    sS = None
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth,
                                          distance_in_degree=distance,
                                          phase_list=phase_list)
    except (ValueError, TauModelError):
        pass
    else:
        for arr in arrivals:
            if arr.name == phase_list[0] and sP is None:
                sP = np.deg2rad(arr.ray_param)
            elif arr.name == phase_list[1] and sS is None:
                sS = np.deg2rad(arr.ray_param)

    if sP is None or sS is None:
        return -10000.
    else:
        return sS - sP