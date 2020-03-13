#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
'''

import numpy as np
import os
from scipy.interpolate import interp1d
from obspy.taup import TauPyModel
from collections import OrderedDict
from argparse import ArgumentParser
from obspy.taup.taup_create import build_taup_model

def define_arguments():
    helptext = 'Predict tstar from TauP model, given a S-P traveltime distance'
    parser = ArgumentParser(description=helptext)

    helptext = "Input TauP file"
    parser.add_argument('fnam_nd', help=helptext)

    helptext = "S-P arrival time differences"
    parser.add_argument('times_SmP', nargs='+', type=float, help=helptext)

    return parser.parse_args()


def plot_path(model, ax, distance, phase_list,
              layer_bounds=None):
    arrivals = model.get_ray_paths(source_depth_in_km=0,
                                   distance_in_degree=distance,
                                   phase_list=phase_list)

    arrivals.plot_rays(ax=ax, show=False, plot_all=False, legend=True)
    if layer_bounds is not None:
        for layer in layer_bounds:
            theta = np.linspace(0, 2*np.pi, 200)
            r = np.ones_like(theta) * (model.model.radius_of_planet - layer)
            ax.plot(theta, r)
    ax.set_rmax(model.model.radius_of_planet)


def calc_sens_full(model, fnam_tvel, distance, phase, depth=50.):
    arrivals = model.get_ray_paths(source_depth_in_km=depth,
                                   distance_in_degree=distance,
                                   phase_list=[phase, 'SKKKSSmP'])
    if len(arrivals) > 0:
        vel_model = np.genfromtxt(fnam_tvel, invalid_raise=False).T
        depth_model = vel_model[0]
        qmu_model = vel_model[4]
        vp_model = vel_model[1]
        vs_model = vel_model[2]
        qmu_ipl = interp1d(x=depth_model, y=qmu_model)
        vp_ipl = interp1d(x=depth_model, y=vp_model)
        vs_ipl = interp1d(x=depth_model, y=vs_model)

        path = arrivals[0].path
        dist = np.zeros(len(path))
        depth = np.zeros(len(path))
        times = np.zeros_like(depth)
        tstar = np.zeros_like(depth)
        for i in range(1, len(path)):
            dist[i] = path[i][2]
            depth[i] = path[i][3]
            times[i] = path[i][1] - path[i - 1][1]
            if depth[i] < 50.:
                qscatinv = 1./300.
            else:
                qscatinv = 0
            if phase[-1] == 'S':
                qs_inv = 1./qmu_ipl(depth[i]) + qscatinv
                tstar[i] = times[i] * qs_inv
            elif phase[-1] == 'P':
                L = 4./3. * (vs_ipl(depth[i]) / vp_ipl(depth[i]))**2
                qp_inv = L / qmu_ipl(depth[i]) + (1 - L) / 1e4 + qscatinv
                tstar[i] = times[i] * qp_inv

        return np.sum(tstar), arrivals[0].time
    else:
        return None, None


def calc_tstar_tvel(model, fnam_tvel, distance, phase):
    tstar, time = calc_sens_full(model, fnam_tvel=fnam_tvel,
                                 distance=distance, phase=phase)

    if time is None:
        time = -1.
        tstar = -1.
    return tstar, time


def get_dist(model, tSmP):
    from scipy.optimize import newton
    dist0 = tSmP / 6.5
    try:
        dist = newton(func=get_TSmP, fprime=get_SSmP,
                      x0=dist0, args=(model, tSmP), maxiter=10)
    except RuntimeError:
        dist = None
    return dist


def get_TSmP(distance, model, tmeas, depth=50.):
    arrivals = model.get_travel_times(source_depth_in_km=depth,
                                      distance_in_degree=distance,
                                      phase_list=['P', 'S'])
    tP = None
    tS = None
    for arr in arrivals:
        if arr.name == 'P' and tP is None:
            tP = arr.time
        elif arr.name == 'S' and tS is None:
            tS = arr.time

    if tP is None or tS is None:
        return -1000.
    else:
        return (tS - tP) - tmeas


def get_SSmP(distance, model, tmeas, depth=50.):
    arrivals = model.get_travel_times(source_depth_in_km=depth,
                                      distance_in_degree=distance,
                                      phase_list=['P', 'S'])
    sP = None
    sS = None
    for arr in arrivals:
        if arr.name == 'P' and sP is None:
            sP = np.deg2rad(arr.ray_param)
        elif arr.name == 'S' and sS is None:
            sS = np.deg2rad(arr.ray_param)

    if sP is None or sS is None:
        return -10000.
    else:
        return sS - sP


def main(fnam_nd, times_SmP, fnam_out='tstars.txt'):
    with open(fnam_out, 'w') as f:
        fnam_npz = './taup_tmp/' \
            + os.path.split(fnam_nd)[-1][:-3] + '.npz'
        build_taup_model(fnam_nd,
                         output_folder='./taup_tmp')
        cache = OrderedDict()
        model = TauPyModel(model=fnam_npz, cache=cache)

        f.write('%s ' % os.path.split(fnam_nd)[-1])
        for tSmP in times_SmP:
            dist = get_dist(model, tSmP=tSmP)
            if dist is not None:
                tstar_P, time_P = calc_tstar_tvel(model=model,
                                                  fnam_tvel=fnam_nd,
                                                  distance=dist, phase='P')
                tstar_S, time_S = calc_tstar_tvel(model=model,
                                                  fnam_tvel=fnam_nd,
                                                  distance=dist, phase='S')
            else:
                tstar_P = -1.
                tstar_S = -1.
            if dist is None:
                dist = 0.
            f.write('%5.2f %5.3f %5.3f  ' %
                    (dist,
                     tstar_P, tstar_S))

if __name__== '__main__':
    args = define_arguments()
    main(fnam_nd=args.fnam_nd,
         times_SmP=args.times_SmP)
