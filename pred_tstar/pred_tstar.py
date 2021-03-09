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
import warnings
from scipy.interpolate import interp1d
from obspy.taup import TauPyModel
from collections import OrderedDict
from argparse import ArgumentParser
from obspy.taup.taup_create import build_taup_model
from obspy.taup.helper_classes import TauModelError

from pred_tstar.distance import get_dist, get_TSmP


def define_arguments():
    helptext = 'Predict tstar from TauP model, given a S-P traveltime distance'
    parser = ArgumentParser(description=helptext)

    helptext = "Input TauP file"
    parser.add_argument('fnam_nd', nargs='+', help=helptext)

    helptext = "S-P arrival time differences"
    parser.add_argument('--times', nargs='+', type=float, help=helptext)

    helptext = "Phases for travel-time difference (default: P, S)"
    parser.add_argument('--phase_list', nargs=2, type=str,
                        default=['P', 'S'], help=helptext)

    helptext = "Plot convergence in T-X plot"
    parser.add_argument('--plot', action="store_true", default=False,
                        help=helptext)
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


def calc_sens_full(model, fnam_tvel, distance, phase, depth=60.):
    try:
        arrivals = model.get_ray_paths(source_depth_in_km=depth,
                                       distance_in_degree=distance,
                                       phase_list=[phase, 'SKKKSSmP'])
    except TauModelError:
        arrivals = []

    if len(arrivals) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            if phase[0] == 'S':
                qs_inv = 1./qmu_ipl(depth[i])
                tstar[i] = times[i] * qs_inv
            elif phase[0] == 'P':
                L = 4./3. * (vs_ipl(depth[i]) / vp_ipl(depth[i]))**2
                qp_inv = L / qmu_ipl(depth[i]) + (1 - L) / 1e4
                tstar[i] = times[i] * qp_inv

        return np.sum(tstar), arrivals[0].time, \
               np.deg2rad(arrivals[0].ray_param)
    else:
        return None, None, None


def calc_tstar_tvel(model, fnam_tvel, distance, phase, depth):
    tstar, time, p = calc_sens_full(model, fnam_tvel=fnam_tvel,
                                    distance=distance, phase=phase,
                                    depth=depth)

    if time is None:
        time = -1.
        tstar = -1.
        p = -1.
    return tstar, time, p


def main(fnams_nd, times, phase_list,
         fnam_out='tstars.txt',
         fnam_out_p='rayparams.txt',
         fnam_out_pred='phase_predictions.txt',
         depth=40.,
         plot=False):
    with open(fnam_out, 'w') as f_tstar, \
         open(fnam_out_p, 'w') as f_p,   \
         open(fnam_out_pred, 'w') as f_pred:
        if type(fnams_nd) is list:
            fnams = fnams_nd
        else:
            fnams = [fnams_nd]

        for fnam_nd in fnams:
            fnam_npz = './taup_tmp/' \
                       + os.path.split(fnam_nd)[-1][:-3] + '.npz'
            build_taup_model(fnam_nd,
                             output_folder='./taup_tmp'
                             )
            cache = OrderedDict()
            model = TauPyModel(model=fnam_npz, cache=cache)

            f_tstar.write('%s ' % os.path.split(fnam_nd)[-1])
            for tSmP in times:
                dist = get_dist(model, tSmP=tSmP, phase_list=phase_list,
                                plot=plot, depth=depth)
                if dist is not None:
                    tstar_P, time_P, ray_param_P = calc_tstar_tvel(model=model,
                                                      fnam_tvel=fnam_nd,
                                                      distance=dist,
                                                      phase=phase_list[0],
                                                      depth=depth)
                    tstar_S, time_S, ray_param_S = calc_tstar_tvel(model=model,
                                                      fnam_tvel=fnam_nd,
                                                      distance=dist,
                                                      phase=phase_list[1],
                                                      depth=depth)
                else:
                    tstar_P = -1.
                    tstar_S = -1.
                    ray_param_P = -1.
                    ray_param_S = -1.
                if dist is None:
                    dist = 0.
                f_tstar.write('%5.2f %5.3f %5.3f ' %
                              (dist, tstar_P, tstar_S))
                f_p.write('%5.2f %5.3f %5.3f ' %
                          (dist, ray_param_P, ray_param_S))
                f_pred.write('%5.2f ' % dist)
                for phase in ['PP', 'SS', 'SSS', 'ScS', 'SP', 'SKS', 'PKP']:
                    tt_phase = get_TSmP(distance=dist, model=model,
                                        plot=False,
                                        tmeas=0., phase_list=['P', phase],
                                        depth=depth)
                    f_pred.write('%7.1f ' % tt_phase)
                f_pred.write('\n')
            f_tstar.write('\n')

if __name__== '__main__':
    args = define_arguments()
    main(fnams_nd=args.fnam_nd,
         times=args.times,
         phase_list=args.phase_list,
         plot=args.plot)
