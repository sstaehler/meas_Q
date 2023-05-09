# pred_tstar

## Howto

```
usage: pred_tstar.py [-h] [--times TIMES [TIMES ...]] 
                     [--phase_list_dist PHASE_LIST_DIST PHASE_LIST_DIST]
                     [--phase_list_pred PHASE_LIST_PRED [PHASE_LIST_PRED ...]] 
                     [--plot]
                     [--absolute_times]
                     fnam_nd [fnam_nd ...]

Predict tstar from TauP model, given a S-P traveltime distance

positional arguments:
  fnam_nd               Input TauP file

optional arguments:
  -h, --help            show this help message and exit
  --times TIMES [TIMES ...]
                        S-P arrival time differences
  --phase_list_dist PHASE_LIST_DIST PHASE_LIST_DIST
                        Phases for travel-time difference (default: P, S)
  --phase_list_pred PHASE_LIST_PRED [PHASE_LIST_PRED ...]
                        Phases to compute times and tstar for (default: PP, PPP, SS, SSS, ScS, SKS)
  --plot                Plot convergence in T-X plot
  --absolute_times      Return absolute travel times (default: False, ie return time relative to first phase in phase_list_dist)
```

## versions

### v0.4:

Reverse mode

### v0.3:

Proper support for other reference phases than P, S

### v0.2:

Reformatting and fixes