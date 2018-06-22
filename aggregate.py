"""

Example:
    min, expected, max = aggregate_wofs(ndarray_of_wofs_bitfield, obs_dates)

"""

import numpy as np

def daterange(t0, tf, dt=np.timedelta64(1, 'D')):
    epochs = int((tf - t0) / dt) + 1 # inclusive count of distinct dates
    return t0 + dt * np.arange(epochs)

defaultdates = daterange(np.datetime64('1986-01-01', 'D'),
                         np.datetime64('2020-01-01', 'D'))

def prefilter_wofs(chunk):
    """Convert WOfS-bitfield (ndarray) to masked bool"""
    chunk &= ~np.uint8(4) # remove ocean masking

    wet = chunk == 128
    dry = chunk == 0
    clear = wet | dry

    return np.ma.array(wet, mask=~clear)


def foliate(chunk, chunkdates, newdates):
    """Temporal up-sampling (interleaves observations with masked nodata)"""
    t0 = newdates[0]
    epochs = len(newdates)
    dt = (newdates[-1] - newdates[0]) / (epochs - 1)

    indices = ((chunkdates - t0) /  dt).round().astype(int)

    # note, data array is pre-zeroed here.
    foliage = np.zeros((epochs,) + chunk.shape[1:], dtype=np.bool)
    foliage = np.ma.array(data=foliage, mask=True)

    foliage[indices] = chunk
    return foliage


def tails(maskedchunk):
    """Output booleans indicating bracketing"""
    clear = ~maskedchunk.mask

    past = clear.cumsum(axis=0, dtype=np.bool)
    future = clear[::-1,...].cumsum(axis=0, dtype=np.bool)[::-1,...]

    return past, future

def constant_interpolate(maskedchunk):
    """Infill by constant-value extrapolation in either temporal direction"""
    data = maskedchunk.data
    mask = maskedchunk.mask
    epochs = len(data)

    forward = data
    reverse = data.copy()

    for i in range(1, epochs):
        gaps = mask[i]
        forward[i][gaps] = forward[i-1][gaps]
    for i in range(epochs-1)[::-1]:
        gaps = mask[i]
        reverse[i][gaps] = reverse[i+1][gaps]

    return forward, reverse

def estimate(maskedchunk, conservative=True):
    """Uncertainty (for each pixel) according to bracketing obsevations.

    Where unbracketed, assert both states possible.

    If not conservative: exclude areas that are never observed,
    and disregard unprecedented possibilities.
    (That is, a pixel is not wettable unless it has been seen to wet,
    and not unwettable unless it has been seen to dry.)
    """
    past, future = tails(maskedchunk)
    forward, reverse = constant_interpolate(maskedchunk)

    bracketed = past & future

    upper = forward | reverse | ~bracketed
    lower = forward & reverse # & bracketed (implied since nodata pre-zeroed)

    observed = np.any(~maskedchunk.mask, axis=0) # check extrapolatable

    if not conservative:
        always_wet = maskedchunk.all(axis=0).data & observed
        always_dry = (~maskedchunk).all(axis=0).data & observed

        upper &= observed & ~always_dry # exclude historicaly unwettable
        lower |= always_wet # assume permanent waterbodies

    estimate = 0.5 * upper + 0.5 * lower # beware: bool + bool -> bool
    # Does this implementation actually perform multiplication operation?

    # extrapolate tails (constant)
    tail = ~future & observed
    estimate[tail] = forward[tail]
    nose = ~past & observed
    estimate[nose] = reverse[nose]

    return lower, estimate, upper

def aggregate(*args):
    """Coarsen spatially"""
    return tuple(x.sum(axis=(1,2)) for x in args)

def aggregate_wofs(obsarray, obsdates, newdates=defaultdates, conservative=True):
    """Take input bitfield, orchestrate aggregation"""

    return aggregate(*estimate(foliate(prefilter_wofs(obsarray),
                                       obsdates, newdates),
                               conservative))
