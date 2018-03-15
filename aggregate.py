

"""

 - User may supply intended output temporal basis

 - Calculation may be performed in chunks, because sparse big data cannot
   be entirely unpacked in memory.

 - Chunking regime: full temporal depth gives simplicity of algorithm,
   although one-temporal-layer-at-a-time would likely give best performance.

 - Trivial to add spatially-parallel chunks, assuming identical temporal basis.

 - Nontrivial to concatenate temporally-consecutive chunks, likely require
   knowledge of last value and (per-pixel) time of last observation, and also
   may require full-temporal-depth statistics if selecting more sophisticated
   models.

 - Some algorithms may permit up-sampling temporal axis at a later stage, but
   others may not after aggregation.

Question: masked boolean array versus (-1, 0, 1) array?
Presumably masked (implemented as 8bit) is more memory intensive than 3-value.
Unclear whether 2x1bit is feasible, and practical performance implications.
Assume not critical for now.

"""
import numpy as np
import matplotlib.pyplot as plt
import collections
import xarray

class Aggregator:
    def __init__(self, timeaxis=None): # TODO: permit a mask argument (e.g. a geometry)
        if timeaxis is None:
            self.time = None
        else:
            n = len(timeaxis)
            if n > 3:
                self.time = np.asarray(timeaxis).astype(np.datetime64)
            else:
                t0 = np.datetime64(timeaxis[0])
                tf = np.datetime64(timeaxis[1])
                if n == 3:
                    step = timeaxis[2]
                else:
                    step = np.timedelta64(1, 'D')
                self.time = np.arange(t0, tf, step)
    def __call__(self, *args):
        """ Aggregate an xarray.DataArray or timeseries,array """
        if len(args) == 1:
            target, = args
            assert isinstance(args[0], xarray.DataArray)
            assert len(target.dims) == 3
            assert 'time' in target.dims
            if target.dims[0] != 'time':
                target = target.transpose('time',
                                          *(set(target.dims) - set('time')))
            time, data = target.time.data, target.data
        else:
            time, data = target
        assert len(time) == len(data)
        time = np.asarray(time).astype(np.datetime64)
        data = np.asarray(data).reshape((len(time), -1))
        time, data = self.prefilter(time, data)
        if self.time is None:
            self.time = time
        self.history = time, data
        summary = self.compute(time, data)
        return Aggregation(self.time, summary, data.shape[1], self)
    def prefilter(self, time, data):
        return time, data
    def compute(self, time, data):
        """ Aggregate numpy n-vector and n x m inputs """
        raise NotImplementedError
    def foliate(self, time, data, fill=np.nan):
        shape = list(data.shape)
        shape[0] = len(self.time)
        new = np.full(shape, fill, data.dtype)
        raise NotImplementedError

class Aggregation:
    def __init__(self, timeaxis, data, pixels, aggregator):
        self.time = timeaxis
        self.method = aggregator
        self.pixels = pixels
        self.data = data
        self.lower, self.estimate, self.upper = self.data
    def __add__(self, other):
        if not isinstance(other, Aggregation) or self.method is not other.method:
            raise TypeError
        return Aggregation(self.data + other.data,
                           self.pixels + other.pixels,
                           self.method)
    def json(self):
        raise NotImplementedError
    def envelopeplot(self, axes=None):
        ax = axes or plt.gca()
        envelope = ax.fill_between(self.time, self.upper, self.lower, alpha=0.3)
        line = ax.plot(self.time, self.estimate)
        return [envelope, line]
    def discreteplot(self, axes=None, marker='.'):
        ax = axes or plt.gca()
        n = len(self.time)
        tt = np.vstack([self.time]*2)
        ww = np.vstack([self.lower, np.zeros(n)])
        dd = np.vstack([self.upper, np.ones(n)*self.pixels])
        ax.plot(tt, ww, c='b', alpha=0.3)
        ax.plot(tt, dd, c='r', alpha=0.3)
        #ax.plot(tt, np.vstack([self.upper, self.lower]), c='k')
        return ax.plot(self.time, self.estimate, marker=marker, linewidth=0, alpha=0.3, c='k')
    def lineplot(self, axes=None):
         ax = axes or plt.gca()
         return ax.plot(self.time, self.estimate, marker='.', alpha=0.7)

def parsewofs(data):
    data = data & ~np.uint8(4) # disable ocean mask
    wet = data == 128
    dry = data == 0
    clear = wet | dry
    return wet, clear

class WofsMask(Aggregator):
    """Convert WOfS bitfield to masked boolean array"""
    def prefilter(self, time, data):
        wet, clear = parsewofs(data)
        masked = np.ma.array(wet, mask=~clear)
        return time, masked

class WofsExist(WofsMask):
    """Exclude dates with no valid pixels"""
    def prefilter(self, time, data):
        time, data = super().prefilter(time, data)
        inclusion = (~data.mask).sum(axis=1).astype(np.bool)
        return time[inclusion], data[inclusion]

class Naive(WofsExist):
    """Summation provides only a loose lower bound."""
    def compute(self, time, data):
        estimate = data.sum(axis=1).filled(0)
        return np.vstack([estimate]*3)

class Broken(WofsExist):
    """Masked spatial averaging mispredicts when one class is obscured"""
    def compute(self, time, data):
        estimate = data.shape[1] * data.mean(axis=1).filled(np.nan)
        return np.vstack([estimate]*3)

class Conservative(WofsExist):
    """Least assumptions"""
    def compute(self, time, data):
        lower = data.sum(axis=1).filled(0)
        upper = data.shape[1] - (~data).sum(axis=1).filled(0)
        estimate = 0.5 * (lower + upper)
        return np.vstack([lower, estimate, upper])

class Bracket(WofsMask):
    def compute(self, time, obs):
        data, mask = obs.data, obs.mask
        clear = ~mask
        epochs = len(time)

        past = clear.cumsum(axis=0, dtype=np.bool)
        future = clear[::-1,...].cumsum(axis=0, dtype=np.bool)[::-1,...]
        unbound = ~(past & future)

        # constant-value extrapolation
        forward = data
        reverse = data.copy()
        for i in range(1, epochs):
            gaps = mask[i]
            forward[i][gaps] = forward[i-1][gaps]
        for i in range(epochs-1)[::-1]:
            gaps = mask[i]
            reverse[i][gaps] = reverse[i+1][gaps]

        upper = forward | reverse | unbound # might be wet
        lower = forward & reverse & ~unbound # must be wet

        estimate = 0.5 * np.add(lower, upper, dtype=np.float32)

        # extrapolate tails
        ever = np.any(clear, axis=0)
        tail = ~future & ever
        estimate[tail] = forward[tail]
        nose = ~past & ever
        estimate[nose] = reverse[nose]

        upper = upper.sum(axis=1)
        lower = lower.sum(axis=1)
        estimate = estimate.sum(axis=1)
        #estimate = 0.5 * (lower + upper)

        return np.vstack([lower, estimate, upper])

#class TelegraphProcess(WofsMask):
#    pass

