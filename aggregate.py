

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

def wofs_ma(stack):
    stack.water.data &= ~np.uint8(4) # remove ocean masking

    wet = stack.water == 128
    dry = stack.water == 0
    clear = wet | dry

    masked = np.ma.array(wet.data, mask=~clear.data)

    return masked, stack.time

class AggregateDynamics:
    pixels = None
    timeaxis = None
    upper = None
    lower = None
    bestestimate = None
    def __init__(self, stack):
        stack.water.data &= ~np.uint8(4) # remove ocean masking

        wet = stack.water == 128
        dry = stack.water == 0
        clear = wet | dry

        self.masked = np.ma.array(wet.data, mask=~clear.data)
        t,x,y = self.masked.shape
        self.pixels = x*y
        self.timeaxis = stack.time.data
        self.compute()
        if self.bestestimate is None:
            self.bestestimate = 0.5 * (self.lower + self.upper)
    def compute(self):
        raise NotImplementedError
    def plot(self, axes=None):
        ax = axes or plt.gca()
        envelope = ax.fill_between(self.timeaxis, self.upper, self.lower, alpha=0.3)
        line = ax.plot(self.timeaxis, self.bestestimate)
        return [envelope, line] # should rather be a join
    def __str__(self):
        return '<{n} aggregated areas>'.format(n=self.pixels)
    def __add__(self, other):
        if not isinstance(other, AggregateDynamics):
            raise TypeError
        new = AggregateDynamics.__new__()
        new.pixels = self.pixels + other.pixels
        assert len(self.timeaxis) == len(other.timeaxis)
        assert all(self.timeaxis == other.timeaxis)
        # TODO: trim/filter time series if operands not perfectly matched
        new.timeaxis = self.timeaxis
        new.upper = self.upper + other.upper
        new.lower = self.lower + other.lower
        new.bestestimate = self.bestestimate + other.bestestimate
        return new

class Naive(AggregateDynamics):
    def compute(self):
        self.bestestimate = self.masked.sum(axis=(1,2)).filled(0)
    def plot(self, axes=None):
         ax = axes or plt.gca()
         return ax.plot(self.timeaxis, self.bestestimate,
                        marker='.', alpha=0.7)
class Broken(Naive):
    def compute(self):
        self.bestestimate = self.pixels * self.masked.mean(axis=(1,2))

        self.timeaxis = self.timeaxis[~self.bestestimate.mask]
        self.bestestimate = self.bestestimate[~self.bestestimate.mask]

class NaiveBounds(AggregateDynamics):
    def compute(self):
        self.lower = self.masked.sum(axis=(1,2)).filled(0)
        self.upper = self.pixels - (~self.masked).sum(axis=(1,2)).filled(0)
    def plot(self, axes=None):
        ax = axes or plt.gca()
        n = len(self.timeaxis)
        tt = np.vstack([self.timeaxis]*2)
        ww = np.vstack([self.lower, np.zeros(n)])
        dd = np.vstack([self.upper, np.ones(n)*self.pixels])
        ax.plot(tt, ww, c='b', alpha=0.3)
        ax.plot(tt, dd, c='r', alpha=0.3)
        #ax.plot(tt, np.vstack([self.upper, self.lower]), c='k')
        ax.plot(self.timeaxis, self.bestestimate, marker='.', linewidth=0, alpha=0.3, c='k')


class ConstantBracket(AggregateDynamics):
    def compute(self):
        mask = self.masked.mask
        forward = self.masked.data.copy()
        reverse = forward.copy()

        epochs = len(self.timeaxis)

        past = mask.cumsum(axis=0)
        future = mask[::-1,...].cumsum(axis=0)[::-1,...]
        unbound = (past == 0) | (future == 0)

        # constant-value extrapolation
        for i in range(1, epochs):
            gaps = mask[i]
            forward[i][gaps] = forward[i-1][gaps]
        for i in range(epochs-1)[::-1]:
            gaps = mask[i]
            reverse[i][gaps] = reverse[i+1][gaps]

        upper = forward | reverse | unbound
        lower = forward & reverse & ~unbound

        self.model = 0.5 * (upper + lower)

        self.upper = upper.sum(axis=(1,2))
        self.lower = lower.sum(axis=(1,2))


class TelegraphProcess(AggregateDynamics):
    pass




