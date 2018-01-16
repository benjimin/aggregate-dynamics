

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

"""
class AggregateDynamics:
    pixels = None
    timeaxis = None
    upper = None
    lower = None
    bestestimate = None
    def __init__(self, data, time):
        raise NotImplementedError
    def plot(self, axes=None):
        ax = axes or plt.gca()
        return [ax.plot(i) for i in j]
    def __repr__(self):
        pass
    def __add__(self, other):
        raise NotImplementedError


class ConstantBracket(AggregateDynamics):
    pass

class TelegraphProcess(AggregateDynamics):
    pass