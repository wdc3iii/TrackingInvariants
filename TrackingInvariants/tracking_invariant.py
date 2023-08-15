import numpy as np
from tqdm import trange

class TrackingInvariant:

    def __init__(self, errorSet, sampleSchedule, s2sDynamics, logger=None):

        self.errorSet = errorSet
        self.sampleSchedule = sampleSchedule
        self.s2sDynamics = s2sDynamics

        self.reachableList = None
        self.reachableTable = {}
        self.setDescriptions = {0: errorSet.getDesc()}

        self.iteration = 0
        self.logger = logger

    def iterateSetMap(self, verbose:bool=True) -> None:

        if verbose:
            print(f"Iteration {self.iteration}")
            print("..... Sampling Set .....")
        
        points = self.errorSet.sampleSet(self.sampleSchedule(self.iteration))

        if verbose:
            print("..... Computing S2S .....")
        func = trange if verbose else range
        propogatedPoints = np.zeros_like(points)
        for ii in func(points.shape[0]):
            x0 = points[ii, :]

            propogatedPoints[ii, :] = self.s2sDynamics(x0)

            if self.logger is not None:
                self.logger.write(np.hstack((self.iteration, points[ii, :], propogatedPoints[ii, :])))
        
        self.reachableList = np.vstack((self.reachableList, propogatedPoints))
        self.iteration += 1
        self.reachableTable[self.iteration] = propogatedPoints

        if verbose:
            self.verboseOut()

        if verbose:
            print("..... Fitting Set .....")
        # Fit a new convex outerapproximation
        self.errorSet.fitSet(self.reachableList)
        self.setDescriptions[self.iteration] = self.errorSet.getDesc()

        
    def verboseOut(self) -> None:
        """Print some statistics after a run
        """
        # First, proportion of points which were set invariant
        prop_in_set = self.errorSet.inSet(self.reachableTable[self.iteration])
        print(f"Proportion of Points in set: {np.sum(prop_in_set) / np.size(prop_in_set)}")
        print(self.errorSet.DeltaString())
