import numpy as np
from joblib import Parallel, delayed

class TrackingInvariant:

    def __init__(self, errorSet, romSet, sampleSchedule, s2sDynamics, Nsim, logger=None):

        self.errorSet = errorSet
        self.romSet = romSet
        self.sampleSchedule = sampleSchedule
        self.s2sDynamics = s2sDynamics

        self.reachableList = None
        self.reachableTable = {}
        self.setDescriptions = {0: errorSet.getDesc()}

        self.iteration = 0
        self.Nsim = Nsim
        self.logger = logger

    @staticmethod
    def S2SDyn_helper(points, romPoints, s2sDyn):
        propogatedPoints = np.zeros_like(points)
        for ii in range(points.shape[0]):
            x0 = points[ii, :]
            rom0 = romPoints[ii, :]

            propogatedPoints[ii, :] = s2sDyn(x0, rom0)

        return propogatedPoints       

    def iterateSetMap(self, verbose:bool=True) -> None: 
        N = self.sampleSchedule(self.iteration)
        points = self.errorSet.sampleSet(N)
        romPoints = self.romSet.sampleSet(N)

        ptsPer = points.shape[0] // self.Nsim
        subsets = [
            (points[ii * ptsPer:min((ii + 1) * ptsPer, points.shape[0]), :] , romPoints[ii * ptsPer:min((ii + 1) * ptsPer, points.shape[0]), :])
            for ii in range(self.Nsim)
        ]
        results = Parallel(n_jobs=self.Nsim, verbose=verbose)(delayed(TrackingInvariant.S2SDyn_helper)(subsetPts, romSubPts, self.s2sDynamics) for subsetPts, romSubPts in subsets)
        
        propogatedPoints = np.vstack(results)
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

        # Log S2S dynamics
        if self.logger is not None:
            for ind in range(propogatedPoints.shape[0]):
                self.logger.write(np.hstack((self.iteration, points[ind, :], propogatedPoints[ind, :])))

        
    def verboseOut(self) -> None:
        """Print some statistics after a run
        """
        # First, proportion of points which were set invariant
        prop_in_set = self.errorSet.inSet(self.reachableTable[self.iteration])
        print(f"Proportion of Points in set: {np.sum(prop_in_set) / np.size(prop_in_set)}")
        print(self.errorSet.DeltaString())
