import numpy as np
from tqdm import trange
from collections.abc import Callable
from HLIP.utils.logger import Logger
from HLIP.control_py.hlip_controller import HLIPControllerPD_GC
from SetRepresentations.set_representation import AbstractSet, ExtremePoints
from HLIP.simulation_py.mujoco_interface import MujocoInterface

Q_IK = np.array([
    0, 0, 0, -0.4, 0.8, -0.4, 0.8
])
QD_ZERO = np.zeros_like(Q_IK)

class TrackingInvariant:

    def __init__(
            self, errorSet:AbstractSet, romSet:AbstractSet,
            s2sDynamics, # :Callable[[np.ndarray, np.ndarray], tuple]
            errorLogger:Logger=None, setLogger:Logger=None
    ):

        self.errorSet = errorSet
        self.initErrorSet = None
        self.romSet = romSet
        self.s2sDynamics = s2sDynamics

        self.reachableList = None
        self.reachableTable = {}
        self.setDescriptions = {0: errorSet.getDesc()}

        self.iteration = 0
        self.proportionInSet = 0
        self.errorLogger = errorLogger
        self.setLogger = setLogger

    def iterateSetMap(self, N, verbose:bool=True, converged:bool=False) -> None:

        if verbose:
            print(f"\nIteration {self.iteration}")
            print("..... Sampling Set .....")
        
        points = self.errorSet.sampleSet(N) if self.initErrorSet is None else self.initErrorSet.sampleSet(N)
        romPoints = self.romSet.sampleSet(N)

        if self.reachableList is None:
            self.reachableList = points
            self.reachableTable[self.iteration] = points

        if verbose:
            print("..... Computing S2S .....")
        func = trange if verbose else range
        propogatedPoints = np.zeros_like(points)
        for ii in func(points.shape[0]):
            x0 = points[ii, :]
            rom0 = romPoints[ii, :]

            propogatedPoints[ii, :], IK_error = self.s2sDynamics(x0, rom0)

            if self.errorLogger is not None:
                self.errorLogger.write(np.hstack((self.iteration, IK_error, points[ii, :], romPoints[ii, :], propogatedPoints[ii, :])))
        
        self.reachableList = np.vstack((self.reachableList, propogatedPoints))
        self.iteration += 1
        self.reachableTable[self.iteration] = propogatedPoints

        self.calcProportionInSet()
        if verbose:
            self.verboseOut()

        if verbose:
            print("..... Fitting Set .....")
        # Fit a new convex outerapproximation
        self.errorSet.fitSet(self.reachableList)
        self.setDescriptions[self.iteration] = self.errorSet.getDesc()
        if self.setLogger is not None:
            self.setLogger.write(np.hstack((self.iteration, self.errorSet.getLog())))

    
    def calcProportionInSet(self) -> float:
        if self.iteration > 0:
            inSet = self.errorSet.inSet(self.reachableTable[self.iteration])
            self.proportionInSet = np.sum(inSet) / inSet.size
    
    def getProportionInSet(self):
        return self.proportionInSet

    def verboseOut(self) -> None:
        """Print some statistics after a run
        """
        # First, proportion of points which were set invariant
        prop_in_set = self.getProportionInSet()
        print(f"Proportion of Points in set: {np.sum(prop_in_set) / np.size(prop_in_set)}")
        print(F"Volume of Set: {self.errorSet.getVolume()}")
        print(self.errorSet.deltaString())


def s2sDynamics(mjInt:MujocoInterface, ctrl:HLIPControllerPD_GC, e0:np.ndarray, rom0:np.ndarray, vis:bool=False, video:bool=False) -> tuple:
    u_prev = rom0[0] # Previous step length
    u_nom = rom0[3]  # Current (i.e. in progress) step length
    z0 = rom0[1:3]   # Post-impact state

    # get nominal initial condition (Pre-Impact)
    y0, yd0, yF, ydF = ctrl.getNominalS2S(u_prev, z0, u_nom)

    # Modify IC with the error
    y0 += e0[:5]
    # And compute the initial position
    q0, _ = ctrl.adamKin.solveIK(Q_IK, y0, False)
    # Modify for contact
    mjInt.setState(q0, QD_ZERO)
    mjInt.forward()
    ft_pos = mjInt.getFootPos()
    q0[1] -= ft_pos[1][1] - ctrl.adamKin.getContactOffset(q0, False)
    
    # Modify with error
    yd0 += e0[5:]
    # Compute initial velocity
    qd0 = ctrl.adamKin.solveIKVel(q0, yd0, False)

    # Set the simulator state
    mjInt.setState(q0, qd0)
    mjInt.forward()
    # Get the starting time of the simulation
    startTime = mjInt.time()
    # Reset the controller
    ctrl.reset()

    frames = []

    IK_error = False
    while True:
        if vis:
            mjInt.updateScene()
        if video:
            frames.append(mjInt.readPixels())

        # Compute the S2S time
        t = mjInt.time() - startTime
        # Query state from Mujoco
        qpos = mjInt.getGenPosition()
        qvel = mjInt.getGenVelocity() 
        # Compute control action 
        q_pos_ref, q_vel_ref, q_ff_ref, sol_found = ctrl.gaitController(qpos, qvel, u_nom, t)

        if not sol_found:
            IK_error = True

        # Apply control action
        mjInt.jointPosCmd(q_pos_ref)
        mjInt.jointVelCmd(q_vel_ref)
        mjInt.jointTorCmd(q_ff_ref)

        # If stance foot has changed, stop simulation
        if not ctrl.cur_stf:
            break

        # Step simulation forward
        mjInt.step()
    
    # Swap legs (take advantage of symmetry)
    qpos_copy = np.copy(qpos)
    qvel_copy = np.copy(qvel)
    qpos[3:5] = qpos_copy[5:]
    qpos[5:] = qpos_copy[3:5]
    qvel[3:5] = qvel_copy[5:]
    qvel[5:] = qvel_copy[3:5]
    # Compute with stanceFoot = False to swap legs
    qCpos = ctrl.adamKin.calcOutputs(qpos, False)
    qCvel = ctrl.adamKin.calcDOutputs(qpos, qvel, False)

    e1 = qCpos - yF
    ed1 = qCvel - ydF

    if video:
        return frames
    return np.hstack((e1, ed1)), IK_error