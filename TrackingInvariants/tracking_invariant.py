import numpy as np
from tqdm import trange
from collections.abc import Callable
from HLIP.utils.logger import Logger
from HLIP.control_py.hlip_controller import HLIPControllerPD_GC
from SetRepresentations.set_representation import AbstractSet
from HLIP.simulation_py.mujoco_interface import MujocoInterface

Q_IK = np.array([
    0, 0, 0, -0.4, 0.8, -0.4, 0.8
])
QD_ZERO = np.zeros_like(Q_IK)

class TrackingInvariant:

    def __init__(
            self, errorSet:AbstractSet, romSet:AbstractSet, sampleSchedule, #:Callable[[int, bool], int]
            s2sDynamics, # :Callable[[np.ndarray, np.ndarray], tuple]
            errorLogger:Logger=None, setLogger:Logger=None
    ):

        self.errorSet = errorSet
        self.romSet = romSet
        self.sampleSchedule = sampleSchedule
        self.s2sDynamics = s2sDynamics

        self.reachableList = None
        self.reachableTable = {}
        self.setDescriptions = {0: errorSet.getDesc()}

        self.iteration = 0
        self.errorLogger = errorLogger
        self.setLogger = setLogger

    def iterateSetMap(self, verbose:bool=True, converged:bool=False) -> None:

        if verbose:
            print(f"Iteration {self.iteration}")
            print("..... Sampling Set .....")
        
        N = self.sampleSchedule(self.iteration, converged)
        points = self.errorSet.sampleSet(N)
        romPoints = self.romSet.sampleSet(N)

        if verbose:
            print("..... Computing S2S .....")
        func = trange if verbose else range
        propogatedPoints = np.zeros_like(points)
        for ii in func(points.shape[0]):
            x0 = points[ii, :]
            rom0 = romPoints[ii, :]

            propogatedPoints[ii, :] = self.s2sDynamics(x0, rom0)

            if self.errorLogger is not None:
                self.errorLogger.write(np.hstack((self.iteration, points[ii, :], romPoints[ii, :], propogatedPoints[ii, :])))
        
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
        if self.setLogger is not None:
            self.setLogger.write(np.hstack((self.iteration, self.errorSet.getLog())))

    
    def getProportionInSet(self) -> float:
        if self.iteration == 0:
            return 0
        return self.errorSet.inSet(self.reachableTable[self.iteration])
    
    def verboseOut(self) -> None:
        """Print some statistics after a run
        """
        # First, proportion of points which were set invariant
        prop_in_set = self.getProportionInSet()
        print(f"Proportion of Points in set: {np.sum(prop_in_set) / np.size(prop_in_set)}")
        print(self.errorSet.DeltaString())


def sampleSchedule(iter_:int, converged:bool=False) -> int:
    if converged:
        return 1000
    if iter_ < 5:
        return 20
    if iter_ < 10:
        return 100
    return 250


def s2sDynamics(mjInt:MujocoInterface, ctrl:HLIPControllerPD_GC, e0:np.ndarray, rom0:np.ndarray, vis:bool=False) -> tuple:
    u_prev = rom0[0]
    u_nom = rom0[3]
    z0 = rom0[1:3]

    # get nominal initial condition
    y0 = np.array([
        ctrl.pitch_ref,
        u_prev,
        0,
        z0[0] + u_prev,
        ctrl.z_ref
    ])
    # Modify with the error
    y0 += e0[:5]
    # And compute the initial position
    q0, _ = ctrl.adamKin.solveIK(Q_IK, y0, False)
    # Modify for contact
    mjInt.setState(q0, QD_ZERO)
    mjInt.forward()
    ft_pos = mjInt.getFootPos()
    q0[1] -= ft_pos[1][1] - ctrl.adamKin.getContactOffset(q0, False)
    
    # Construct nominal velocity
    yd0 = np.array([
        0, 0, ctrl.swf_pos_z_poly.evalPoly(ctrl.T_SSP, 1), z0[1], 0, 0, 0
    ])
    # Modify with error
    yd0[:5] += e0[5:]
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

    while True:
        if vis:
            frames.append(mjInt.readPixels())

        # Compute the S2S time
        t = mjInt.time() - startTime
        # Query state from Mujoco
        qpos = mjInt.getGenPosition()
        qvel = mjInt.getGenVelocity() 
        # Compute control action 
        q_pos_ref, q_vel_ref, q_ff_ref = ctrl.gaitController(qpos, qvel, u_nom, t)

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

    z1 = ctrl.hlip(z0, u_nom)
    e1 = qCpos - np.array([
        ctrl.pitch_ref,
        u_nom,
        0,
        z1[0] + u_nom,
        ctrl.z_ref
    ])
    ed1 = qCvel - np.array([
        0, 0, ctrl.swf_pos_z_poly.evalPoly(ctrl.T_SSP, 1), z1[1], 0
    ])

    if vis:
        return frames
    return np.hstack((e1, ed1))