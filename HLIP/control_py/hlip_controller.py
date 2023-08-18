import numpy as np
from HLIP.utils.logger import Logger
from HLIP.control_py.poly import Poly
from HLIP.control_py.bezier import Bezier
from HLIP.kinematics_py.adam_kinematics import Kinematics
from abc import ABC, abstractmethod


class HLIPController(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def gaitController(self, q_pos:np.ndarray, q_vel:np.ndarray, u_nom:float, t:float) -> tuple:
        pass

class HLIPControllerPD_GC:

    def __init__(
            self, T_SSP:float, z_ref:float, urdf_path:str, mesh_path:str, mass:float,
            pitch_ref:float=0.025, x_bez:np.ndarray=np.array([0,0,0,1,1]),
            vswf_tof:float=0.05, vswf_imp:float=-0.05, zswf_max:float=0.075, pswf_max:float=0.7,
            logger:Logger=None
        ):
        self.T_SSP = T_SSP
        self.g = 9.81
        self.mass = mass

        self.z_ref = z_ref
        self.calcLambda()
        self.calcSigma1()
        self.calcSigma2()
        self.pitch_ref = pitch_ref

        self.computeGain()

        self.cur_stf = False
        self.cur_swf = not self.cur_stf

        self.swf_x_start = 0

        self.t_phase_start = -2 * self.T_SSP

        self.pos_swf_imp = 0
        self.v_swf_tof = vswf_tof
        self.v_swf_imp = vswf_imp
        self.z_swf_max = zswf_max
        self.t_swf_max_height = pswf_max

        self.adamKin = Kinematics(urdf_path, mesh_path, False)

        self.swf_x_bez = Bezier(x_bez)

        x_swf_pos_z = np.array([0, self.T_SSP, self.t_swf_max_height * self.T_SSP, 0, self.T_SSP])
        y_swf_pos_z = np.array([0, self.pos_swf_imp, self.z_swf_max, self.v_swf_tof, self.v_swf_imp])
        d_swf_pos_z = np.array([0, 0, 0, 1, 1])
        self.swf_pos_z_poly = Poly(x_swf_pos_z, y_swf_pos_z, d_swf_pos_z)
        z_bez = np.array([0, 0.25 * self.z_swf_max, 0.5 * self.z_swf_max, self.z_swf_max, 0])
        self.swf_pos_z_bez = Bezier(z_bez)

        self.logger = logger

    def calcPreImpactStateRef_HLIP(self, u_ref:float) -> np.ndarray:
        sigma_1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
        p_pre_ref = -u_ref / 2
        v_pre_ref = sigma_1 * u_ref / 2
        return np.array([p_pre_ref, v_pre_ref])
    
    def calcPreimpactState(self, x0:np.ndarray, t:float) -> np.ndarray:
        V = np.array([[1, 1], [self.lmbd, -self.lmbd]])
        S = np.array([[np.exp(self.lmbd * t), 0], [0, np.exp(-self.lmbd * t)]])

        return V @ S @ np.linalg.inv(V) @ x0

    def getU(self) -> np.ndarray:
        return self.u
    
    def calcLambda(self) -> None:
        self.lmbd = np.sqrt(self.g / self.z_ref)

    def computeGain(self) -> None:
        self.K_deadbeat = np.array([1, 1 / (self.lmbd * np.tanh(self.T_SSP * self.lmbd))])
    
    def calcSigma1(self) -> None:
        self.sigma1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
    
    def calcSigma2(self) -> None:
        self.sigma2 = self.lmbd * np.tanh(self.T_SSP * self.lmbd / 2)
    
    # def calcD2(self, lmbd:float, T_SSP:float, T_DSP:float, v_ref:float) -> float:
    #     return (lmbd * lmbd / np.cosh(lmbd * T_SSP / 2) * (T_SSP + T_DSP) * v_ref) / (lmbd * lmbd * T_DSP + 2 * HLIPController.calcSigma2(lmbd, T_SSP))
    
    def setT_SSP(self, T_SSP:float) -> None:
        self.T_SSP = T_SSP

    def setZ_ref(self, z_ref):
        self.z_ref = z_ref

    def setPitchRef(self, pitch_ref):
        self.pitch_ref = pitch_ref

    def gaitController(self, q_pos:np.ndarray, q_vel:np.ndarray, u_nom:float, t:float) -> tuple:
        t_phase = t - self.t_phase_start

        t_scaled = t_phase / self.T_SSP

        y_out = self.adamKin.calcOutputs(q_pos, self.cur_stf)
        swf_height = y_out[Kinematics.OUT_ID["SWF_POS_Z"]]

        
        if t_scaled >= 1 or (t_scaled > 0.5 and swf_height < 0.001):
            t_scaled = 0
            t_phase = 0
            self.t_phase_start = t

            self.cur_stf = not self.cur_stf
            self.cur_swf = not self.cur_swf

            # Recompute outputs with relabeled stance/swing feet
            y_out = self.adamKin.calcOutputs(q_pos, self.cur_stf)
            self.swf_x_start = y_out[Kinematics.OUT_ID["SWF_POS_X"]]

            self.calcLambda()
            self.calcSigma1()
            self.calcSigma2()

        # X-Dynamics
        x_ssp_impact_ref = np.array([
            u_nom / 2,
            self.sigma1 * u_nom / 2
        ])

        v_com_use = self.adamKin.getVCom(q_pos, q_vel)
        x_ssp_curr = np.array([
            y_out[Kinematics.OUT_ID["COM_POS_X"]],
            v_com_use[0]
        ])

        x_ssp_impact = self.calcPreimpactState(x_ssp_curr, self.T_SSP - t_phase)
        
        self.u = u_nom + self.K_deadbeat @ (x_ssp_impact - x_ssp_impact_ref)

        if self.u > 0.5:
            print(f"Large Step {self.u} Requested")

        bht = self.swf_x_bez.eval(t_scaled)

        # New method, relative to swing foot position at beginning of stride
        swf_pos_x_ref = self.swf_x_start * (1 - bht) + self.u * bht

        # Z-pos
        swf_pos_z_ref = self.swf_pos_z_poly.evalPoly(t_phase, 0)
        # swf_pos_z_ref = self.swf_pos_z_bez.eval(t_scaled)

        y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
        y_out_ref[Kinematics.OUT_ID["PITCH"]] = self.pitch_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = swf_pos_x_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = swf_pos_z_ref
        y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = y_out[Kinematics.OUT_ID["COM_POS_X"]] # No control authority over x position
        y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = self.z_ref

        q_gen_ref, sol_found = self.adamKin.solveIK(q_pos, y_out_ref, self.cur_stf)

        if not sol_found:
            print('No solution found for IK', y_out_ref)

        q_ref = q_gen_ref[-4:]
        # Set desired joint velocities/torques
        qd_ref = np.zeros((Kinematics.N_JOINTS,))

        q_ff_ref_gravcomp = self.adamKin.calcGravityCompensation(q_pos, self.cur_stf)

        return q_ref, qd_ref, q_ff_ref_gravcomp, sol_found
    
    def reset(self):
        self.cur_stf = False
        self.cur_swf = not self.cur_stf

        self.swf_x_start = 0

        self.t_phase_start = -2 * self.T_SSP
    
    def getNominalS2S(self, u_prev, x0, u_nom):
        x1 = self.calcPreimpactState(x0, self.T_SSP)
        y0 = np.array([
            self.pitch_ref,     # Desired pitch
            u_prev,             # Xswf is step length
            0,                  # Zswf = 0 pre-impact
            x0[0] + u_prev,     # Xcom = Xhlip(post_impact) + u_prev (to get pre-impact)
            self.z_ref          # Zcom = Zref
        ])
        yd0 = np.array([
            0, self.swf_x_bez.deval(0), self.swf_pos_z_poly.evalPoly(0, 1), x0[1], 0
        ])
        yF = np.array([
            self.pitch_ref,     # Desired pitch
            u_nom,             # Xswf is step length
            0,                  # Zswf = 0 pre-impact
            x1[0],              # Xcom = Xhlip(pre_impact)
            self.z_ref          # Zcom = Zref
        ])
        ydF = np.array([
            0, self.swf_x_bez.deval(1), self.swf_pos_z_poly.evalPoly(self.T_SSP, 1), x1[1], 0
        ])
        return y0, yd0, yF, ydF

    # def hlip(self, x, u):
    #     x[0] -= u
    #     return self.calcPreimpactState(x, self.T_SSP)
