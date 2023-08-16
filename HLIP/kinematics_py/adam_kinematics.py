import numpy as np
import pinocchio as pin

class Kinematics:
    
    OUT_ID = {
        "PITCH": 0, "SWF_POS_X": 1, "SWF_POS_Z": 2,
        "COM_POS_X": 3, "COM_POS_Z": 4
    }
    OUT_IK_ID = {
        "R_X": 0, "R_Y": 1, "R_Z": 2, "SWF_POS_X": 3, "SWF_POS_Z": 4,
        "COM_POS_X": 5, "COM_POS_Z": 6
    }
    GEN_POS_ID = {
        "P_X": 0, "P_Z": 1, "R_Y": 2, 
        "P_LHP": 3, "P_LKP": 4, "P_RHP": 5, "P_RKP": 6,
    }
    GEN_VEL_ID = {
        "V_X": 0, "V_Z": 1, "W_Y": 2,
        "V_LHP": 3, "V_LKP": 4, "V_RHP": 5, "V_RKP": 6
    }
    JOINT_ID = {"P_LHP": 0, "P_LKP": 1, "P_RHP": 2, "P_RKP": 3}

    N_JOINTS = 4
    N_POS_STATES = 7
    N_VEL_STATES = 7
    N_OUTPUTS = 5

    def __init__(self, urdf_path: str, mesh_path:str, use_static_com: bool=False, eps:float=1e-4, damping_factor:float=1e-6, alpha:float=0.2, max_iter:int=300):
        self.eps = eps
        self.alpha = alpha
        self.damping_factor = damping_factor
        self.max_iter = max_iter

        self.use_static_com = use_static_com

        self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, mesh_path)

        self.pin_data = self.pin_model.createData()

        self.TORSO_FID = self.pin_model.getFrameId("torso")
        self.LEFT_HIP_YAW_FID = self.pin_model.getFrameId("left_hip_yaw")
        self.RIGHT_HIP_YAW_FID = self.pin_model.getFrameId("right_hip_yaw")
        self.LEFT_FOOT_FID = self.pin_model.getFrameId("left_foot")
        self.RIGHT_FOOT_FID = self.pin_model.getFrameId("right_foot")
        self.LEFT_HIP_ROLL_FID = self.pin_model.getFrameId("left_hip_roll")
        self.RIGHT_HIP_ROLL_FID = self.pin_model.getFrameId("right_hip_roll")
        self.LEFT_HIP_PITCH_FID = self.pin_model.getFrameId("left_hip_pitch")
        self.RIGHT_HIP_PITCH_FID = self.pin_model.getFrameId("right_hip_pitch")
        self.LEFT_SHIN_FID = self.pin_model.getFrameId("left_shin")
        self.RIGHT_SHIN_FID = self.pin_model.getFrameId("right_shin")
        self.STATIC_COM_FID = self.pin_model.getFrameId("static_com")

        q_nom = self.getZeroPos()
        self.updateModelPose(q_nom)

    def calcOutputs(self, q: np.ndarray, stanceFoot: bool) -> np.ndarray:
        self.updateModelPose(q)
        
        stf_fid = self.stanceFootID(stanceFoot)
        swf_fid = self.swingFootID(stanceFoot)

        if self.use_static_com:
            com_pos_world = self.pin_data.oMf[self.STATIC_COM_FID].translation
        else:
            com_pos_world = pin.centerOfMass(self.pin_model, self.pin_data)
        
        stf_pos_world = self.pin_data.oMf[stf_fid].translation
        swf_pos_world = self.pin_data.oMf[swf_fid].translation

        y_out = np.array([
            q[Kinematics.GEN_POS_ID["R_Y"]],
            swf_pos_world[0] - stf_pos_world[0],
            swf_pos_world[2] - stf_pos_world[2],
            com_pos_world[0] - stf_pos_world[0],
            com_pos_world[2] - stf_pos_world[2]
        ])

        return y_out
    
    def calcDOutputs(self, q:np.ndarray, qdot:np.ndarray, stanceFoot:bool) -> np.ndarray:
        self.updateModelPose(q)

        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        if self.use_static_com:
            Jcom = pin.getFrameJacobian(self.pin_model, self.pin_data, self.STATIC_COM_FID, pin.LOCAL_WORLD_ALIGNED)[0:3:2, :]
        else:
            Jcom = self.getCoMJacobian(q)[0:3:2, :]

        Jswf = self.getSWFJacobian(q, stanceFoot)[0:3:2, :]
        Jstf = self.getSTFJacobian(q, stanceFoot)[0:3:2, :]
        v_stf = Jstf @ qdot
        
        return np.hstack((
            qdot[2],
            Jswf @ qdot,
            Jcom @ qdot
        ))

    def getCoMJacobian(self, q:np.ndarray) -> np.ndarray:
        return pin.jacobianCenterOfMass(self.pin_model, self.pin_data, q, False)
    
    def getCoMJacobianTimeDerivative(self, q:np.ndarray, qdot:np.ndarray) -> np.ndarray:
        pin.centerOfMass(self.pin_model, self.pin_data, q, qdot)
        return pin.getCenterOfMassVelocityDerivatives(self.pin_model, self.pin_data)
    
    def getSWFJacobian(self, q:np.ndarray, stanceFoot:bool) -> np.ndarray:
        swf_fid = self.swingFootID(stanceFoot)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        return pin.getFrameJacobian(self.pin_model, self.pin_data, swf_fid, pin.LOCAL_WORLD_ALIGNED)
    
    def getSWFJacobianTimeDerivative(self, q:np.ndarray, qdot:np.ndarray, stanceFoot:bool) -> np.ndarray:
        swf_fid = self.swingFootID(stanceFoot)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, qdot)
        return pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, swf_fid, pin.LOCAL_WORLD_ALIGNED)
    
    def getSTFJacobian(self, q:np.ndarray, stanceFoot:bool) -> np.ndarray:
        stf_fid = self.stanceFootID(stanceFoot)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        return pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)
    
    def getSTFJacobianTimeDerivative(self, q:np.ndarray, qdot:np.ndarray, stanceFoot:bool) -> np.ndarray:
        stf_fid = self.stanceFootID(stanceFoot)
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q, qdot)
        return pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)
    
    def getDynamics(self, q:np.ndarray, qdot:np.ndarray, stanceFoot:bool) -> tuple:
        # Lagrangian Mechanics 
        # Mddq + Cdq + G = Btau + Jh^T \lambda 
        # Jh ddq + dJh dq = 0
        # Jh dq = 0
        M = pin.crba(self.pin_model, self.pin_data, q)
        # M = m_tri + m_tri.T - np.diag(np.diag(m_tri))
        H = pin.nonLinearEffects(self.pin_model, self.pin_data, q, qdot)
        B = np.vstack((np.zeros((3, 4)), np.eye(4)))
        Jh = self.getSTFJacobian(q, stanceFoot)[0:3:2, :]
        dJh = self.getSTFJacobianTimeDerivative(q, qdot, stanceFoot)[0:3:2, :]
        return M, H, B, Jh, dJh

    def calcGravityCompensation(self, q:np.ndarray, stanceFoot:bool) -> np.ndarray:
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
        g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        Jc = pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)[0:3:2, :]

        Q, R = np.linalg.qr(Jc.transpose(), mode='complete')
        Su = np.hstack((np.zeros((5, 2)), np.eye(5)))
        S = np.vstack((np.zeros((3, 4)), np.eye(4))).transpose()
        return np.linalg.pinv(Su @ np.transpose(Q) @ np.transpose(S)) @ Su @ np.transpose(Q) @ g

        # Attempt with phantom ankle actuation for gravity comp
        # Su = np.hstack((np.zeros((5, 2)), np.eye(5)))
        # S = np.vstack((np.zeros((2, 5)), np.eye(5))).transpose()
        # tau = np.linalg.solve(Su @ np.transpose(Q) @ np.transpose(S), Su @ np.transpose(Q) @ g)
        # return tau[-4:]

        # Lol there are contact dynamics this doesn't consider
        # return g[-4:]

    def fk_CoM(self) -> np.ndarray:
        return self.pin.centerOfMass(self.pin_model, self.pin_data)

    def fk_StaticCom(self) -> np.ndarray:
        self.pin_data.oMf[self.STATIC_COM_FID].translation

    def v_CoM(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        Jcom = pin.jacobianCenterOfMass(self.pin_model, self.pin_data, q)
        return Jcom @ qd

    def v_StaticCom(self, q:np.ndarray, qd:np.array) -> np.ndarray:
        self.updateModelPose(q)
        Jstatic = pin.getFrameJacobian(self.pin_model, self.pin_data, self.STATIC_COM_FID, pin.LOCAL_WORLD_ALIGNED)
        return Jstatic @ qd

    def getVCom(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        if self.use_static_com:
            return self.v_StaticCom(q, qd)
        return self.v_CoM(q, qd)
    
    def getComMomentum(self, q, qd) -> np.ndarray:
        return pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, qd)
    
    def getSTFAngularMomentum(self, q, qd, stf) -> np.ndarray:
        comMom = self.getComMomentum(q, qd)
        L_com = comMom.linear[0:3:2]
        angMomCom = comMom.angular[1]
        y_out = self.calcOutputs(q, stf)
        return angMomCom + L_com[0] * y_out[Kinematics.OUT_ID["COM_POS_Z"]] - L_com[1] * y_out[Kinematics.OUT_ID["COM_POS_X"]]
    
    def stanceFootID(self, stanceFoot:bool) -> int:
        return self.LEFT_FOOT_FID if stanceFoot else self.RIGHT_FOOT_FID
    
    def swingFootID(self, stanceFoot:bool) -> int:
        return self.RIGHT_FOOT_FID if stanceFoot else self.LEFT_FOOT_FID
    
    def getContactOffset(self, q:np.ndarray, stanceFoot:bool) -> float:
        d1 = np.sqrt(0.01**2 + 0.02**2)
        alpha = np.arctan(0.01 / 0.02)
        if stanceFoot:
            beta = abs(np.pi / 2 + q[2] + q[3] + q[4])
        else:
            beta = abs(np.pi / 2 + q[2] + q[5] + q[6])
        return d1 * np.sin(beta - alpha)

    def solveIK(self, q: np.ndarray, y_des: np.ndarray, stanceFoot: bool) -> tuple:
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
            swf_fid = self.RIGHT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
            swf_fid = self.LEFT_FOOT_FID

        ii = 0
        while ii < self.max_iter:
            y_out = self.calcOutputs(q, stanceFoot)
            y_err = y_des - y_out

            if np.linalg.norm(y_err) < self.eps:
                break

            pin.computeJointJacobians(self.pin_model, self.pin_data, q)
            J_torso_world = pin.getFrameJacobian(self.pin_model, self.pin_data, self.TORSO_FID, pin.LOCAL_WORLD_ALIGNED)
            J_stf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)
            J_swf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, swf_fid, pin.LOCAL_WORLD_ALIGNED)

            if self.use_static_com:
                J_com_world = pin.getFrameJacobian(self.pin_model, self.pin_data, self.STATIC_COM_FID, pin.LOCAL_WORLD_ALIGNED)
            else:
                J_lin = pin.jacobianCenterOfMass(self.pin_model, self.pin_data, q, False)
                J_com_world = np.vstack((J_lin, np.zeros_like(J_lin)))


            J_swf_rel = J_swf_world - J_stf_world
            J_com_rel = J_com_world - J_stf_world

            J_out = np.vstack([
                J_torso_world[4, :],
                J_swf_rel[0, :],
                J_swf_rel[2, :],
                J_com_rel[0, :],
                J_com_rel[2, :]
            ])

            JJt = J_out @ J_out.transpose()
            JJt += np.eye(Kinematics.N_OUTPUTS) * self.damping_factor

            v = J_out.transpose() @ np.linalg.solve(JJt, y_err)
            
            q = pin.integrate(self.pin_model, q, self.alpha * v)

            ii += 1

        q = Kinematics.wrapAngle(q)

        return q, ii < self.max_iter

    def solveIKVel(self, qpos, ydot, stanceFoot):
        Jy_out_ref = np.vstack((
            np.array([0, 0, 1, 0, 0, 0, 0]),
            self.getSWFJacobian(qpos, stanceFoot)[0:3:2, :],
            self.getCoMJacobian(qpos)[0:3:2, :],
            self.getSTFJacobian(qpos, stanceFoot)[0:3:2, :]
        ))

        qvel = np.linalg.inv(Jy_out_ref) @ ydot
        return qvel


    def fk_Frame(self, frame_name: str) -> np.ndarray:
        return self.pin_data.oMf[self.pin_model.getFrameId(frame_name)].translation

    def updateModelPose(self, q:np.ndarray) -> None:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    @staticmethod
    def wrapAngle(q: np.ndarray) -> np.ndarray:
        return np.mod((q + np.pi), 2 * np.pi) - np.pi

    def getZeroPos(self) -> np.ndarray:
        q_zero = np.zeros((self.pin_model.nq,))
        return q_zero


# if __name__ == "__main__":
#     from simulation_py.mujoco_interface import MujocoInterface

#     adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")
#     mjInt = MujocoInterface("rsc/models/adam2d.xml", vis_enabled=False)

#     # First, check the Forward kinematics in the zero (base) position
#     # q = np.array([-1.13989699e-03, -3.70178040e-02, 4.98557348e-02, -6.37404332e-01, 1.37197239e+00, -4.15495872e-01, 8.03258440e-01])
#     # qv = np.array([-0.01266918, 0.03132298, 0.27444285, -1.6057377, 5.6096941, -0.45891118, 0.05415881])

#     pitch = 1
#     q = np.array([-1.13989699e-03, 1, pitch, -6.37404332e-01, 1.37197239e+00, -4.15495872e-01, 8.03258440e-01])
#     qv = np.array([-1, 5, 4.47444285, 0, 0, 0, 0])

#     cp = np.cos(pitch)
#     sp = np.sin(pitch)
#     qv_body = np.hstack((qv[0] * cp - qv[1] * sp, qv[0] * sp + qv[1] * cp, qv[2:]))
#     qv_zero = np.zeros_like(qv)
#     # qzero = np.zeros_like(q)
#     # qvzero = np.zeros_like(qv)
#     stanceFoot = True
    
#     mjInt.setState(q, qv)
#     M, H, B, Jh, dJh = adamKin.getDynamics(q, qv, stanceFoot)

#     M_mjc, H_mjc, Jh_mjc, F_mjc = mjInt.getDynamics()

#     print(f"Mpin: {M}\nMmjc: {M_mjc}")
#     print(f"Mdiff{abs(M - M_mjc)}")
#     print(f"Hpin: {H}\nHmjc: {H_mjc}")
#     print(f"Hdiff{abs(H - H_mjc)}")
#     # print(f"Jpin: {Jh}\nJmjc: {Jh_mjc}")
#     # print(f"Jdiff{abs(Jh - Jh_mjc)}")


#     M, H, B, Jh, dJh = adamKin.getDynamics(q, qv_body, stanceFoot)


#     print(f"\n\nHpin: {H}\nHmjc: {H_mjc}")
#     print(f"Hdiff{abs(H - H_mjc)}")

#     print("here")

if __name__ == "__main__":
    adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")

    q_zero = adamKin.getZeroPos()
    # q_zero[2] = 0.2
    stanceFoot = True

    coff = adamKin.getContactOffset(q_zero, stanceFoot)

    print(f"coff: {coff}")

# if __name__ == "__main__":
#     adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")

#     q_zero = adamKin.getZeroPos()
#     stanceFoot = True

#     y_pin = adamKin.calcOutputs(q_zero, stanceFoot)
#     dy_pin = adamKin.calcDOutputs(q_zero, q_zero, stanceFoot)

#     tau = adamKin.calcGravityCompensation(q_zero, stanceFoot)

#     Jcom = adamKin.getCoMJacobian(q_zero)
#     dJcom = adamKin.getCoMJacobianTimeDerivative(q_zero, q_zero)

#     Jswf = adamKin.getSWFJacobian(q_zero, stanceFoot)
#     dJswf = adamKin.getSWFJacobianTimeDerivative(q_zero, q_zero, stanceFoot)

#     M, H, B, Jh, dJh = adamKin.getDynamics(q_zero, q_zero, stanceFoot)

#     print(f"y: {y_pin}\ndy: {dy_pin}\ntau: {tau}\nJcom: {Jcom}\ndJcom: {dJcom}\nJswf: {Jswf}\ndJswf: {dJswf}")

#     print(f"M: {M}\nH: {H}\nB: {B}\nJh: {Jh}\ndJh: {dJh}")

# if __name__ == "__main__":
#     from simulation_py.mujoco_interface import MujocoInterface

#     adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")
#     mjInt = MujocoInterface("rsc/models/adam2d.xml", vis_enabled=False)

#     # First, check the Forward kinematics in the zero (base) position
#     q_zero = adamKin.getZeroPos()
#     stanceFoot = True
#     y_pin = adamKin.calcOutputs(q_zero, stanceFoot)
    
#     q_zerovel = np.zeros_like(mjInt.getGenVelocity())
#     mjInt.setState(q_zero, q_zerovel)
#     mjInt.forward()

#     mj_com = mjInt.getCoMPosition()
#     mj_feet = mjInt.getFootPos()
#     mj_stance = mj_feet[int(stanceFoot)]
#     mj_swing = mj_feet[int(not stanceFoot)]
#     y_mj = np.hstack(
#         (q_zero[Kinematics.GEN_POS_ID["R_Y"]], mj_swing - mj_stance, mj_com - mj_swing)
#     )

#     print("Check the Forward kinematics in the zero (base) position")
#     print("Y (pin): ", y_pin)
#     print("Y (mj) : ", y_mj) 
#     print("error  : ", y_pin - y_mj)

#     # Now, check the pinocchio inverse kinematics
#     max_error = 0
#     delta = 0.1
#     for ii in range(1000):
#         print("\n\nIK Test ", ii)
#         q_sol = adamKin.getZeroPos()

#         q_sol[Kinematics.GEN_POS_ID["R_Y"]] = np.random.random() * 0.4 - 0.2
#         q_sol[Kinematics.GEN_POS_ID["P_LHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
#         q_sol[Kinematics.GEN_POS_ID["P_RHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
#         q_sol[Kinematics.GEN_POS_ID["P_LKP"]] = np.random.random() * np.pi / 2
#         q_sol[Kinematics.GEN_POS_ID["P_RKP"]] = np.random.random() * np.pi / 2

#         y_sol = adamKin.calcOutputs(q_sol, stanceFoot)

#         q_ik = np.copy(q_zero)
#         q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = q_sol[Kinematics.GEN_POS_ID["P_LHP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = q_sol[Kinematics.GEN_POS_ID["P_RHP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = q_sol[Kinematics.GEN_POS_ID["P_LKP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = q_sol[Kinematics.GEN_POS_ID["P_RKP"]] + np.random.random() * delta * 2 - delta

#         q_ik, sol_found = adamKin.solveIK(q_ik, y_sol, stanceFoot)

#         y_ik = adamKin.calcOutputs(q_ik, stanceFoot)

#         print("\tq_sol: ", q_sol)
#         print("\tq_ik : ", q_ik)
#         print("\ty_sol: ", y_sol)
#         print("\ty_ik : ", y_ik)

#         if not sol_found:
#             print("Error in solution")
#             exit(0)

#         mjInt.setState(q_sol, q_zerovel)
#         mjInt.forward()

#         mj_com = mjInt.getCoMPosition()
#         mj_feet = mjInt.getFootPos()
#         mj_stance = mj_feet[int(stanceFoot)]
#         mj_swing = mj_feet[int(not stanceFoot)]
#         y_mj = np.hstack(
#             (q_sol[Kinematics.GEN_POS_ID["R_Y"]], mj_swing - mj_stance, mj_com - mj_stance)
#         )

#         print("\n\tCheck the Forward kinematics in the IK position vs Mujoco")
#         print("\tY (pin): ", y_sol)
#         print("\tY (mj) : ", y_mj) 
#         print("\terror  : ", y_sol - y_mj)

#         if np.linalg.norm(y_sol - y_mj) > max_error:
#             max_error = np.linalg.norm(y_sol - y_mj)
#             max_error_pin_y = y_sol
#             max_error_mjc_y = y_mj
#             max_error_q = q_sol
    
#     print("\n\nMaximum Error between Pinocchio and Mujoco Outputs: ", max_error, "\n\tPin Y: ", max_error_pin_y, "\n\tMjc Y: ", max_error_mjc_y, "\n\tQ: ", max_error_q)


# if __name__ == "__main__":

#     model, _, _ = pin.buildModelsFromUrdf("/home/wcompton/Repos/ADAM-2D/rsc/models/adam_planar.urdf", "/home/wcompton/Repos/ADAM-2D/rsc/models/")

#     data = model.createData()
#     pin.updateFramePlacements(model, data)

#     for ii in range(model.njoints):
#         print(model.names[ii], ": ", data.oMi[ii].translation)


#     zero_config = np.zeros_like(pin.randomConfiguration(model))

#     for jj in range(zero_config.size):
#         print(f"\n\n{jj}\n")
#         zero_config = np.zeros_like(pin.randomConfiguration(model))
#         zero_config[jj] = 0.1
#         # rng_config = pin.randomConfiguration(model)
#         # print(rng_config)

#         pin.forwardKinematics(model, data, zero_config)
#         pin.updateFramePlacements(model, data)

#         for ii in range(model.njoints):
#             print(model.names[ii], ": ", data.oMi[ii].translation, "\n", data.oMi[ii].rotation, "\n")
