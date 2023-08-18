import yaml
import numpy as np
from HLIP.utils.logger import Logger
from HLIP.control_py.hlip_controller import HLIPControllerPD_GC
from HLIP.simulation_py.mujoco_interface import MujocoInterface
from SetRepresentations.set_representation import Ellipsoid, Polytope, Cross, InftyNorm, Points
from TrackingInvariants.tracking_invariant import TrackingInvariant, s2sDynamics


mesh_path = "HLIP/rsc/models/"
urdf_path = "HLIP/rsc/models/adam2d.urdf"
xml_path = "HLIP/rsc/models/adam2d.xml"
err_log_path = "log/log_tracking_invariant_error.csv"
set_log_path = "log/log_tracking_invariant_set.csv"

def main():
    with open('TrackingInvariants/track_inv_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    pitch_ref = config["pitch_ref"]

    z_ref = config["z_ref"]
    T_SSP = config["T_SSP"]
    vis = config["vis"]
    romA = np.array(config["romA"])
    romb = np.array(config["romb"])
    ulim = np.array(config["u_lim"])
    errEps = config["errorEps0"]

    pts = np.array(config["operatingPts"])

    # romSet = Cross(
    #     InftyNorm(lb=np.array([ulim[0]]), ub=np.array([ulim[1]])),
    #     Polytope(A=romA, b=romb)
    # )
    romSet = Points(pts)
    errorSet = Ellipsoid(A=np.eye(10) / errEps, c=np.zeros((10,)))

    mjInt = MujocoInterface(xml_path, vis_enabled=vis)
    hlip = HLIPControllerPD_GC(
        T_SSP, z_ref, urdf_path, mesh_path, mjInt.mass, pitch_ref=pitch_ref
    )

    errorLogger = Logger(err_log_path,
        "iter,IKerror,ep0,esx0,esz0,ecx0,ecz0,epd0,esxd0,eszd0,ecxd0,eczd0,um1,z00,z10,u1,epF,esxF,eszF,ecxF,eczF,epdF,esxdF,eszdF,ecxdF,eczdF\n"
    )
    setLogger = Logger(set_log_path,
        "iter,A00,A10,A20,A30,A40,A50,A60,A70,A80,A90,A01,A11,A21,A31,A41,A51,A61,A71,A81,A91,A02,A12,A22,A32,A42,A52,A62,A72,A82,A92,A03,A13,A23,A33,A43,A53,A63,A73,A83,A93,A04,A14,A24,A34,A44,A54,A64,A74,A84,A94,A05,A15,A25,A35,A45,A55,A65,A75,A85,A95,A06,A16,A26,A36,A46,A56,A66,A76,A86,A96,A07,A17,A27,A37,A47,A57,A67,A77,A87,A97,A08,A18,A28,A38,A48,A58,A68,A78,A88,A98,A09,A19,A29,A39,A49,A59,A69,A79,A89,A99,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n"
    )

    tracking_inv = TrackingInvariant(
        errorSet, romSet, lambda x0, rom0: s2sDynamics(mjInt, hlip, x0, rom0, vis=vis), errorLogger=errorLogger, setLogger=setLogger
    )

    sampleSizes = [20, 100, 1000]

    for N in sampleSizes:
        tracking_inv.iterateSetMap(N, verbose=True)
        while tracking_inv.getProportionInSet() < 1:
            tracking_inv.iterateSetMap(N, verbose=True)
    

if __name__ == "__main__":
    main()
