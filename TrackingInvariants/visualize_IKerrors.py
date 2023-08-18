import cv2
import yaml
import numpy as np
import pandas as pd
from TrackingInvariants.tracking_invariant import s2sDynamics
from HLIP.control_py.hlip_controller import HLIPControllerPD_GC
from HLIP.simulation_py.mujoco_interface import MujocoInterface


mesh_path = "HLIP/rsc/models/"
urdf_path = "HLIP/rsc/models/adam2d.urdf"
xml_path = "HLIP/rsc/models/adam2d.xml"
err_log_path = "log/log_tracking_invariant_error.csv"
set_log_path = "log/log_tracking_invariant_set.csv"
video_path = "log/videos"

def main():
    with open('TrackingInvariants/track_inv_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    error_log = pd.read_csv(err_log_path)
    error_log = error_log.loc[error_log["IKerror"] == 1, :]
    error_log = error_log.reset_index(drop=True)

    pitch_ref = config["pitch_ref"]

    z_ref = config["z_ref"]
    T_SSP = config["T_SSP"]

    mjInt = MujocoInterface(xml_path, vis_enabled=True)
    hlip = HLIPControllerPD_GC(
        T_SSP, z_ref, urdf_path, mesh_path, mjInt.mass, pitch_ref=pitch_ref
    )

    for ii in error_log.index:
        e0 = error_log.iloc[ii, 2:12].values
        rom0 = error_log.iloc[ii, 12:16].values
        frames = s2sDynamics(mjInt, hlip, e0, rom0, vis=True, video=True)

        out = cv2.VideoWriter(f'{video_path}/IK_error_{ii}_S2S.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, (frames[0].shape[1], frames[0].shape[0]))

        for frame in frames:
            out.write(np.flip(frame, axis=2))
        out.release()



    
    

if __name__ == "__main__":
    main()
