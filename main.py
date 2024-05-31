from pynwb import NWBHDF5IO
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns

class params:
    frate_thresh = 2
    snr_thresh = 3
    bad_units_list = None
    binwidth = 10
    FN_metric = 'fMI'
    mua_to_fix = []
    fps = 150
    dpi = 300

def compute_derivatives(marker_pos=None, marker_vel=None, smooth=True):
    if marker_pos is not None and marker_vel is None:
        marker_vel = np.diff(marker_pos, axis=-1) * params.fps
        if smooth:
            for dim in range(3):
                marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)

    marker_acc = np.diff(marker_vel, axis=-1) * params.fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)

    return marker_vel, marker_acc

def examine_single_reach_kinematic_distributions(reaches, plot=False):
    pos_color = 'black'
    linewidth = 1

    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer

    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
        shoulder_label = 'shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
        shoulder_label = 'l-shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
        shoulder_label = 'r-shoulder'

    kin_df = pd.DataFrame()
    for start, stop in zip(range(0, 91, 10), range(10, 102, 10)):
        # fig0 = plt.figure(figsize = plot_params.traj_pos_sample_figsize, dpi=plot_params.dpi)
        # ax0 = plt.axes(projection='3d')
        for reachNum, reach in reaches.iloc[start:stop, :].iterrows():
            # get event data using container and ndx_pose names from segment_info table following form below:
            # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
            event_data = kin_module.data_interfaces[reach.video_event]

            wrist_kinematics = event_data.pose_estimation_series[wrist_label].data[reach.start_idx:reach.stop_idx + 1].T
            shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[
                                  reach.start_idx:reach.stop_idx + 1].T
            timestamps = event_data.pose_estimation_series[wrist_label].timestamps[reach.start_idx:reach.stop_idx + 1]

            pos = wrist_kinematics - shoulder_kinematics
            print(np.array(pos).shape)





            vel, tmp_acc = compute_derivatives(marker_pos=pos, marker_vel=None, smooth=True)

            tmp_df = pd.DataFrame(data=zip(np.sqrt(np.square(vel).sum(axis=0)),
                                           vel[0],
                                           vel[1],
                                           vel[2],
                                           pos[0, :-1],
                                           pos[1, :-1],
                                           pos[2, :-1],
                                           np.repeat(reachNum, vel.shape[-1]), ),
                                  columns=['speed', 'vx', 'vy', 'vz', 'x', 'y', 'z', 'reach', ])
            kin_df = pd.concat((kin_df, tmp_df))

        #     ax0.scatter(wrist_kinematics[0], wrist_kinematics[1], wrist_kinematics[2],
        #                   marker='.', s=plot_params.spksamp_markersize/50)

        #     for ax in [ax0]:
        #         ax.set_xlabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_ylabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_zlabel('', fontsize = plot_params.axis_fontsize)
        #         ax.set_xlim(apparatus_min_dims[0], apparatus_max_dims[0]),
        #         ax.set_ylim(apparatus_min_dims[1], apparatus_max_dims[1])
        #         ax.set_zlim(apparatus_min_dims[2], apparatus_max_dims[2])
        #         ax.set_xticks([apparatus_min_dims[0], apparatus_max_dims[0]], labels=['','']),
        #         ax.set_yticks([apparatus_min_dims[1], apparatus_max_dims[1]], labels=['',''])
        #         ax.set_zticks([apparatus_min_dims[2], apparatus_max_dims[2]], labels=['',''])
        #         ax.set_title(f'Reach = {reachNum}')

        #         ax.view_init(view_angle[0], view_angle[1])

        # plt.show()

    kin_df = kin_df.loc[~np.isnan(kin_df['speed'])]
    for key in kin_df.columns:
        if key == 'reach':
            continue

        fig, ax = plt.subplots(figsize=(8, 4), dpi=params.dpi)
        sns.kdeplot(data=kin_df, ax=ax, x=key, hue='reach', legend=False, linewidth=1, common_norm=False, bw_adjust=0.5)
        ax.set_title(key)
        if key in ['vx', 'vy', 'vz', ]:
            ax.set_xlim(-75, 75)
        elif key in ['x', 'y', 'z', ]:
            ax.set_xlim(-10, 10)
        elif key == 'speed':
            ax.set_xlim(0, 100)
        plt.show()

    for vel in ['vx', 'vy', 'vz']:
        kin_df[f'{vel}_mag'] = np.abs(kin_df[vel])
    g = sns.PairGrid(kin_df.loc[:, ['vx_mag', 'vy_mag', 'vz_mag']])
    g.map_upper(sns.scatterplot, s=2)
    g.map_lower(sns.scatterplot, s=2)
    g.map_diag(sns.kdeplot, lw=2)
    plt.show()



def process_kinematic_data(reaches, plot=False):

    kinematic_data = np.zeros((100, 3, 180))

    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer

    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
        shoulder_label = 'shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
        shoulder_label = 'l-shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
        shoulder_label = 'r-shoulder'

    for start, stop in zip(range(0, 91, 10), range(10, 102, 10)):

        for reachNum, reach in reaches.iloc[start:stop, :].iterrows():

            event_data = kin_module.data_interfaces[reach.video_event]

            wrist_kinematics = event_data.pose_estimation_series[wrist_label].data[reach.start_idx:reach.stop_idx + 1].T

            shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx + 1].T

            timestamps = event_data.pose_estimation_series[wrist_label].timestamps[reach.start_idx:reach.stop_idx + 1]

            pos = wrist_kinematics - shoulder_kinematics

            #kinematic_data.append(np.array(pos))
            kinematic_data[reachNum, :, :] = pos[:, 0:180]

            print('-------------------')
            print(np.array(pos).shape[1])
            print(np.array(kinematic_data).shape)
            print('-------------------')

    return kinematic_data

nwb_infile = '/home/christopher/Desktop/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'

with NWBHDF5IO(nwb_infile, 'r') as io_in:
    nwb = io_in.read()

    reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]

    print(reaches_key)

    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False)

    print(reaches.keys())

    kinematic_data = process_kinematic_data(reaches, plot=False)



