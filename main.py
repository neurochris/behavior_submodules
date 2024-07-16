import math

from pynwb import NWBHDF5IO
from hatlab_nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, SpectralClustering


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
    kin_df = pd.DataFrame()

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

            #pos = wrist_kinematics - shoulder_kinematics
            pos = wrist_kinematics

            #kinematic_data.append(np.array(pos))

            #print('-------------------')
            #print(kin_df.to_string())
            #print('-------------------')

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


    return kin_df





##check transition period as well - maybe thats the third cluster?




def get_labels_y(data):
    unique_reaches = np.unique(data["reach"].to_numpy())
    num_of_reaches = len(unique_reaches)
    print(num_of_reaches)
    labels = []

    for reach_num in unique_reaches:
        print('------------------------------------------')
        x = data[data.reach == reach_num].x
        y = data[data.reach == reach_num].y
        trans_counter = 0
        increasing = False
        decreasing = False

        for idx, y_coordinate in enumerate(y):

            if idx == 1:
                if y_coordinate > prev_y:
                    #print('increasing')
                    increasing = True
                elif y_coordinate < prev_y:
                    #print('decreasing')
                    decreasing = True


            if idx == 0:
                prev_y = y_coordinate
                labels.append(2)
            else:
                if y_coordinate > prev_y:
                    #print('increasing')
                    labels.append(0)
                    if decreasing:
                        decreasing = False
                elif y_coordinate == prev_y:
                    #print('constant')
                    labels.append(2)
                elif y_coordinate < prev_y:
                    #print('decreasing')
                    labels.append(1)

                prev_y = y_coordinate

    return labels

nwb_infile = '/home/christopher/Desktop/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'

with NWBHDF5IO(nwb_infile, 'r') as io_in:
    nwb = io_in.read()

    reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]

    print(reaches_key)

    units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False)

    print(reaches.keys())

    kin_df = process_kinematic_data(reaches, plot=False)

    reach_idx = 5

    indv_reach = kin_df[kin_df.reach == reach_idx]
    cols = 'rgbcmy'


    print(indv_reach.x)
    print(indv_reach.y)
    print(indv_reach.z)

    print(np.sqrt(((indv_reach.vx*indv_reach.vx) + (indv_reach.vy*indv_reach.vy))))

    plt.plot(range(0, len(indv_reach.vx)), np.sqrt((indv_reach.vx*indv_reach.vx) + (indv_reach.vy*indv_reach.vy)))
    plt.title("Speed profile for reach " + str(reach_idx))
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.show()

    plt.plot(indv_reach.x, indv_reach.y)
    plt.title("X and y pos for reach " + str(reach_idx))
    plt.xlabel("X pos")
    plt.ylabel("Y pos")
    plt.show()

    i = 0

    fig, ax = plt.subplots()

    for i in range(len(indv_reach.x)):
        if len(indv_reach.x[i:i + 2]) == 2:
            if i != 1:
                ax.plot(indv_reach.vx[i:i + 2], indv_reach.vy[i:i + 2])
            else:
                ax.plot(indv_reach.vx[i:i + 2], indv_reach.vy[i:i + 2])

    plt.title("X and y pos for reach " + str(reach_idx))
    plt.xlabel("X pos")
    plt.ylabel("Y pos")
    plt.show()

    i = 0

    fig, ax = plt.subplots()

    for i in range(len(indv_reach.x)):
        if len(indv_reach.vx[i:i+2]) == 2:
            if i != 1:
                ax.plot(range(i, i+2), np.sqrt(indv_reach.vx[i:i + 2]*indv_reach.vx[i:i + 2] + indv_reach.vy[i:i + 2]*indv_reach.vy[i:i + 2]))
            else:
                ax.plot(range(i, i+2), np.sqrt(indv_reach.vx[i:i + 2]*indv_reach.vx[i:i + 2] + indv_reach.vy[i:i + 2]*indv_reach.vy[i:i + 2]))

    plt.title("Speed for reach " + str(reach_idx))
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.show()

    speed = np.sqrt(indv_reach.vx*indv_reach.vx + indv_reach.vy*indv_reach.vy)

    smooth = gaussian_filter1d(speed, 1)

    # compute second derivative
    smooth_d2 = np.gradient(smooth)

    stationary_points = np.where(np.diff(np.sign(smooth_d2)))[0]
    print(stationary_points)
    # plot results
    #plt.plot(speed, label='Noisy Data')
    #plt.plot(smooth, label='Smoothed Data')
    #plt.plot(np.max(smooth) * (smooth_d2) / (np.max(smooth_d2) - np.min(smooth_d2)), label='First Derivative (scaled)')

    prev_idx = 0
    for i, idx in enumerate(stationary_points, 1):
        plt.plot(speed[prev_idx:idx], color=cols[i % 6])
        prev_idx = idx


    for i, spt in enumerate(stationary_points, 1):
        plt.axvline(x=spt, color='k', label=f'Inflection Point {i}')


    #plt.legend()
    plt.title("Stationary points in speed profile for reach " + str(reach_idx))
    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.show()





    i = 0

    fig, ax = plt.subplots()
    prev_idx = 0

    for i, idx in enumerate(stationary_points, 1):
        ax.plot(indv_reach.vx[prev_idx:idx], indv_reach.vy[prev_idx:idx], color=cols[i % 6])
        prev_idx = idx

    plt.title("Segmented x and y pos for reach " + str(reach_idx))
    plt.xlabel("X pos")
    plt.ylabel("Y pos")
    plt.show()




    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(indv_reach.x, indv_reach.y,indv_reach.z)

    N = len(indv_reach.z)
    cols = 'rgbcmy'

    for i in range(N):
        if i != 1:
            ax.plot(indv_reach.vx[i:i + 2], indv_reach.vy[i:i + 2], color=cols[i % 6])
        else:
            ax.plot(indv_reach.vx[i:i + 2], indv_reach.vy[i:i + 2], color=cols[i % 6])

    ax.set_xlabel("vx (side to side)")
    ax.set_ylabel("vy (front and back)")
    ##ax.set_zlabel("vz (up and down)")

    plt.show()

    print(indv_reach)


    kin_df = kin_df[kin_df.reach == reach_idx]
    indv_reach = kin_df.dropna()
    print(kin_df)

    labels = get_labels_y(kin_df)

    plt.scatter(indv_reach.vx, indv_reach.vy, c=labels)  # without scaling
    plt.show()

    print(labels)

 

    print("computing PCA")

    indv_reach = indv_reach[['vx', 'vy']]

    pca = KernelPCA(n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1)
    pca.fit(indv_reach)
    x_new = pca.transform(indv_reach)

    score = x_new[:, 0:2]
    print(indv_reach.shape)
    print(score.shape)

    xs = score[:, 0]
    ys = score[:, 1]



    plt.scatter(xs, ys, c=labels)  # without scaling
    plt.show()


    kmeans = KMeans(n_clusters=2)

    label = kmeans.fit_predict(score)

    print(label)
    print(label.shape)


    for lab in np.unique(label):
        print(lab)
        filtered_label = score[label == lab]

        plt.scatter(filtered_label[:, 0], filtered_label[:, 1], color=cols[lab % 6])

    plt.show()

    print(indv_reach)
    '''



