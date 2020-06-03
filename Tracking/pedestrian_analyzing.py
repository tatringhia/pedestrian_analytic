"""
Perform pedestrian activity analytic

tracking data format:
    video_name frame_index timestamp person_id bb_x bb_y bb_w bb_h
    (bb_x bb_y bb_w bb_h are top left width high of detected bounding box)

"""


import os
import numpy as np


def activity_analyzing(in_tracking_path):
    """
    Analyze pedestrian's activity upon tracking data

    parameters:
    -----------
        tracking path: path to tracking data

    Return:
    -------
        num_person: Number of pedestrians in the video
        person_info: pedestrian's activity and time
            person_id: assigned pedestrian's ID in video
            appearance_time: time pedestrian appeared in video
            time_in_status: activity and time of pedestrian
                (status: appear, standing, slow walking, walking, running)

    """

    status = ["appear", "standing", "slow walking", "walking", "running"]

    # define iou_threshold to classify activity of pedestrian
    # iou >= 0.97: status ~ "standing"
    # iou >= 0.9 and iou < 0.97: status ~ "slow walking"
    # iou >= 0.3 and iou < 0.9: status ~ "walking"
    # iou > 0 and iou < 0.3: status ~ "running"
    # iou = 0: status ~ "appear"
    iou_threshold = [0, 0.3, 0.9, 0.97]

    # get video name
    tracking_file = os.path.basename(in_tracking_path)
    video_name = tracking_file.split(".")[0][9:]

    print(video_name)

    # read tracking file to numpy array
    tracking_data = np.genfromtxt(in_tracking_path, delimiter=" ")
    # remove column 0 (video_name)
    data = tracking_data[:, 1:]

    # get number of person in video
    p_id = np.unique(data[:, 2])
    num_person = len(p_id)

    # Analyzing
    person_info = []
    for p in p_id:
        # get only data of person has id == p
        p_data = data[data[:, 2] == p]
        # insert 2 empty columns for iou and status of person
        p_data = np.column_stack((p_data, np.zeros((p_data.shape[0], 2))))
        # get number of frames person appeared
        p_num_frames = p_data.shape[0]

        # Calculate appearance time of person p
        p_appearance_time = 0
        for i in range(p_num_frames - 1):
            if p_data[i + 1][0] == (p_data[i][0] + 1):
                p_appearance_time += p_data[i + 1][1] - p_data[i][1]
        p_appearance_time = round(p_appearance_time, 3)

        # calculate iou of bounding box
        for i in range(p_num_frames - 1):
            if p_data[i + 1][0] == (p_data[i][0] + 1):
                bbox1 = p_data[i][3:7]
                bbox2 = p_data[i+1][3:7]
                p_data[i][7] = calculate_iou(bbox1, bbox2)

        # assign status of person
        for i in range(p_num_frames):
            if p_data[i][7] >= iou_threshold[3]:
                p_data[i][8] = 1    # person is standing
            elif p_data[i][7] >= iou_threshold[2]:
                p_data[i][8] = 2    # person is slow walking
            elif p_data[i][7] > iou_threshold[1]:
                p_data[i][8] = 3    # person is walking
            elif p_data[i][7] > iou_threshold[1]:
                p_data[i][8] = 4  # person is running
            else:
                if i <= p_num_frames - 1:
                    if p_data[i][0] == p_data[i-1][0] + 1:
                        p_data[i][8] = p_data[i-1][8]
                else:
                    p_data[i][8] = 0

        # calculate time of each status
        status_time = [0., 0., 0., 0., 0.]
        for i in range(p_num_frames - 1):
            if p_data[i + 1][0] == (p_data[i][0] + 1):
                if p_data[i+1][8] == p_data[i][8]:
                    status_code = int(p_data[i][8])
                    status_time[status_code] += p_data[i+1][1] - p_data[i][1]
                else:
                    status_code = int(p_data[i+1][8])
                    status_time[status_code] += p_data[i+1][1] - p_data[i][1]

        status_time = [round(x, 3) for x in status_time]

        # get main activity
        main_activity = status[status_time.index(max(status_time))]

        p_time_in_status = list(zip(status, status_time))

        p_info = {"person_id": int(p),
                  "appearance_time": p_appearance_time,
                  "main_activity": main_activity,
                  "time_in_status": p_time_in_status}

        person_info.append(p_info)

    return num_person, person_info


def calculate_iou(bbox1, bbox2):
    """
    Calculate iou between 2 bounding boxes

    parameters:
    ----------
        bbox1, bbox2: numpy array, in tlwh format

    return:
    -------
        iou

    """
    # calculate tl, br, wh of intersection
    tl = np.array([np.maximum(bbox1[0], bbox2[0]), np.maximum(bbox1[1], bbox2[1])])
    br = np.array([np.minimum(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]), np.minimum(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])])
    wh = np.maximum(0, br - tl)

    # calculate areas
    intersection_area = np.prod(wh)
    area_bbox1 = np.prod(bbox1[2:])
    area_bbox2 = np.prod(bbox2[2:])

    # calculate iou
    iou = intersection_area / (area_bbox1 + area_bbox2 - intersection_area)

    return iou
