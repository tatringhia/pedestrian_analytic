"""
Perform tracking pedestrian in video.

tracking application (tracking.py) can be used as a standalone application

How to use as standalone application:
    Create input directory, copy video files to this directory
    Run tracking.py file
    Result files
        output files (filename: "tracking_(video_name).zip") are in output directory
        log file is in log directory
        tracking data files (filename: "tracking_video_name.txt") in tracking_data directory

    Note: Arguments can be added as option, running tracking.py --help for more details

"""

import sys
import os
import posixpath
from timeit import time
from datetime import datetime
import argparse
import configparser
import shutil

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


def get_args():
    # get argument from users
    parser = argparse.ArgumentParser(description="Tracking argsions")

    # input and output directories
    parser.add_argument("--input", help="Input Video directory")
    parser.add_argument("--output", help="Output video directory ")
    parser.add_argument("--tracking_data", help="tracking data directory for further analysis")
    parser.add_argument("--log", help="Log directory")

    # YOLO initial parameters
    parser.add_argument("--yolo_model", help="yolo model path")
    parser.add_argument("--yolo_anchor", help="yolo anchor path")
    parser.add_argument("--yolo_classes", help="yolo classes path")
    parser.add_argument("--yolo_score", help="yolo object threshold")
    parser.add_argument("--yolo_iou", help="yolo non-max suppression threshold")

    # DeepSort initial parameters
    parser.add_argument("--feature_extraction", help="Feature extraction model path")
    parser.add_argument("--nms_max_overlap", help="nms_max_overlap threshold for detected bounding_box")
    parser.add_argument("--metric_criteria", help="cosine or euclidian")
    parser.add_argument("--max_distance", help="max_distance of the metric")
    parser.add_argument("--nn_budget", help="number of samples saved for id tracking")
    parser.add_argument("--max_iou", help="max_iou_distance for tracking")
    parser.add_argument("--max_age", help="Maximum number of frame to be tracked")
    parser.add_argument("--n_init", help="Minimum number of detection before assigning an id")

    # flags
    parser.add_argument("--no_log", action="store_true", default=False, help="log file")
    parser.add_argument("--no_writevideo", action="store_true", default=False, help="Write analyzed video")
    parser.add_argument("--show", action="store_true", default=False, help="Show the video during inference time")
    parser.add_argument("--from_beginning", action="store_true", default=False, help="Start from beginning")

    args = parser.parse_args()

    return args


def main():
    # get user's input arguments
    args = get_args()

    # read configuration file
    config_file = "./config/tracking.cfg"
    config = configparser.ConfigParser()
    config.read(config_file)

    # input and output directories
    if not args.input:
        args.input = config.get("directories", "input")
    if not args.output:
        args.output = config.get("directories", "output")
    if not args.tracking_data:
        args.tracking_data = config.get("directories", "tracking_data")
    if not args.log:
        args.log = config.get("directories", "log")

    # get flags from args
    no_log_flag = args.no_log
    no_writevideo_flag = args.no_writevideo
    show_flag = args.show
    from_beginning_flag = args.from_beginning

    # get input directory
    in_dir = args.input

    # create output directories
    # output directory
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    # tracking_data directory
    td_dir = args.tracking_data
    os.makedirs(td_dir, exist_ok=True)
    # log directory
    log_dir = args.log
    os.makedirs(log_dir, exist_ok=True)

    # get the last log file, logfile is log file name, logpath is full path of log file
    log_list = os.listdir(log_dir)
    if len(log_list) > 0:
        log_list = [log for log in log_list if log.split("_")[0] == "tracking"]
        log_indices = [int(log.split(".")[0].split("_")[2]) for log in log_list]
        last_index = max(log_indices)
        last_logfile = log_list[log_indices.index(last_index)]
    else:
        last_index = 0
        last_logfile = "tracking_log_" + str(last_index).zfill(3) + ".txt"
    # create logpath of last log file
    last_logpath = posixpath.join(log_dir, last_logfile)

    # if not running from beginning, check video files have been performed tracking from last log file
    done_list = []
    if not from_beginning_flag:
        # create done_list from the last log_file
        _log = configparser.ConfigParser(strict=False)
        _log.read(last_logpath)
        video_log = _log.sections()
        done_list = list(set(video_log))

    # create list of video need to perform tracking
    # get list of videos in in_dir
    if os.path.isdir(in_dir):
        in_vlist = os.listdir(in_dir)
    elif os.path.isfile(in_dir):
        in_vlist = os.path.basename(in_dir)
        in_dir = os.path.dirname(in_dir)
    else:
        print("please provide correct input directory or file")
        sys.exit()
    # crete list of new video files to perform detection
    new_vlist = [posixpath.join(in_dir, f) for f in in_vlist if f not in done_list]

    video_tracking(new_vlist, out_dir, td_dir, last_logpath, no_log_flag, show_flag, no_writevideo_flag)


def video_tracking(in_vlist, out_dir, tracking_data_dir, logpath,
                   no_log_flag=False, show_flag=False, no_writevideo_flag=False):
    """
    Perform pedestrian detection and tracking
        detection: using YOLOV3 model, retrain to detect pedestrian only
            (YOLOv3 reference: to https://github.com/qqwweee/keras-yolo3)

        tracking: using DEEPSORT algorithm with feature extraction model based upon RESNET50 trained with DUKE-MTMC
                    dataset. Re_id was added to DEEPSORT algorithm
            (DEEPSORT reference:  https://github.com/nwojke/deep_sort)
            (Featrue extraction referece: https://github.com/layumi/Person_reID_baseline_pytorch)

    Parameters:
    -----------
        in_vlist: input video list to perform tracking
        out_dir: output directory to save output files
        tracking_data_dir: tracking data directory to save tracking data for further analysis (pedestrian_analyzing)
        log_path: path of log file saving inference only use in case of no_log_flag=True
        show_flag: show tracking video during inference
        no_writevideo_flag: not save tracking video

    Return:
    -------
        output_list: list of output files. Each output has 3 files (detection file, tracking file, tracking video)
                        in .zip format
        logfile: log file save inference log
        tracking_file_list: list of tracking file using for pedestrian analyzing

    """

    # read configuration file
    config_file = "./config/tracking.cfg"
    config = configparser.ConfigParser()
    config.read(config_file)

    # get default/initial parameters from config_file
    # get yolo initial parameters
    det_model = config.get("yolo", "model_path")
    anchors_path = config.get("yolo", "anchors_path")
    classes_path = config.get("yolo", "classes_path")
    score = config.getfloat("yolo", "score")
    iou = config.getfloat("yolo", "iou")

    # get feature extraction model
    feat_model = config.get("feature_extraction", "model_path")

    # get initial parameters for distance metrics
    metric_criteria = config.get("distance_metrics", "metric_criteria")
    max_distance = config.getfloat("distance_metrics", "max_distance")
    nn_budget = config.getint("distance_metrics", "nn_budget")

    # get initial parameters for processing
    nms_max_overlap = config.getfloat("processing", "nms_max_overlap")

    # get initial parameters for tracker
    max_iou_distance = config.getfloat("tracker", "max_iou_distance")
    max_age = config.getint("tracker", "max_age")
    n_init = config.getint("tracker", "n_init")

    # create yolo_init dictionary and update from user's input arguments
    yolo_init = {"model_path": det_model, "anchors_path": anchors_path, "classes_path": classes_path,
                 "score": score, "iou": iou}

    # initiate yolo and tracker objects
    yolo = YOLO(**yolo_init)

    encoder = gdet.create_box_encoder(feat_model, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(metric_criteria, max_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance, max_age, n_init)

    # create temporary directory to save results before moving to out_dir
    tmp_dir = posixpath.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # initialize output_list and tracking_data
    output_list = []
    logfile = None
    tracking_file_list = []

    # perform tracking for video in in_vlist
    for in_video in in_vlist:

        # get basename of the video
        basename = os.path.basename(in_video)

        # create output video directory for input video file
        out_vdir = posixpath.join(tmp_dir, basename.split(".")[0])
        os.makedirs(out_vdir, exist_ok=True)

        # create video_capture object of in_video
        video_capture = cv2.VideoCapture(in_video)

        readvideo_flag = video_capture.isOpened()
        if readvideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")

            # create tracking, detection and out_video object
            # out_video file save video performed tracking
            if not no_writevideo_flag:
                out_videofile = "tracking_" + basename
                out_video_path = posixpath.join(out_vdir, out_videofile)
                out_video = cv2.VideoWriter(out_video_path, fourcc, 15, (w, h))
            # tracking file save tracking information
            tracking_file = "tracking_" + basename.split(".")[0] + ".txt"
            tracking_path = posixpath.join(out_vdir, tracking_file)
            tracking = open(tracking_path, "w")
            # detection file save detection information
            detection_file = "detection_" + basename.split(".")[0] + ".txt"
            detection_path = posixpath.join(out_vdir, detection_file)
            detection = open(detection_path, "w")

            # print out basename of video to be tracked
            print("\n" + "tracking for video file {}".format(basename) + "\n")

            frame_index = 0
            video_logtime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            start = time.time()
            frame_timestamp = 0
            while True:

                ret, frame = video_capture.read()
                if not ret:
                    break

                # Create timestamp of video frame
                frame_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # log start time when processing video frame
                t1 = time.time()

                # convert image from BRG to RGB
                image = Image.fromarray(frame[..., ::-1])

                # Run yolo detection
                boxs, scores = yolo.detect_image(image)

                # Extract features of detected boxs
                features = encoder(frame, boxs)

                # Create detection object
                detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxs, scores, features)]

                # Run non-max suppression to delete overlapped detection > nms_max_overlap
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker predict and then update with new detections
                tracker.predict()
                tracker.update(detections)

                # put track_id (in green) and draw rectangle of tracked detection (in white) to video frame
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 150, (0, 255, 0), 2)

                # draw rectangle of detection (in blue) to video frame
                for i in range(len(boxs)):
                    bbox = boxs[i]
                    bbox[2] = bbox[0] + bbox[2]
                    bbox[3] = bbox[1] + bbox[3]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

                # Put frame index to video frame
                cv2.putText(frame, "Frame : {}".format(frame_index), (5, 20), 0, 5e-3 * 120, (0, 255, 0), 2)

                # Show video frame
                if show_flag:
                    cv2.imshow("tracking", frame)

                # Save analyzed frame to out_video
                if not no_writevideo_flag:
                    out_video.write(frame)

                # Save detection and tracking
                # create frame_name
                frame_name = basename.split(".")[0] + " " + str(frame_index).zfill(4)

                # only save frame having detected bounding boxes to detection_log
                if len(boxs) != 0:
                    for i in range(0, len(boxs)):
                        detection.write(frame_name + " " +
                                        "{}".format(boxs[i][0]) + " " + "{}".format(boxs[i][1]) + " " +
                                        "{}".format(boxs[i][2]) + " " + "{}".format(boxs[i][3]) + " " +
                                        "{:.2f}".format(scores[i]) + "\n")

                # only save frame having assigned id to tracking_log
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    tracking.write(frame_name + " " + "{:.3f}".format(frame_timestamp) + " ")
                    tracking.write("{}".format(track.track_id) + " ")
                    bbox = track.to_tlwh()
                    tracking.write("{}".format(int(bbox[0])) + " " + "{}".format(int(bbox[1])) + " " +
                                   '{}'.format(int(bbox[2])) + " " + '{}'.format(int(bbox[3])))
                    tracking.write('\n')

                # print out processed time of a video frame
                process_time = time.time() - t1
                print("frame {} processed in {:.3f}".format(frame_index, process_time))

                # Increase frame_index
                frame_index += 1

                # Press Q to stop!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            detection.close()
            tracking.close()
            if not no_writevideo_flag:
                out_video.release()

            # log the end of analyzed time a video
            end = time.time()

            # print out processing time of in_video
            print("\n" + "file {} is processed in {} seconds".format(basename, int((end - start))))

            # copy tracking_file, detection_file to tracking_data_dir for further analysis
            # shutil.copyfile(detection_path, posixpath.join(tracking_data_dir, detection_file))
            shutil.copyfile(tracking_path, posixpath.join(tracking_data_dir, tracking_file))

            # add tracking_file to tracking_file_list
            tracking_file_list.append(posixpath.join(tracking_data_dir, tracking_file))

            # zip all out_vdir
            zip_name = "tracking_" + basename.split(".")[0]
            arc_format = "zip"
            shutil.make_archive(zip_name, arc_format, out_vdir)
            zip_file = zip_name + "." + arc_format
            shutil.move(zip_file, os.path.join(out_dir, zip_file))

            # add the zip file to output_list
            output_list.append(posixpath.join(out_dir, zip_file))

            # Save log
            if not no_log_flag:
                # open logfile
                if os.path.isfile(logpath):
                    logfile = logpath
                    log = open(logfile, "a")
                else:
                    logfile = posixpath.join(logpath + "log.txt")
                    log = open(logfile, "w")

                # write data to loglfie
                log.write("[{}]".format(basename) + "\n")
                log.write("video_name = {}".format(basename) + "\n")
                log.write("status = {}".format("readable") + "\n")
                log.write("date = {}".format(video_logtime) + "\n")
                if readvideo_flag:
                    log.write("video_time = {:.3f}".format(frame_timestamp) + "\n")
                    log.write("num_frames = {}".format(frame_index) + "\n")
                    log.write("inference_time = {}".format(int(end - start)) + "\n")
                    # if not no_writevideo_flag:
                    #     log.write("tracking_video = {}".format(out_videofile) + "\n")
                    # else:
                    #     log.write("tracking_video = {}".format(None) + "\n")
                    # log.write("tracking_log = {}".format(tracking_file) + "\n")
                    # log.write("detection_log = {}".format(detection_file) + "\n")
                    log.write("output_file = {}".format(zip_file) + "\n")
                    log.write("detection_model = {}".format(os.path.basename(det_model)) + "\n")
                    log.write("feat_model = {}".format(os.path.basename(feat_model)) + "\n")
                    log.write("\n")
                else:
                    log.write("total_frames = {}".format(0) + "\n")
                    log.write("total_time = {:.3f}".format(0) + "\n")
                    log.write("inference_time = {}".format(0) + "\n")
                    # log.write("tracking_video = {}".format(None) + "\n")
                    # log.write("tracking_log = {}".format(None) + "\n")
                    # log.write("detection_log = {}".format(None) + "\n")
                    log.write("output_file = {}".format(None) + "\n")
                    log.write("detection_model = {}".format(None) + "\n")
                    log.write("feat_model = {}".format(None) + "\n")
                    log.write("\n")
                # close logfile
                log.close()

        else:   # if readvideo_flag=false
            print("file {} not able to read")

    # Close all cv2 windows
    cv2.destroyAllWindows()
    # print out when finishing tasks
    print("\n" + "all files done" + "\n")

    # remove tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_list, logfile, tracking_file_list


if __name__ == '__main__':
    main()
