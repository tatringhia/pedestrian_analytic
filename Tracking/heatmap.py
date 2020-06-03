import os
import sys
import numpy as np
import cv2
import copy


def create_heatmap(in_video, out_dir):
    """
    Create motion heat map of a video

    Parameters:
    ----------
        video_path: path to video file
        heatmap_dir: directory to store heat map file of video

    Return:
    -------
        heatmap_file: heat map file of input video

    """

    threshold = 2
    max_value = 2

    video_name = os.path.basename(in_video)
    heatmap_file = "heatmap_" + "_".join(video_name.split(".")) + ".jpg"
    heatmap_path = os.path.join(out_dir, heatmap_file)

    if not os.path.isfile(in_video):
        print("Video path error")
        sys.exit()

    # create video_capture object
    video_capture = cv2.VideoCapture(in_video)

    # Create background_subtractor object
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    # get first frame and create accum_image
    ret, frame = video_capture.read()
    first_frame = copy.deepcopy(frame)
    height, width = frame.shape[:2]
    accum_image = np.zeros((height, width), np.uint8)

    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1, length):

        ret, frame = video_capture.read()

        # remove the background from frame
        obj_frame = background_subtractor.apply(frame)

        # apply binary threshold to the frame
        ret, frame_obj = cv2.threshold(obj_frame, threshold, max_value, cv2.THRESH_BINARY)

        # add to the accumulated image
        accum_image = cv2.add(accum_image, frame_obj)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # apply ColorMap to accum_image and create overlay picture
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final heat map picture
    cv2.imwrite(heatmap_path, result_overlay)

    # cleanup
    video_capture.release()
    cv2.destroyAllWindows()

    return heatmap_path
