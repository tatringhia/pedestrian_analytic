"""
Create REST API for video tracking using Flask

How to use:
    Run this application (demo.py)
    Open web browser and type "localhost:5000" at address bar to start application

Note: postgresql version > 9.5 is required to run this application

"""

import os
import configparser
from flask import Flask,  request, render_template, send_from_directory, session, abort
from keras import backend as K
from tracking import video_tracking
from pedestrian_analyzing import activity_analyzing
from heatmap import create_heatmap
from database import Database
import psycopg2

# Set root dir
APP_ROOT = os.path.abspath(__file__)
project_directory = os.path.dirname(APP_ROOT)
parent_directory = os.path.dirname(project_directory)

# create upload directory
upload_dir = os.path.join(parent_directory, "upload")
os.makedirs(upload_dir, exist_ok=True)

# create download directory
download_dir = os.path.join(parent_directory, "download")
os.makedirs(download_dir, exist_ok=True)

# create log directory
log_dir = os.path.join(parent_directory, "log")
os.makedirs(log_dir, exist_ok=True)

# create tracking_data directory
td_dir = os.path.join(parent_directory, "tracking_data")
os.makedirs(td_dir, exist_ok=True)

# create heatmap directory
heatmap_dir = os.path.join(parent_directory, "overlay")
os.makedirs(heatmap_dir, exist_ok=True)

app = Flask(__name__)
app.secret_key = "Mysecretkey@Flash"

db = Database()


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # initialize session["upload_video"]
    session["upload_video"] = []

    # get files from submitted form
    for f in request.files.getlist("file"):
        video_path = os.path.join(upload_dir, f.filename)
        f.save(video_path)
        # append video_path to session["upload_video"]
        session["upload_video"].append(video_path)

    return render_template("upload.html")


@app.route("/analyzing")
def analyzing():
    # get upload video
    in_vlist = session.get("upload_video")

    # Perform video tracking and write information to database
    # clear the graph
    K.clear_session()
    # perform video_tracking
    det_list, logfile, tracking_list = video_tracking(in_vlist, download_dir, td_dir, log_dir)
    # assign det_list to out_vlist for downloading
    out_vlist = [os.path.basename(f) for f in det_list]
    # write log data to database
    log = configparser.ConfigParser()
    log.read(logfile)
    log_section = log.sections()
    for section in log_section:
        log_data = dict(log.items(section))
        db.upsert_data(table_name="log", **log_data)

    # Save video's heat_map to database
    for in_video in in_vlist:
        # create heat map for the video
        heatmap_path = create_heatmap(in_video, heatmap_dir)
        # read heatmap file as binary
        heatmap_binary = open(heatmap_path, "rb").read()
        # prepare data to write to overlay table
        heatmap_data = {"video_name": os.path.basename(in_video),
                        "file_extension": "jpg",
                        "heatmap_binary": psycopg2.Binary(heatmap_binary)}
        # write heat map data to database
        db.upsert_data("heatmap", **heatmap_data)

    # save pedestrian' activity data to database
    for video_path, tracking_path in zip(in_vlist, tracking_list):
        num_pedestrians, pedestrian_info = activity_analyzing(tracking_path)
        video_name = "_".join(os.path.basename(video_path).split("."))
        db.create_table(table_name=video_name)
        for person in pedestrian_info:
            print(person)
            db.upsert_data(table_name=video_name, condition="person_id", **person)

    return render_template("result.html", list=out_vlist)


@app.route("/file-return/<filename>")
def file_return(filename):
    try:
        return send_from_directory(download_dir, filename=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
