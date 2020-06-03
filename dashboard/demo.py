"""
Create simple dashboard for pedestrian analytic application

How to use:
    Run this application (demo.py)
    Open web browser and type "localhost:4040" at address bar to start application
    Enter video name (has performed pedestrian analytic) to get information of the video
        video information
        pedestrian information
        pedestrian heat map of the video

Note: postgresql version > 9.5 is required to run this application

"""


import os
from flask import Flask,  request, render_template, send_from_directory, abort
from database import Database


# Set root dir
APP_ROOT = os.path.abspath(__file__)
project_directory = os.path.dirname(APP_ROOT)
parent_directory = os.path.dirname(project_directory)

# create heatmap directory
heatmap_dir = os.path.join(parent_directory, "heatmap")
os.makedirs(heatmap_dir, exist_ok=True)

app = Flask(__name__)
db = Database()


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/show_analytic", methods=["POST"])
def show_analytic():

    # get video name
    video_name = request.form["vname"]

    # read log data from "log" table
    log_data = db.read("log", "video_name", video_name)
    if len(log_data) == 0:
        error_code = "Log data"
        return render_template("error.html", video_name=video_name, error_code=error_code)

    # read heatmap table
    heatmap_data = db.read("heatmap", "video_name", video_name)
    if len(heatmap_data) == 0:
        error_code = "Heat map data"
        return render_template("error.html", video_name=video_name, error_code=error_code)
    heatmap_file = "heatmap_" + "_".join(heatmap_data[0][0].split(".")) + "." + heatmap_data[0][1]
    heatmap_path = os.path.join(heatmap_dir, heatmap_file)
    open(heatmap_path, "wb").write(heatmap_data[0][2])

    # read data from analysis table
    pedestrian_info = db.read("_".join(video_name.split(".")))
    if len(pedestrian_info) == 0:
        error_code = "Analysis data"
        return render_template("error.html", video_name=video_name, error_code=error_code)
    num_pedestrian = len(pedestrian_info)

    return render_template("result.html", log_data=log_data, heatmap_file=heatmap_file, pedestrian_info=pedestrian_info,
                           num_pedestrian=num_pedestrian)


@app.route("/show_image/<filename>")
def show_image(filename):
    try:
        return send_from_directory(heatmap_dir, filename=filename)
    except FileNotFoundError:
        abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040)
