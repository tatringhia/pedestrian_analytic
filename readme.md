### Pedestrian analytic
This is a demo for my ML/AI capstone project. It is able to detect and track pedestrians in video, analyze some basic activities, and generate motion heatmap. There are 2 parts:

   * pedestrian_analytic detects, tracks and analyzing pedestrians of uploaded video
   * dashboard summarizes analytic information and show motion heatmap
   
 please visit readme.pdf for more details.

**How to use:**

* Download [detection model](https://drive.google.com/file/d/1a-F8CpPmf6e5Pr3hDVWhV3KhfynphQiI/view?usp=sharing) and [feature extraction model](https://drive.google.com/file/d/1AviA59mV9wAAaI4btUL6VoCzFh89GtXS/view?usp=sharing) and save to directory /tracking/model_data
    
* Build docker images (only in first time)
  > docker-compose Build

* Run docker container
  > docker-compose up

* Run application: open web browser and go to
  > `localhost:5000` for uploading video and performing pedestrian analyzing\
  > `localhost:4040` for viewing the dashboard of analyzed video

* Stop docker container
  > docker-compose down
