### Pedestrian analytic
This is a demo for my ML/AI capstone project. It is able to detect and track pedestrians in video, analyze some basic activities, and generate motion heatmap. There are 2 parts:

   * pedestrian analytic detects, tracks and analyzing pedestrians in uploaded video
   * dashboard summarizes analytic information and show motion heatmap

**How to use:**

* Download [detection model](https://drive.google.com/file/d/1a-F8CpPmf6e5Pr3hDVWhV3KhfynphQiI/view?usp=sharing) and [feature extraction model](https://drive.google.com/file/d/1a-F8CpPmf6e5Pr3hDVWhV3KhfynphQiI/view?usp=sharing) and save to directory /tracking/model_data
    
    * Build docker images (only in first time)
        > docker-compose Build

    * Run docker container
        > docker-compose up

    * Run application: open web browser and go to
        > "localhost:5000" for uploading video and perform pedestrian analyzing
        > "localhost:4040" for viewing the dashboard analyzed video

    * Stop docker container
        > docker-compose down
