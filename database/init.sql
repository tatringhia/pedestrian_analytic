CREATE TABLE log (
    video_name TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    date TEXT,
    video_time FLOAT,
    num_frames INT,
    inference_time INT,
    Output_file TEXT,
    detection_model TEXT,
    feat_model TEXT
);

CREATE TABLE heatmap (
    video_name TEXT UNIQUE NOT NULL,
    file_extension TEXT NOT NULL,
    heatmap_binary bytea
);
