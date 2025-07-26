# Traffic Jam Detector

## How to use:

1) Clone this repository using:

```
git clone https://github.com/pikachoof/traffic_jam_detector.git
```

2) Put your video in the 'videos/' folder

3) Copy your video's name and put it in src/main.py in this line:

```
FILENAME = PATH_PREFIX + "videos/**your_video_name**"
```

NOTE: Don't put file extensions (like .avi, .mp4, e.t.c.) after your_video_name

4) Replace your gps and radar data with the 'gps_front.json' and 'radar.json' files respectively.\
The gps and radar .json files need to contain frame by frame data of the main car's and surrounding car's speeds, otherwise the program won't work.

5) Run the program using:

```
.venv/Scripts/activate

python src/main.py
```
