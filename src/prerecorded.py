import numpy as np
import json
import cv2
import regex as re


# Offsets
PATH_PREFIX = "C:/Users/aliha/Alikhive/programming/AI/traffic_jam_cv/"
VIDEO_DIRECTORY = "../videos/"
FILENAME = PATH_PREFIX + "videos/data_video"
DATAPATH = PATH_PREFIX + "debug/"
camera_type = "front"

with open(PATH_PREFIX + "debug/calib_config.json", "r") as file:
    calib = json.load(file)

with open(PATH_PREFIX + "debug/config.json", "r") as file:
    config = json.load(file)

car_height = calib["car_height"]
pitch0 = calib["pitch0"]
roll0 = calib["roll0"]


for camera in config["cameras"]:
    if camera["name"] == camera_type:
        fx = camera["radar_config"]["fx"]
        fy = camera["radar_config"]["fy"]
        cx = camera["radar_config"]["cx"]
        cy = camera["radar_config"]["cy"]
        Tx = camera["radar_config"]["Tx"]
        Ty = camera["radar_config"]["Ty"]
        Tz = camera["radar_config"]["Tz"]


def radar2pixel(xr, yr, zr, pitch, roll):

    xc = np.cos(roll0) * xr + Tx
    yc = np.cos(pitch0) * np.sin(roll0) * xr - np.sin(pitch0) * zr + car_height
    zc = np.sin(pitch0) * np.sin(roll0) * xr + np.cos(pitch0) * zr + Tz

    u = round(xc / zc * fx + cx)
    v = round(yc / zc * fy + cy)
    return u, v


def offset(xr, zr):
    zc = np.sin(pitch0) * np.sin(roll0) * xr + np.cos(pitch0) * zr + Tz
    u = round(0.85 / zc * fx)
    v = round(1 / zc * fy)
    return u, v


cap = cv2.VideoCapture(PATH_PREFIX + "videos/data_video.avi")

#             # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
#             # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
#             # cv2.resizeWindow('frame', 640, 480)
if not cap.isOpened():
    print("Не удалось открыть видео")
else:
    # Получить количество кадров
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Количество фреймов в видео:", total_frames)
    if total_frames == 0:
        total_frames = 5000

# 241 frames in cut 1.0
# 221 frames in cut 2.0
# /// frames in cut 3.0
# 241 frames in cut 4.0
# 281 frames in cut 5.0
# 2301 frames in cut 6.0
# 1401 frames in cut 7.0
# 1361 frames in cut 8.0
# 401 frames in cut 9.0


radar_file = PATH_PREFIX + "debug/radar.json"
gps_file = PATH_PREFIX + "debug/gps_radar.json"
cv2.namedWindow("Traffic Jam Detector 1.0", cv2.WINDOW_AUTOSIZE)

with open(radar_file, "r") as file:
    data1 = file.read()

radar_data = []
pattern = r"(\{(?:[^{}]|(?1))*\})"
radar_json_objects = re.findall(pattern, data1)
for obj_str in radar_json_objects:
    try:
        radar = json.loads(obj_str)
        # print(radar)
        if "Objects" in radar and radar["Objects"]:
            radar_data.append(radar)
            # print("radar_data: ", radar_data)
    except json.JSONDecodeError as e:
        # print("ERROR in json decoding: ", e)
        continue

gps_data = []
with open(gps_file, "r") as file:
    gps = file.read()
for obj in gps.split("}{"):
    if not obj.startswith("{"):
        obj = "{" + obj
    if not obj.endswith("}"):
        obj += "}"
    try:
        parsed = json.loads(obj)
        gps_data.append(parsed)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON object: {e}")
s = 0
frameNum = 1
max_frameNum = 401  # 31112
currentFrameId = 0
while total_frames > frameNum:
    ret, frame = cap.read()
    if int(gps_data[currentFrameId]["currentFrameId"]) == frameNum:
        currentFrameId += 1
        V_gps = gps["speed"]
        gps_timestamp = float(gps_data[currentFrameId]["Timestamp"])
        for radar in radar_data:
            objects = radar["Objects"]
            radar_timestamp = float(radar["Timestamp"])
            if abs(gps_timestamp - float(radar["Timestamp"])) < 0.03:
                text_video = str(gps_timestamp - float(radar["Timestamp"]))
                cv2.putText(
                    frame,
                    text_video,
                    (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                for object in objects:
                    Xr = object["Distance_Lat"]
                    Yr = 0
                    Zr = object["Distance_Long"]
                    RCS = object["RCS"]
                    id = object["ID"]
                    V_long = object["Long_Velocity"]
                    V_lat = object["Lat_Velocity"]
                    V_abs = np.sqrt((V_gps + V_long) ** 2 + V_lat**2)

                    if V_abs > 2:
                        u, v = radar2pixel(Xr, Yr, Zr, pitch0, roll0)
                        # print(u, v)
                        u_offset, v_offset = offset(Xr, Zr)
                        frame = cv2.circle(
                            frame, (u, v), radius=10, color=(0, 0, 255), thickness=-1
                        )
                        # text = str(V_abs)[:4] + ";" + str(Zr)[:4] + ";" + str(id)[:3]
                        text = str(V_abs)[:4]
                        frame = cv2.putText(
                            frame,
                            text,
                            (u + 10, v + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            thickness=2,
                        )
                        print(text)
                        """
                        if Xr > 3:
                            cv2.rectangle(frame, (max(u, 0), max(v - v_offset, 0)),
                                        (min(u + 2 * u_offset, 1920), min(v, 1080)), (0, 255, 0), 2)
                        elif Xr < -2.8:
                            cv2.rectangle(frame, (max(u - 2 * u_offset, 0), max(v - v_offset, 0)),
                                        (min(u, 1920), min(v, 1080)), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (max(u - u_offset, 0), max(v - v_offset, 0)),
                                        (min(u + u_offset, 1920), min(v, 1080)), (0, 255, 0), 2)
                        """
                break

    cv2.imshow("w", frame)
    k = cv2.waitKey(30)

    if k == ord("q"):
        break

    if k == ord(" "):
        if s == 18:
            s = 0
        elif s == 0:
            s = 18
        cv2.waitKey(s)
    # if k == ord('b'):
    #     cur_frame = cap.get(1)
    #     prev_frame = cur_frame
    #     if cur_frame > 2:
    #         prev_frame -= 2

    #     cap.set(1, prev_frame)
    #     i -= 2
    # if k == 81:
    #     cur_frame = cap.get(1)
    #     prev_frame = cur_frame
    #     if cur_frame > 41:
    #         prev_frame -= 41

    #     cap.set(1, prev_frame)
    #     i -= 41
    # if k == ord('f'):
    #     cur_frame = cap.get(1)
    #     next_frame = cur_frame
    #     if cur_frame < 31113 - 5:
    #         next_frame += 5

    #     cap.set(1, next_frame)
    #     i += 5
    # print(i)
    # print("text: ", text, "FrameId: ", frameNum)

    frameNum += 1
cap.release()
