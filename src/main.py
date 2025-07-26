from ultralytics import YOLO
import cv2
import numpy as np
import json
import regex as re

# TODO: ADD PATROL SPEED

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def radar2pixel(xr, yr, zr, pitch, roll):

    xc = np.cos(roll) * xr + Tx
    yc = np.cos(pitch) * np.sin(roll) * xr - np.sin(pitch) * zr + car_height
    zc = np.sin(pitch) * np.sin(roll) * xr + np.cos(pitch) * zr + Tz

    u = round(xc / zc * fx + cx)
    v = round(yc / zc * fy + cy)
    return u, v

def find_gps(frame_number, gps_data):
    for gps in gps_data:
        if gps["cameraFrameId"] == frame_number:
            return gps

    return None

def find_radar(gps_timestamp, radar_data, timestamp_difference_margin): # Time difference margin is in SECONDS
    for radar in radar_data:
        radar_timestamp = radar["Timestamp"]
        timestamp_difference = abs(gps_timestamp - radar_timestamp)
        if timestamp_difference < timestamp_difference_margin:
            return radar, timestamp_difference
    
    return None, 0

def find_last_closest_data_index(xbox, ybox, last_frame_data):
    closest_index = 0
    closest_distance = 1e9
    for i in range(len(last_frame_data)):
        coords = last_frame_data[i]["box"].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        current_distance = distance(xbox, ybox, xm, ym)
        if current_distance < closest_distance:
            closest_distance = current_distance
            closest_index = i

    return closest_index


def find_closest_object(xbox, ybox, objects):
    closest_object = None
    closest_object_distance = 1e9
    for object in objects:
        Xr = object["Distance_Lat"]
        Yr = 0
        Zr = object["Distance_Long"]
        u, v = radar2pixel(Xr, Yr, Zr, pitch0, roll0)
        current_distance = distance(xbox, ybox, u, v)
        if current_distance < closest_object_distance:
            closest_object = object
            closest_object_distance = current_distance

    return closest_object

# SETUP BEGIN #
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

radar_file = PATH_PREFIX + "debug/radar.json"
gps_file = PATH_PREFIX + "debug/gps_radar.json"

saved_radar_data_file = PATH_PREFIX + "src/data/saved_radar_data.json"
saved_gps_data_file = PATH_PREFIX + "src/data/saved_gps_data.json"
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

    #with open(saved_radar_data_file, 'w') as file:
    #    json.dump(radar_data, file, indent=4)

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

    #with open(saved_gps_data_file, 'w') as file:
    #    json.dump(gps_data, file, indent=4)

# SETUP END #

model = YOLO(PATH_PREFIX + "src/best.pt")

class_names = {
    'front': 0,
    'rear': 1,
    'side': 2
}

class_names_reversed = {
    0 : 'front',
    1 : 'rear',
    2 : 'side'
}

# Get the class ID for 'rear'
rear_class_id = class_names.get('rear') # More direct way to get the ID if you control the dict
print(rear_class_id)

if rear_class_id is None:
    print("Error: 'rear' class not found in class_names mapping. Please check your class_names.")
    exit()

video_path = 'videos/data_video.avi' # Replace with the actual path to your input video
output_video_path = 'videos/output_data_video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties for setting up the output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# Use 'mp4v' for .mp4 output. For .avi, you might use 'XVID' or 'MJPG'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_video_path}")

results_generator = model.predict(source=video_path, stream=True, conf=0.6, iou=0.45, show=False, verbose=False)

current_frame_number = 0
total_rear_items_across_video = 0

gps_data_list = list(gps_data)

currentFrameId = 0

last_frame_data = [] 
last_frame_filled = False

last_gps_speed = 0
last_timestamp_difference = 0

timestamp_difference = 0

jam_speed_threshold = 5
jam_car_threshold = 5

for r in results_generator:
    current_frame_number += 1
    current_frame_rear_count = 0

    ret, frame = cap.read()
    current_frame = r.orig_img.copy()
    boxes = r.boxes
    
    gps = find_gps(current_frame_number, gps_data)

    if gps:
        radar, timestamp_difference = find_radar(gps["Timestamp"], radar_data, 0.025)
        last_timestamp_difference = timestamp_difference

    if radar:
        objects = radar["Objects"]

    rear_boxes = [box for box in boxes if box.cls[0] == rear_class_id]
    rear_boxes_count = len(rear_boxes)

    current_frame_rear_count += rear_boxes_count
    total_rear_items_across_video += rear_boxes_count

    for box in rear_boxes:
        class_id = int(box.cls[0])  # Get class ID
        confidence = float(box.conf[0]) # Get confidence score
    
        print(class_id, confidence)
        
        # Ensure coordinates are integers for drawing
        bbox_coords = box.xyxy[0].cpu().numpy().astype(int)

        # 1) Include & only draw boxes around the items with class "rear"
        x1, y1, x2, y2 = bbox_coords
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        # Draw the bounding box
        color = (0, 255, 0) # Green color (BGR format)
        thickness = 2
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness)
        
        closest_object = None
        closest_last_data_index = find_last_closest_data_index(xm, ym, last_frame_data)
        print(closest_last_data_index)
        if objects is None:
            if last_frame_filled:
                closest_object = last_frame_data[closest_last_data_index]["object"]
                last_frame_data[closest_last_data_index]["box"] = box
            else:
                last_frame_data.append({"box": box, "object": None}) # Worst case
        else:
            closest_object = find_closest_object(xm, ym, objects)
            if last_frame_filled:
               last_frame_data[closest_last_data_index]["box"] = box
               last_frame_data[closest_last_data_index]["object"] = closest_object
            else:
                last_frame_data.append({"box": box, "object": closest_object})

        if gps:
            V_gps = gps["speed"]
            last_gps_speed = V_gps
        else:
            V_gps = last_gps_speed
        V_closest_lat = closest_object["Lat_Velocity"]
        V_closest_long = closest_object["Long_Velocity"]
        V_closest_total = V_gps + np.sqrt(V_closest_long ** 2 + V_closest_lat ** 2)

        label = f"{class_names_reversed[class_id]} Confidence:{confidence:.2f} Speed:{V_closest_total:.4f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        text_x = x1
        text_y = y1 - 10 
        if text_y < text_size[1]:
            text_y = y1 + text_size[1] + 5

        cv2.rectangle(current_frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1) 
        cv2.putText(current_frame, label, (text_x + 5, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    if not last_frame_filled and len(rear_boxes) != 0:
        last_frame_filled = True

    if not timestamp_difference:
        timestamp_difference = last_timestamp_difference

    speed_threshold_count = 0
    for data in last_frame_data:
        car = data["object"]
        V_lat = car["Lat_Velocity"] * 3.6
        V_long = car["Long_Velocity"] * 3.6
        V_total = V_gps * 3.6 + np.sqrt(V_long ** 2 + V_lat ** 2)

        if V_total <= jam_speed_threshold:
            speed_threshold_count += 1

    if speed_threshold_count >= jam_car_threshold:
        cv2.putText(current_frame, f"PROBKA", (int(frame_width / 2), 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(current_frame, f"Time Difference: {timestamp_difference:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(current_frame, f"Rear Objects: {current_frame_rear_count}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    V_gps_kph = V_gps * 3.6
    cv2.putText(current_frame, f"Patrol Speed: {V_gps_kph:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Rear Detections', current_frame)

    # Write the frame to the output video file
    out.write(current_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nFinished processing. Total 'rear' items detected across all frames: {total_rear_items_across_video}")
print(f"Processed {current_frame_number} frames.")

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()