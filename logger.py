import datetime
 
def log_detection(person, frame_path):
    with open('data/detections.log', 'a') as f:
        f.write(f"{datetime.datetime.now()} - Detected: {person[1]}, Frame: {frame_path}\n") 