import cv2

def start_camera(process_frame_callback):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # Skip if frame is not valid
        matches = process_frame_callback(frame)
        for match in matches:
            cv2.putText(frame, f"Match: {match[1]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Missing Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 