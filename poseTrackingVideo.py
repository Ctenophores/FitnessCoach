import json
import os
import cv2
import mediapipe as mp

# Helper function to process each frame
def process_frame(frame, holistic, hands, mpDraw):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_result = holistic.process(imgRGB)
    hand_result = hands.process(imgRGB)

    frame_data = {"pose": [], "hands": []}

    # Extract and draw pose landmarks
    if holistic_result.pose_landmarks:
        for lm in holistic_result.pose_landmarks.landmark:
            frame_data["pose"].append([lm.x, lm.y, lm.z, lm.visibility])
        mpDraw.draw_landmarks(frame, holistic_result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

    # Extract and draw hand landmarks
    if hand_result.multi_hand_landmarks:
        for handLms in hand_result.multi_hand_landmarks:
            hand_points = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
            frame_data["hands"].append(hand_points)
            mpDraw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

    return frame, frame_data

def video_to_skeleton(video_dir='videos', output_dir='video_skeleton_data', output_video_dir='skeleton_videos'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)

    mpHands = mp.solutions.hands
    mpHolistic = mp.solutions.holistic
    hands = mpHands.Hands()
    holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(output_video_dir, f"{os.path.splitext(video_name)[0]}_skeleton.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        data = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, frame_data = process_frame(frame, holistic, hands, mpDraw)
            frame_data["frame"] = frame_count
            data.append(frame_data)

            out.write(processed_frame)
            frame_count += 1

        cap.release()
        out.release()

        skeleton_data_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}.json")
        with open(skeleton_data_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Skeleton data for {video_name} saved as: {skeleton_data_path}")
        print(f"Output skeleton video saved as: {output_video_path}")

def capture_realtime_skeleton():
    cap = cv2.VideoCapture(0)  # Open default camera

    mpHands = mp.solutions.hands
    mpHolistic = mp.solutions.holistic
    hands = mpHands.Hands()
    holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    data = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame.")
                break

            frame, frame_data = process_frame(frame, holistic, hands, mpDraw)
            frame_data["frame"] = frame_count
            data.append(frame_data)
            frame_count += 1

            cv2.imshow('Real-Time Skeleton', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Save the captured data
        output_dir = 'realtime_skeleton_data'
        os.makedirs(output_dir, exist_ok=True)
        skeleton_data_path = os.path.join(output_dir, "skeleton_data.json")
        with open(skeleton_data_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Real-time skeleton data saved as: {skeleton_data_path}")

if __name__ == "__main__":
    isCap = False
    if isCap:
        capture_realtime_skeleton()
    else:
        video_to_skeleton()
        