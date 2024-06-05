import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import mediapipe as mp
import numpy as np
import math

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to capture image from webcam
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame from webcam.")
            return None

        # Display the frame in a window
        cv2.imshow("Press 'c' to capture the image", frame)

        # Wait for the user to press 'c' to capture the image or 'q' to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            break
        elif key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the window

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(rgb_frame)
    return image, rgb_frame

# Calculate angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return angle + 360 if angle < 0 else angle

# Recognize hand gestures
def recognize_hand_gestures(image):
    gestures = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        # Process the image and find hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks for each finger
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                # Calculate angles to determine if fingers are folded
                thumb_angle = calculate_angle(thumb_cmc, thumb_mcp, thumb_tip)
                index_angle = calculate_angle(index_mcp, index_pip, index_tip)
                middle_angle = calculate_angle(middle_mcp, middle_pip, middle_tip)
                ring_angle = calculate_angle(ring_mcp, ring_pip, ring_tip)
                pinky_angle = calculate_angle(pinky_mcp, pinky_pip, pinky_tip)

                thumb_is_up = thumb_angle < 30
                index_is_up = index_angle < 30
                middle_is_up = middle_angle < 30
                ring_is_up = ring_angle < 30
                pinky_is_up = pinky_angle < 30

                if thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
                    gestures.append("thumbs up")
                elif not thumb_is_up and index_is_up and middle_is_up and not ring_is_up and not pinky_is_up:
                    gestures.append("peace")
                elif not thumb_is_up and not index_is_up and not middle_is_up and not ring_is_up and not pinky_is_up:
                    gestures.append("closed fist")
                elif thumb_is_up and index_is_up and middle_is_up and ring_is_up and pinky_is_up:
                    gestures.append("open palm")
                else:
                    gestures.append("other gesture")

    return gestures

# Capture an image from the webcam
raw_image, rgb_image = capture_image_from_webcam()
if raw_image is None:
    raise RuntimeError("Failed to capture image from webcam.")

# Recognize hand gestures and show the captured image with landmarks
hand_gestures = recognize_hand_gestures(rgb_image)

# Convert the numpy array back to PIL Image for display
image_with_landmarks = Image.fromarray(rgb_image)
image_with_landmarks.show()

# Display recognized gestures
print(f"Recognized Hand Gestures: {hand_gestures}")

# Prompt the user to enter a question about the image
question = input("Please enter your question about the image: ")

# Process the inputs
inputs = processor(raw_image, question, return_tensors="pt")

# Generate the answer
out = model.generate(**inputs)

# Decode and print the answer
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")
