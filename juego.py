import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import random
import time
import pyttsx3

# Constants
THRESHOLD = 10
THRESHOLD_RESTART = 50
PIEDRA = np.array([False, False, False, False, False])
PAPEL = np.array([True, True, True, True, True])
TIJERAS = np.array([False, True, True, False, False])
WIN_GAME = ["02", "10", "21"]

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize text-to-speech engine
engine = pyttsx3.init()

def palm_centroid(coordinates_list):
    """Calculate the centroid of given coordinates."""
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    return int(centroid[0]), int(centroid[1])

def calculate_thumb_angle(thumb_coords):
    """Calculate if the thumb is pointing up based on angle."""
    p1, p2, p3 = map(np.array, thumb_coords)
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    to_angle = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
    angle = degrees(acos(min(max(to_angle, -1), 1)))
    return angle > 150

def fingers_up_down(hand_landmarks, thumb_points, palm_points, fingertips_points, finger_base_points, width, height):
    """Determine which fingers are up or down."""
    coordinates_thumb = []
    coordinates_palm = []
    coordinates_ft = []
    coordinates_fb = []

    for index in thumb_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_thumb.append([x, y])

    for index in palm_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_palm.append([x, y])

    for index in fingertips_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_ft.append([x, y])

    for index in finger_base_points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinates_fb.append([x, y])

    # Thumb
    thumb_finger = calculate_thumb_angle(coordinates_thumb)
    
    # Other fingers
    nx, ny = palm_centroid(coordinates_palm)
    coordinates_centroid = np.array([nx, ny])
    coordinates_ft = np.array(coordinates_ft)
    coordinates_fb = np.array(coordinates_fb)
    
    d_centroid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
    d_centroid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
    fingers = d_centroid_ft > d_centroid_fb
    fingers = np.insert(fingers, 0, thumb_finger)

    return fingers

class RockPaperScissorsGame:
    def __init__(self):
        self.pc_option = False
        self.detect_hand = True
        self.count_like = 0
        self.count_piedra = 0
        self.count_papel = 0
        self.count_tijeras = 0
        self.count_restart = 0
        self.player = None
        self.pc = None
        self.image_map = {
            'start': cv2.imread("1.jpg"),
            'choose': cv2.imread("2.jpg"),
            'win': cv2.imread("3.jpg"),
            'tie': cv2.imread("4.jpg"),
            'lose': cv2.imread("5.jpg")
        }
        self.imAux = self.image_map['start']
        self.start_time = None  # To track when the game was activated
        self.said_result = False  # To prevent repeated voice announcements
        self.restart_time = None  # Time to trigger game reset

    def speak(self, message):
        """Function to trigger audio feedback."""
        engine.say(message)
        engine.runAndWait()

    def update_game_state(self, fingers):
        if self.detect_hand:
            if np.array_equal(fingers, TO_ACTIVATE) and not self.pc_option:
                if self.count_like >= THRESHOLD:
                    self.pc = random.randint(0, 2)
                    print("pc:", self.pc)
                    self.pc_option = True
                    self.imAux = self.image_map['choose']
                    self.start_time = time.time()  # Record start time
                self.count_like += 1
            
            if self.pc_option:
                # Check if 3 seconds have passed since activation
                if self.start_time and (time.time() - self.start_time) < 3:
                    # Show a countdown or waiting message
                    self.imAux = self.image_map['choose']
                    return

                if np.array_equal(fingers, PIEDRA):
                    if self.count_piedra >= THRESHOLD:
                        self.player = 0
                    self.count_piedra += 1
                elif np.array_equal(fingers, PAPEL):
                    if self.count_papel >= THRESHOLD:
                        self.player = 1
                    self.count_papel += 1
                elif np.array_equal(fingers, TIJERAS):
                    if self.count_tijeras >= THRESHOLD:
                        self.player = 2
                    self.count_tijeras += 1
        
        if self.player is not None and not self.said_result:  # Only say result once
            self.detect_hand = False
            if self.pc == self.player:
                self.imAux = self.image_map['tie']
                self.speak("Es un empate")
            else:
                if f"{self.player}{self.pc}" in WIN_GAME:
                    self.imAux = self.image_map['win']
                    self.speak("Felicidades, ganaste")
                else:
                    self.imAux = self.image_map['lose']
                    if self.pc == 0:
                        self.speak("Te gané con piedra")
                    elif self.pc == 1:
                        self.speak("Te gané con papel")
                    elif self.pc == 2:
                        self.speak("Te gané con tijeras")
            self.said_result = True  # Ensure voice announcement only happens once
            self.restart_time = time.time()  # Set the time to reset

        # Wait for 3 seconds after the result is said to reset the game
        if self.said_result and (time.time() - self.restart_time) > 3:
            self.reset_game()

    def reset_game(self):
        self.pc_option = False
        self.detect_hand = True
        self.player = None
        self.pc = None
        self.count_like = 0
        self.count_piedra = 0
        self.count_papel = 0
        self.count_tijeras = 0
        self.count_restart = 0
        self.imAux = self.image_map['start']
        self.start_time = None  # Reset the start time
        self.said_result = False  # Allow new voice announcement in the next game
        self.restart_time = None  # Reset restart timer

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    game = RockPaperScissorsGame()

    # Create a custom drawing style for green color
    green_drawing_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    fingers = fingers_up_down(hand_landmarks, thumb_points, palm_points, fingertips_points, finger_base_points, width, height)
                    game.update_game_state(fingers)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=green_drawing_style,  # Green color for landmarks
                        connection_drawing_spec=green_drawing_style)  # Green color for connections
            
            # Create a full screen window
            cv2.namedWindow("n_image", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("n_image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            n_image = cv2.hconcat([game.imAux, frame])
            cv2.imshow("n_image", n_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    thumb_points = [1, 2, 4]
    palm_points = [0, 1, 2, 5, 9, 13, 17]
    fingertips_points = [8, 12, 16, 20]
    finger_base_points = [6, 10, 14, 18]
    TO_ACTIVATE = np.array([True, False, False, False, False])
    
    main()
