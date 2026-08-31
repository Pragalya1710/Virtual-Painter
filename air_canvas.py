import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# VIRTUAL PAINTER - FINGER TRACKING
# ============================================================

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Try to set a good resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------
# Canvas
# -----------------------------
canvas = None

# Previous finger position
prev_x = 0
prev_y = 0

# -----------------------------
# Drawing settings
# -----------------------------
draw_color = (255, 0, 0)       # Blue
brush_size = 8

# Eraser
eraser_mode = False

# Current color name
current_color = "BLUE"


# ============================================================
# Function: Check which fingers are up
# ============================================================

def get_finger_status(hand_landmarks):

    landmarks = hand_landmarks.landmark

    # Index finger
    index_up = landmarks[8].y < landmarks[6].y

    # Middle finger
    middle_up = landmarks[12].y < landmarks[10].y

    # Ring finger
    ring_up = landmarks[16].y < landmarks[14].y

    # Pinky finger
    pinky_up = landmarks[20].y < landmarks[18].y

    return index_up, middle_up, ring_up, pinky_up


# ============================================================
# Function: Draw toolbar
# ============================================================

def draw_toolbar(frame):

    # Toolbar background
    cv2.rectangle(
        frame,
        (0, 0),
        (900, 90),
        (40, 40, 40),
        -1
    )

    # Title
    cv2.putText(
        frame,
        "VIRTUAL PAINTER",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Color boxes
    colors = [
        ("B", (255, 0, 0)),
        ("G", (0, 255, 0)),
        ("R", (0, 0, 255)),
        ("Y", (0, 255, 255))
    ]

    x = 15

    for name, color in colors:

        cv2.rectangle(
            frame,
            (x, 45),
            (x + 40, 80),
            color,
            -1
        )

        cv2.putText(
            frame,
            name,
            (x + 12, 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2
        )

        x += 50

    # Eraser
    cv2.rectangle(
        frame,
        (225, 45),
        (305, 80),
        (200, 200, 200),
        -1
    )

    cv2.putText(
        frame,
        "ERASER",
        (233, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1
    )

    # Clear
    cv2.rectangle(
        frame,
        (315, 45),
        (380, 80),
        (100, 100, 100),
        -1
    )

    cv2.putText(
        frame,
        "CLEAR",
        (323, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    # Save
    cv2.rectangle(
        frame,
        (390, 45),
        (455, 80),
        (100, 100, 100),
        -1
    )

    cv2.putText(
        frame,
        "SAVE",
        (401, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    # Brush size
    cv2.putText(
        frame,
        f"Brush: {brush_size}",
        (475, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Current mode
    if eraser_mode:

        cv2.putText(
            frame,
            "Mode: ERASER",
            (650, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Color: {current_color}",
            (650, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2
        )


# ============================================================
# Main program
# ============================================================

while True:

    # Read webcam
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not access the webcam.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Create canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process hand
    results = hands.process(rgb_frame)

    # --------------------------------------------------------
    # If a hand is detected
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Check fingers
        index_up, middle_up, ring_up, pinky_up = get_finger_status(
            hand_landmarks
        )

        # ----------------------------------------------------
        # Get index finger position
        # ----------------------------------------------------

        index_x = int(
            hand_landmarks.landmark[8].x * frame.shape[1]
        )

        index_y = int(
            hand_landmarks.landmark[8].y * frame.shape[0]
        )

        # Keep point below toolbar
        if index_y < 90:
            index_y = 90

        # ----------------------------------------------------
        # DRAWING MODE
        #
        # Index finger UP
        # Middle finger DOWN
        # ----------------------------------------------------

        if index_up and not middle_up:

            # Green tracking circle
            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )

            # First point
            if prev_x == 0 and prev_y == 0:

                prev_x = index_x
                prev_y = index_y

            # -----------------------------------------------
            # Eraser
            # -----------------------------------------------

            if eraser_mode:

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (index_x, index_y),
                    (0, 0, 0),
                    brush_size * 5,
                    cv2.LINE_AA
                )

                cv2.circle(
                    canvas,
                    (index_x, index_y),
                    brush_size * 2,
                    (0, 0, 0),
                    -1
                )

            # -----------------------------------------------
            # Normal drawing
            # -----------------------------------------------

            else:

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (index_x, index_y),
                    draw_color,
                    brush_size,
                    cv2.LINE_AA
                )

            # Update previous position
            prev_x = index_x
            prev_y = index_y

        # ----------------------------------------------------
        # SELECTION / STOP DRAWING MODE
        #
        # Index + Middle finger UP
        # ----------------------------------------------------

        elif index_up and middle_up:

            # Reset drawing position
            prev_x = 0
            prev_y = 0

            # Show selection circle
            cv2.circle(
                frame,
                (index_x, index_y),
                15,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Any other hand position
        # ----------------------------------------------------

        else:

            prev_x = 0
            prev_y = 0

    else:

        # No hand detected
        prev_x = 0
        prev_y = 0

    # ========================================================
    # Combine canvas with webcam
    # ========================================================

    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, canvas_mask = cv2.threshold(
        gray_canvas,
        10,
        255,
        cv2.THRESH_BINARY
    )

    canvas_mask_inv = cv2.bitwise_not(
        canvas_mask
    )

    # Webcam background
    frame_background = cv2.bitwise_and(
        frame,
        frame,
        mask=canvas_mask_inv
    )

    # Drawing
    canvas_foreground = cv2.bitwise_and(
        canvas,
        canvas,
        mask=canvas_mask
    )

    # Combine
    output = cv2.add(
        frame_background,
        canvas_foreground
    )

    # ========================================================
    # Toolbar
    # ========================================================

    draw_toolbar(output)

    # ========================================================
    # Instructions
    # ========================================================

    cv2.putText(
        output,
        "INDEX FINGER = DRAW",
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        "INDEX + MIDDLE = STOP",
        (15, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        "B=Blue  G=Green  R=Red  Y=Yellow  E=Eraser",
        (15, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        "C=Clear  S=Save  +/-=Brush Size  Q=Quit",
        (15, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # ========================================================
    # Show window
    # ========================================================

    cv2.imshow(
        "Virtual Painter",
        output
    )

    # ========================================================
    # Keyboard controls
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    elif key == ord('b'):

        draw_color = (255, 0, 0)
        current_color = "BLUE"
        eraser_mode = False

    elif key == ord('g'):

        draw_color = (0, 255, 0)
        current_color = "GREEN"
        eraser_mode = False

    elif key == ord('r'):

        draw_color = (0, 0, 255)
        current_color = "RED"
        eraser_mode = False

    elif key == ord('y'):

        draw_color = (0, 255, 255)
        current_color = "YELLOW"
        eraser_mode = False

    # --------------------------------------------------------
    # Eraser
    # --------------------------------------------------------

    elif key == ord('e'):

        eraser_mode = True

    # --------------------------------------------------------
    # Clear
    # --------------------------------------------------------

    elif key == ord('c'):

        canvas = np.zeros_like(frame)

        prev_x = 0
        prev_y = 0

        print("Canvas cleared.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    elif key == ord('s'):

        cv2.imwrite(
            "drawing.png",
            canvas
        )

        print("Drawing saved as drawing.png")

    # --------------------------------------------------------
    # Increase brush size
    # --------------------------------------------------------

    elif key == ord('+') or key == ord('='):

        brush_size += 2

        if brush_size > 50:
            brush_size = 50

        print("Brush size:", brush_size)

    # --------------------------------------------------------
    # Decrease brush size
    # --------------------------------------------------------

    elif key == ord('-'):

        brush_size -= 2

        if brush_size < 2:
            brush_size = 2

        print("Brush size:", brush_size)


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Virtual Painter closed.")