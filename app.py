
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Virtual Painter",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Virtual Painter")
st.write("Raise your index finger and draw in the air! ☝️")


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Purple": (255, 0, 255),
    "White": (255, 255, 255)
}


# ============================================================
# HAND TRACKING
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class VirtualPainter(VideoProcessorBase):

    def __init__(
        self,
        color=(255, 0, 0),
        brush_size=8,
        eraser=False
    ):

        self.color = color
        self.brush_size = brush_size
        self.eraser = eraser

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.canvas = None

        self.prev_x = 0
        self.prev_y = 0

    # --------------------------------------------------------
    # Detect raised fingers
    # --------------------------------------------------------

    def fingers_up(self, hand):

        landmarks = hand.landmark

        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y

        return index_up, middle_up

    # --------------------------------------------------------
    # Process camera frame
    # --------------------------------------------------------

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        # Create canvas
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # MediaPipe needs RGB
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        # ----------------------------------------------------
        # Hand detected
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                img,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            index_up, middle_up = self.fingers_up(hand)

            h, w, _ = img.shape

            # Index fingertip
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # ------------------------------------------------
            # DRAWING MODE
            # Index finger only
            # ------------------------------------------------

            if index_up and not middle_up:

                cv2.circle(
                    img,
                    (x, y),
                    10,
                    (0, 255, 0),
                    -1
                )

                if self.prev_x == 0 and self.prev_y == 0:

                    self.prev_x = x
                    self.prev_y = y

                # Eraser
                if self.eraser:

                    cv2.line(
                        self.canvas,
                        (self.prev_x, self.prev_y),
                        (x, y),
                        (0, 0, 0),
                        self.brush_size * 5,
                        cv2.LINE_AA
                    )

                # Normal drawing
                else:

                    cv2.line(
                        self.canvas,
                        (self.prev_x, self.prev_y),
                        (x, y),
                        self.color,
                        self.brush_size,
                        cv2.LINE_AA
                    )

                self.prev_x = x
                self.prev_y = y

            # ------------------------------------------------
            # STOP DRAWING
            # Index + middle finger
            # ------------------------------------------------

            elif index_up and middle_up:

                self.prev_x = 0
                self.prev_y = 0

                cv2.circle(
                    img,
                    (x, y),
                    15,
                    (0, 255, 255),
                    2
                )

            else:

                self.prev_x = 0
                self.prev_y = 0

        else:

            self.prev_x = 0
            self.prev_y = 0

        # ====================================================
        # COMBINE CAMERA + CANVAS
        # ====================================================

        gray = cv2.cvtColor(
            self.canvas,
            cv2.COLOR_BGR2GRAY
        )

        _, mask = cv2.threshold(
            gray,
            10,
            255,
            cv2.THRESH_BINARY
        )

        mask_inv = cv2.bitwise_not(mask)

        background = cv2.bitwise_and(
            img,
            img,
            mask=mask_inv
        )

        drawing = cv2.bitwise_and(
            self.canvas,
            self.canvas,
            mask=mask
        )

        output = cv2.add(
            background,
            drawing
        )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("🎨 Drawing Controls")

selected_color = st.sidebar.selectbox(
    "Choose Color",
    list(COLORS.keys())
)

brush_size = st.sidebar.slider(
    "🖌️ Brush Size",
    min_value=2,
    max_value=30,
    value=8
)

eraser = st.sidebar.checkbox(
    "🧹 Eraser"
)


# ============================================================
# DISPLAY SELECTED COLOR
# ============================================================

st.sidebar.markdown(
    f"### Selected Color: **{selected_color}**"
)


# ============================================================
# START CAMERA
# ============================================================

st.info(
    "Click START and allow camera access."
)

webrtc_ctx = webrtc_streamer(
    key="virtual-painter",

    video_processor_factory=lambda: VirtualPainter(
        color=COLORS[selected_color],
        brush_size=brush_size,
        eraser=eraser
    ),

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# ============================================================
# UPDATE LIVE CONTROLS
# ============================================================

if webrtc_ctx.video_processor:

    webrtc_ctx.video_processor.color = COLORS[selected_color]

    webrtc_ctx.video_processor.brush_size = brush_size

    webrtc_ctx.video_processor.eraser = eraser


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown("---")

st.subheader("✋ How to Use")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    ### ☝️ Draw

    Raise your **index finger only**
    and move it around.
    """)

with col2:

    st.markdown("""
    ### ✌️ Stop

    Raise **index + middle finger**
    to stop drawing.
    """)

with col3:

    st.markdown("""
    ### 🧹 Eraser

    Turn on **Eraser** from
    the sidebar.
    """)