import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import requests  # For IP-based geolocation

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer

st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# STUN Server
# STUN_STRING = "stun:" + str(getSTUNServer())
# STUN_SERVER = [{"urls": [STUN_STRING]}]

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


st.title("Road Damage Detection - Realtime")

st.write(
    "Detect the road damage in realtime using USB Webcam. This can be useful for on-site monitoring with personel on the ground. Select the video input device and start the inference."
)

# JavaScript to update geolocation every 5 seconds
geolocation_js = """
    <script>
    function updateLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(showPosition, showError);
        } else { 
            alert("Geolocation is not supported by this browser.");
        }
    }

    function showPosition(position) {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        document.getElementById("geo-latitude").innerHTML = latitude.toFixed(6);
        document.getElementById("geo-longitude").innerHTML = longitude.toFixed(6);
        window.localStorage.setItem("latitude", latitude.toFixed(6));
        window.localStorage.setItem("longitude", longitude.toFixed(6));
    }

    function showError(error) {
        switch(error.code) {
            case error.PERMISSION_DENIED:
                alert("User denied the request for Geolocation.");
                break;
            case error.POSITION_UNAVAILABLE:
                alert("Location information is unavailable.");
                break;
            case error.TIMEOUT:
                alert("The request to get user location timed out.");
                break;
            case error.UNKNOWN_ERROR:
                alert("An unknown error occurred.");
                break;
        }
    }

    // Update location every 2 seconds
    setInterval(updateLocation, 2000);
    updateLocation();  // Initial call to get location immediately
    </script>
"""

# Display placeholders for geolocation
st.markdown("### Real-Time Geolocation")
latitude_placeholder = st.empty()
longitude_placeholder = st.empty()

# Display the JavaScript to the frontend
components.html(
    f"""
    {geolocation_js}
    <div>
        <p>Latitude: <span id="geo-latitude"></span></p>
        <p>Longitude: <span id="geo-longitude"></span></p>
    </div>
    """,
    height=100,
)

# Use Streamlit's placeholders to update the values dynamically
latitude = st.session_state.get("latitude", "Not Available")
longitude = st.session_state.get("longitude", "Not Available")

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Get geolocation from local storage (set by the JavaScript)
    latitude = st.session_state.get("latitude", "Not Available")
    longitude = st.session_state.get("longitude", "Not Available")

    # Update placeholders with the current geolocation
    latitude_placeholder.write(f"Latitude: {latitude}")
    longitude_placeholder.write(f"Longitude: {longitude}")

    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    # Save the results on the queue
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    return av.VideoFrame.from_ndarray(_image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    # rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

st.divider()

if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)