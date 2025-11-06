"""
Configuration file for Area Monitoring System
Modify these settings to customize the monitoring behavior
"""
# YOLO settings
IOU_THRESHOLD = 0.3  # NMS IoU threshold

# Auto screenshot settings
AUTO_SCREENSHOT_ON_PERSON = True
SCREENSHOT_COOLDOWN = 5  # seconds

# Camera settings
CAMERA_INDEX = 0  # 0 for default webcam, 1 for external camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection (0.0 to 1.0)
NMS_THRESHOLD = 0.5  # Non-Maximum Suppression threshold

# Zone monitoring settings
ZONE_COLOR = (0, 255, 0)  # Green color for zone outline (BGR format)
ZONE_THICKNESS = 1
ZONE_ALPHA = 0.3  # Transparency for zone fill (0.0 to 1.0)

# Alert settings
ALERT_SOUND_ENABLED = True
ALERT_SOUND_FILE = "alert.wav"  # Path to alert sound file
ALERT_DURATION = 3  # Duration to show alert in seconds
ALERT_COOLDOWN = 5  # Cooldown period between alerts in seconds

# Visual settings
BOUNDING_BOX_COLOR = (0, 0, 255)  # Red color for person bounding box (BGR format)
BOUNDING_BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White color for text
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# Display settings
WINDOW_NAME = " Area Monitoring System"
SHOW_FPS = True
SHOW_DETECTION_COUNT = True

# Zone coordinates (modify these to define your monitoring area)
# Format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] - four corners of the zone
# These are relative coordinates (0.0 to 1.0) that will be scaled to camera resolution
ZONE_COORDINATES = [
    (0.0, 0.0),  # Top-left
    (1.0, 0.0),  # Top-right
    (1.0, 1.0),  # Bottom-right
    (0.0, 1.0)   # Bottom-left
]

