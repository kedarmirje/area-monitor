"""
Area Monitoring System v2.1
A real-time person detection system with cyberpunk UI
"""

import cv2
import numpy as np
import time
import pygame
import torch
import threading
from datetime import datetime
from ultralytics import YOLO
import config
import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class Alert:
    message: str
    timestamp: float
    alert_type: str  # 'info', 'warning', 'danger'
    

class AreaMonitor:
    def __init__(self):
        """Initialize the Area Monitoring System"""
        # Theme colors (Cyberpunk style)
        self.THEME = {
            'bg': (10, 10, 25),
            'primary': (0, 255, 255),      # Cyan
            'secondary': (255, 0, 255),    # Magenta
            'success': (0, 255, 150),
            'warning': (255, 200, 0),
            'danger': (255, 0, 100),
            'text': (200, 200, 200),
            'text_secondary': (120, 120, 120),
            'card': (20, 25, 45),
            'card_accent': (30, 35, 60),
            'border': (0, 150, 150)
        }
        
        # UI state
        self.show_sidebar = True
        self.fullscreen = True
        self.alerts: List[Alert] = []
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.person_count = 0
        self.last_alert_time = 0
        self.show_zones = True
        
        # Auto-screenshot variables
        self.last_screenshot_time = 0
        self.screenshot_counter = 0
        
        # Initialize pygame for sound and UI
        pygame.init()
        # Load alert sound via absolute path and prepare channel
        self.alert_sound = None
        self.sound_channel = None
        if config.ALERT_SOUND_ENABLED:
            try:
                pygame.mixer.init()
                base_dir = os.path.dirname(os.path.abspath(__file__))
                sound_path = os.path.join(base_dir, config.ALERT_SOUND_FILE)
                if not os.path.isfile(sound_path):
                    raise FileNotFoundError(f"Alert sound not found at {sound_path}")
                self.alert_sound = pygame.mixer.Sound(sound_path)
            except Exception as e:
                print(f"Warning: Could not initialize sound: {e}")
                self.alert_sound = None
        
        # Initialize YOLO model
        print("Loading YOLO model for person detection...")
        self.model = YOLO('yolov8n.pt')  # nano version for better performance
        # Use GPU if available for faster and more accurate inference timings
        try:
            if torch.cuda.is_available():
                self.model.to('cuda')
        except Exception:
            pass
        print("Person detection model loaded successfully!")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {config.CAMERA_INDEX}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        # Get actual camera dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get screen info for fullscreen
        info = pygame.display.Info()
        self.screen_width = info.current_w
        self.screen_height = info.current_h
        
        # Calculate window dimensions
        self.sidebar_width = 400
        self.window_width = self.screen_width
        self.window_height = self.screen_height
        
        # Calculate video scaling to fit screen while maintaining aspect ratio
        available_width = self.window_width - self.sidebar_width
        available_height = self.window_height -100  # Leave space for status bar
        
        scale_w = available_width / self.frame_width
        scale_h = available_height / self.frame_height
        scale = min(scale_w, scale_h)
        
        self.scaled_video_width = int(self.frame_width * scale)
        self.scaled_video_height = int(self.frame_height * scale)
        
        # Center video
        self.video_x = (available_width - self.scaled_video_width) // 2
        self.video_y = 50 + (available_height - self.scaled_video_height) // 2
        
        # Initialize pygame display
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height), 
                                                  pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        pygame.display.set_caption(" Area Monitoring System")
        
        # Initialize fonts
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 18)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Convert zone coordinates to pixel coordinates
        self.zone_points = self._convert_zone_coordinates()
        
        # Alert system variables
        self.last_alert_time = 0
        self.alert_active = False
        self.alert_start_time = 0
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        print(f"Display: {self.window_width}x{self.window_height} (FULLSCREEN)")
        print(" Area Monitoring System ready!")
        print(" Controls:")
        print("   ESC/Q - Quit")
        print("   R - Reset alerts") 
        print("   Z - Toggle zone visibility")
        print("   S - Screenshot")
        print("   F - Toggle sidebar")
    
    def _convert_zone_coordinates(self):
        """Convert relative zone coordinates to pixel coordinates"""
        points = []
        if not hasattr(config, 'ZONE_COORDINATES') or not config.ZONE_COORDINATES:
            # Default to full frame if no zone defined
            return np.array([
                [0, 0],
                [self.frame_width - 1, 0],
                [self.frame_width - 1, self.frame_height - 1],
                [0, self.frame_height - 1]
            ], dtype=np.int32)
        
        for rel_x, rel_y in config.ZONE_COORDINATES:
            pixel_x = int(rel_x * self.frame_width)
            pixel_y = int(rel_y * self.frame_height)
            points.append((pixel_x, pixel_y))
        
        return np.array(points, dtype=np.int32)
    
    def _is_point_in_zone(self, point):
        """Check if a point is inside the monitoring zone"""
        return cv2.pointPolygonTest(self.zone_points, point, False) >= 0
    
    def _is_person_in_zone(self, person_bbox):
        """Check if a person is inside the monitoring zone"""
        x1, y1, x2, y2 = person_bbox
        
        # Check multiple points of the bounding box
        corners = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x1, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
            ((x1 + x2) // 2, (y1 + y2) // 2)  # Center
        ]
        
        # If any corner is in the zone, consider person in zone
        for corner in corners:
            if self._is_point_in_zone(corner):
                return True
        
        return False
    
    def _draw_zone(self, frame):
        """Draw the monitoring zone on the frame"""
        # Create overlay for semi-transparent zone
        overlay = frame.copy()
        
        # Fill zone with semi-transparent color
        cv2.fillPoly(overlay, [self.zone_points], config.ZONE_COLOR)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, config.ZONE_ALPHA, frame, 1 - config.ZONE_ALPHA, 0, frame)
        
        # Draw zone outline
        cv2.polylines(frame, [self.zone_points], True, config.ZONE_COLOR, config.ZONE_THICKNESS)
        
        # Add zone label
        if len(self.zone_points) > 0:
            label_pos = tuple(self.zone_points[0])
            cv2.putText(frame, "MONITORING ZONE", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.ZONE_COLOR, 2)
    
    def add_alert(self, message: str, alert_type: str = 'info'):
        """Add an alert to the system"""
        alert = Alert(
            message=message,
            timestamp=time.time(),
            alert_type=alert_type
        )
        self.alerts.append(alert)
        
        # Keep only last 20 alerts
        if len(self.alerts) > 20:
            self.alerts.pop(0)
        
        print(f"[{alert_type.upper()}] {message}")
    
    def _play_alert_sound(self):
        """Play alert sound"""
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except Exception as e:
                print(f"Warning: Could not play alert sound: {e}")
    
    def _draw_info_panel(self, frame, person_count, fps):
        """Draw information panel on the frame (legacy - not used in fullscreen)"""
        pass
    
    def process_frame(self, frame):
        """Process a single frame for person detection"""
        # Run YOLO object detection with tuned parameters
        results = self.model(
            frame, 
            verbose=False,
            conf=config.CONFIDENCE_THRESHOLD,  # Use config threshold
            iou=config.IOU_THRESHOLD,  # Better overlap handling
            classes=[0],  # Only detect persons (class 0)
            agnostic_nms=True,  # Better NMS
            imgsz=640,
            max_det=10
        )
        
        # Reset person count
        self.person_count = 0
        
        # Process person detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Check if the detected object is a person (class 0 in COCO dataset)
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box with confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence label
                    conf_label = f"Person {confidence:.2f}"
                    cv2.putText(frame, conf_label, (x1, y1 - 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Check if person is in the monitoring zone
                    person_bbox = (x1, y1, x2, y2)
                    if self._is_person_in_zone(person_bbox):
                        self.person_count += 1
                        # Console log to confirm detection events
                        print(f"Detected person in zone: conf={confidence:.2f}, bbox=({x1},{y1},{x2},{y2}))")
                        
                        # Draw a different color for people in the zone
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Add label for person in zone
                        label = f"IN ZONE {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Add alert if needed
                        current_time = time.time()
                        if current_time - self.last_alert_time > config.ALERT_COOLDOWN:
                            self.last_alert_time = current_time
                            self.alert_active = True
                            self.alert_start_time = current_time
                            self.add_alert(f"⚠️ Person detected in zone! (Total: {self.person_count})", 'danger')
                            
                            # Auto-capture screenshot if enabled
                            if config.AUTO_SCREENSHOT_ON_PERSON:
                                self._auto_screenshot(frame.copy(), "person_in_zone")
                            
                            # Play alert sound in a separate thread
                            if self.alert_sound and config.ALERT_SOUND_ENABLED:
                                threading.Thread(target=self._play_alert_sound, daemon=True).start()
        
        # Manage continuous alert sound based on presence
        if self.person_count > 0 and self.alert_sound and config.ALERT_SOUND_ENABLED:
            if self.sound_channel is None or not self.sound_channel.get_busy():
                self.sound_channel = self.alert_sound.play(loops=-1)
        else:
            if self.sound_channel is not None and self.sound_channel.get_busy():
                self.sound_channel.stop()
            if self.person_count == 0:
                self.alert_active = False

        # Draw monitoring zone if enabled
        if self.show_zones:
            self._draw_zone(frame)
        
        # Draw information panel
        self._draw_info_panel(frame, self.person_count, self.fps)
        
        return frame
        
    def _take_screenshot(self, frame, reason="manual"):
        """Take a screenshot of the current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{reason}_{timestamp}_{self.screenshot_counter:03d}.jpg"
        cv2.imwrite(filename, frame)
        self.screenshot_counter += 1
        print(f"📸 Screenshot saved: {filename}")
        return filename
    
    def _auto_screenshot(self, frame, reason):
        """Auto-capture screenshot with cooldown"""
        current_time = time.time()
        if current_time - self.last_screenshot_time > config.SCREENSHOT_COOLDOWN:
            self.last_screenshot_time = current_time
            filename = self._take_screenshot(frame, reason)
            self.add_alert(f"📸 Auto-captured: {reason}", 'info')
            return filename
        return None
    
    def draw_sidebar(self):
        """Draw the cyberpunk-styled sidebar with system information and alerts"""
        sidebar = pygame.Surface((self.sidebar_width, self.window_height), pygame.SRCALPHA)
        
        # Gradient background
        for i in range(self.window_height):
            alpha = 220
            color = (
                self.THEME['card'][0] + int((self.THEME['card_accent'][0] - self.THEME['card'][0]) * i / self.window_height),
                self.THEME['card'][1] + int((self.THEME['card_accent'][1] - self.THEME['card'][1]) * i / self.window_height),
                self.THEME['card'][2] + int((self.THEME['card_accent'][2] - self.THEME['card'][2]) * i / self.window_height),
                alpha
            )
            pygame.draw.line(sidebar, color, (0, i), (self.sidebar_width, i))
        
        # Glowing border
        glow_time = time.time() * 2
        glow_intensity = int(100 + 155 * abs(np.sin(glow_time)))
        glow_color = (0, glow_intensity, glow_intensity)
        pygame.draw.line(sidebar, glow_color, (0, 0), (0, self.window_height), 4)
        
        y_offset = 30
        
        # Animated title with glow effect
        title_text = " AREA MONITOR "
        title = self.font_large.render(title_text, True, self.THEME['primary'])
        title_shadow = self.font_large.render(title_text, True, (0, 100, 100))
        sidebar.blit(title_shadow, (22, y_offset + 2))
        sidebar.blit(title, (20, y_offset))
        y_offset += 60
        
        # Animated subtitle
        subtitle = self.font_small.render("SURVEILLANCE SYSTEM v2.0", True, self.THEME['text_secondary'])
        sidebar.blit(subtitle, (25, y_offset))
        y_offset += 40
        
        # Glowing separator
        pygame.draw.line(sidebar, self.THEME['border'], (20, y_offset), (self.sidebar_width - 20, y_offset), 3)
        y_offset += 30
        
        # System Stats with animated indicators
        stats_title = self.font_medium.render("▸ SYSTEM STATUS", True, self.THEME['primary'])
        sidebar.blit(stats_title, (20, y_offset))
        y_offset += 40
        
        # FPS with bar indicator
        fps_label = self.font_small.render("FPS", True, self.THEME['text'])
        sidebar.blit(fps_label, (30, y_offset))
        fps_value = self.font_medium.render(f"{self.fps:.1f}", True, self.THEME['success'])
        sidebar.blit(fps_value, (self.sidebar_width - 100, y_offset - 5))
        
        # FPS bar
        bar_width = int((self.fps / 60.0) * 200)
        pygame.draw.rect(sidebar, self.THEME['card_accent'], (30, y_offset + 25, 200, 8))
        pygame.draw.rect(sidebar, self.THEME['success'], (30, y_offset + 25, min(bar_width, 200), 8))
        y_offset += 50
        
        # Targets in zone with pulsing effect
        target_label = self.font_small.render("TARGETS IN ZONE", True, self.THEME['text'])
        sidebar.blit(target_label, (30, y_offset))
        y_offset += 30
        
        # Large animated count
        if self.person_count > 0:
            pulse_scale = 1.0 + 0.2 * abs(np.sin(time.time() * 4))
            count_size = int(48 * pulse_scale)
            count_font = pygame.font.SysFont('Consolas', count_size, bold=True)
            count_color = self.THEME['danger']
        else:
            count_font = self.font_large
            count_color = self.THEME['success']
        
        count_text = count_font.render(str(self.person_count), True, count_color)
        count_rect = count_text.get_rect(center=(self.sidebar_width // 2, y_offset + 30))
        sidebar.blit(count_text, count_rect)
        y_offset += 80
        
        # Alert status with animated indicator
        alert_status = " ALERT" if self.alert_active else "SECURE"
        alert_color = self.THEME['danger'] if self.alert_active else self.THEME['success']
        
        if self.alert_active:
            # Pulsing background for active alert
            pulse_alpha = int(100 + 100 * abs(np.sin(time.time() * 4)))
            alert_bg = pygame.Surface((self.sidebar_width - 40, 40), pygame.SRCALPHA)
            alert_bg.fill((*self.THEME['danger'], pulse_alpha))
            sidebar.blit(alert_bg, (20, y_offset - 5))
        
        alert_text = self.font_medium.render(alert_status, True, alert_color)
        sidebar.blit(alert_text, (30, y_offset))
        y_offset += 60
        
        # Glowing separator
        pygame.draw.line(sidebar, self.THEME['border'], (20, y_offset), (self.sidebar_width - 20, y_offset), 3)
        y_offset += 30
        
        # Recent Alerts section
        alerts_title = self.font_medium.render("▸ ACTIVITY LOG", True, self.THEME['primary'])
        sidebar.blit(alerts_title, (20, y_offset))
        y_offset += 40
        
        # Display recent alerts with cyberpunk styling
        recent_alerts = self.alerts[-8:]  # Last 8 alerts
        for alert in reversed(recent_alerts):
            # Alert type icon and color
            if alert.alert_type == 'danger':
                icon = "⚠"
                color = self.THEME['danger']
            elif alert.alert_type == 'warning':
                icon = "⚡"
                color = self.THEME['warning']
            else:
                icon = "ℹ"
                color = self.THEME['primary']
            
            # Time
            time_str = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
            time_text = self.font_small.render(f"{icon} {time_str}", True, color)
            sidebar.blit(time_text, (30, y_offset))
            y_offset += 20
            
            # Message (truncate if too long)
            message = alert.message
            if len(message) > 35:
                message = message[:32] + "..."
            msg_text = self.font_small.render(f"  {message}", True, self.THEME['text'])
            sidebar.blit(msg_text, (30, y_offset))
            y_offset += 25
            
            if y_offset > self.window_height - 150:
                break
        
        # Controls section at bottom
        y_offset = self.window_height - 140
        pygame.draw.line(sidebar, self.THEME['border'], (20, y_offset), (self.sidebar_width - 20, y_offset), 3)
        y_offset += 20
        
        controls_title = self.font_medium.render("▸ CONTROLS", True, self.THEME['primary'])
        sidebar.blit(controls_title, (20, y_offset))
        y_offset += 35
        
        controls = [
            ("ESC/Q", "Quit"),
            ("R", "Reset"),
            ("Z", "Zones"),
            ("S", "Screenshot"),
            ("F", "Panel")
        ]
        
        for key, action in controls:
            # Draw key in cyan
            key_text = self.font_small.render(key, True, self.THEME['primary'])
            sidebar.blit(key_text, (30, y_offset))
            
            # Draw separator
            sep_text = self.font_small.render("-", True, self.THEME['text_secondary'])
            sidebar.blit(sep_text, (100, y_offset))
            
            # Draw action in gray
            action_text = self.font_small.render(action, True, self.THEME['text_secondary'])
            sidebar.blit(action_text, (120, y_offset))
            
            y_offset += 20
        
        return sidebar
    
    def run(self):
        """Main loop for the monitoring system"""
        running = True
        
        while running:
            # Update FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_update)
                self.frame_count = 0
                self.last_fps_update = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:  # Quit
                        running = False
                    elif event.key == pygame.K_r:  # Reset alerts
                        self.alert_active = False
                        self.last_alert_time = 0
                        self.add_alert("Alerts reset", 'info')
                    elif event.key == pygame.K_z:  # Toggle zones
                        self.show_zones = not self.show_zones
                        zone_status = "visible" if self.show_zones else "hidden"
                        self.add_alert(f"Zones {zone_status}", 'info')
                    elif event.key == pygame.K_s:  # Screenshot
                        ret, screenshot_frame = self.cap.read()
                        if ret:
                            self._take_screenshot(screenshot_frame, "manual")
                            self.add_alert("📸 Manual screenshot captured", 'info')
                    elif event.key == pygame.K_f:  # Toggle sidebar
                        self.show_sidebar = not self.show_sidebar
                        sidebar_status = "shown" if self.show_sidebar else "hidden"
                        self.add_alert(f"Panel {sidebar_status}", 'info')
            
            # Get camera frame
            ret, frame = self.cap.read()
            if not ret:
                self.add_alert("Error: Could not read frame from camera", 'danger')
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Convert OpenCV frame to Pygame surface and scale it
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.scaled_video_width, self.scaled_video_height))
            frame_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
            
            # Clear screen with gradient background
            self.screen.fill(self.THEME['bg'])
            
            # Draw animated background grid
            grid_time = time.time() * 0.5
            for i in range(0, self.window_width, 50):
                alpha = int(30 + 20 * abs(np.sin(grid_time + i * 0.01)))
                color = (self.THEME['border'][0], self.THEME['border'][1], self.THEME['border'][2])
                pygame.draw.line(self.screen, (*color, alpha) if len(color) == 3 else color, (i, 0), (i, self.window_height), 1)
            for i in range(0, self.window_height, 50):
                alpha = int(30 + 20 * abs(np.sin(grid_time + i * 0.01)))
                color = (self.THEME['border'][0], self.THEME['border'][1], self.THEME['border'][2])
                pygame.draw.line(self.screen, (*color, alpha) if len(color) == 3 else color, (0, i), (self.window_width, i), 1)
            
            # Draw glowing border around video
            glow_time = time.time() * 3
            glow_intensity = int(150 + 105 * abs(np.sin(glow_time)))
            border_color = (0, glow_intensity, glow_intensity)
            border_rect = pygame.Rect(self.video_x - 5, self.video_y - 5, 
                                     self.scaled_video_width + 10, self.scaled_video_height + 10)
            pygame.draw.rect(self.screen, border_color, border_rect, 3)
            
            # Draw camera feed
            self.screen.blit(frame_surface, (self.video_x, self.video_y))
            
            # Draw top status bar
            status_bar_height = 50
            status_bar = pygame.Surface((self.window_width, status_bar_height), pygame.SRCALPHA)
            status_bar.fill((*self.THEME['card'], 200))
            
            # System title
            title = self.font_medium.render("🔮 AREA SURVEILLANCE SYSTEM", True, self.THEME['primary'])
            status_bar.blit(title, (20, 15))
            
            # Live indicator
            live_pulse = int(200 + 55 * abs(np.sin(time.time() * 4)))
            live_color = (255, live_pulse, live_pulse)
            pygame.draw.circle(status_bar, live_color, (self.window_width - 100, 25), 8)
            live_text = self.font_small.render("LIVE", True, self.THEME['danger'])
            status_bar.blit(live_text, (self.window_width - 80, 18))
            
            self.screen.blit(status_bar, (0, 0))
            
            # Draw sidebar if enabled
            if self.show_sidebar:
                sidebar = self.draw_sidebar()
                self.screen.blit(sidebar, (self.window_width - self.sidebar_width, 0))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # Cap at 60 FPS
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'screen'):
            pygame.quit()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    try:
        monitor = AreaMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'monitor' in locals():
            monitor.cleanup()
        print("\nSystem shutdown complete")
if __name__ == "__main__":
    main()
