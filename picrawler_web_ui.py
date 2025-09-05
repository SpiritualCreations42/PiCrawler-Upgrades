#!/usr/bin/env python3
"""
PiCrawler Web UI - Camera feed with object detection and robot control
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import sys
import threading
import base64
from picamera2 import Picamera2

# Add SunFounder path
sys.path.append('/home/captredbeard/picrawler')

try:
    from picrawler import Picrawler
    ROBOT_AVAILABLE = True
except ImportError:
    print("Robot not available - camera only mode")
    ROBOT_AVAILABLE = False

app = Flask(__name__)

class PiCrawlerWebDetector:
    def __init__(self):
        # Initialize robot
        if ROBOT_AVAILABLE:
            self.crawler = Picrawler()
            print("✓ Robot initialized")
        else:
            self.crawler = None
        
        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        print("✓ Camera initialized")
        
        # Detection parameters
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.move_threshold = 50
        self.speed = 80
        
        # Status variables
        self.following_mode = False
        self.last_detection = None
        self.detection_count = 0
        self.last_action = "None"
        self.running = True
        
        # Threading
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
    
    def detect_red_objects(self, frame):
        """Detect red objects in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'center': [center_x, center_y],
                    'area': area
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and UI elements"""
        # Draw detections
        for detection in detections:
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            
            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Label
            label = f"Red: {int(detection['area'])}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw center crosshairs
        cv2.line(frame, (self.frame_center_x - 30, self.frame_center_y), 
                (self.frame_center_x + 30, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 30), 
                (self.frame_center_x, self.frame_center_y + 30), (255, 255, 255), 2)
        
        # Status overlay
        mode_text = "FOLLOWING" if self.following_mode else "DETECTING"
        robot_text = "ROBOT: ON" if ROBOT_AVAILABLE else "ROBOT: OFF"
        
        # Background rectangles for text
        cv2.rectangle(frame, (5, 5), (400, 80), (0, 0, 0), -1)
        
        # Status text
        cv2.putText(frame, f"Mode: {mode_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"{robot_text} | Speed: {self.speed}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Objects: {len(detections)} | Action: {self.last_action}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def control_robot(self, detections):
        """Control robot based on detections"""
        if not self.following_mode or not detections or not ROBOT_AVAILABLE:
            return
        
        # Find largest detection
        largest = max(detections, key=lambda d: d['area'])
        target_x, target_y = largest['center']
        error_x = target_x - self.frame_center_x
        area = largest['area']
        
        # Control logic
        if abs(error_x) < self.move_threshold:
            if area < 8000:  # Too far
                self.last_action = "Forward"
                self.crawler.do_action('forward', 3, self.speed)
            elif area > 15000:  # Too close
                self.last_action = "Backward"
                self.crawler.do_action('backward', 2, self.speed // 2)
            else:
                self.last_action = "Hold Position"
        elif error_x > self.move_threshold:
            self.last_action = "Turn Right"
            self.crawler.do_action('turn right', 2, self.speed)
        else:
            self.last_action = "Turn Left"
            self.crawler.do_action('turn left', 2, self.speed)
    
    def detection_loop(self):
        """Main detection loop running in background"""
        while self.running:
            try:
                # Capture frame
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect objects
                detections = self.detect_red_objects(frame)
                self.detection_count = len(detections)
                
                # Control robot
                self.control_robot(detections)
                
                # Draw UI elements
                frame = self.draw_detections(frame, detections)
                
                # Store frame for web streaming
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                print(f"Detection loop error: {e}")
                time.sleep(1)
    
    def get_frame(self):
        """Get current frame for web streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if ROBOT_AVAILABLE:
            self.crawler.do_action('stop', 1, 0)
        self.picam2.stop()

# Global detector instance
detector = PiCrawlerWebDetector()

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        frame = detector.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS for web

@app.route('/')
def index():
    """Main web page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    """Handle control commands"""
    command = request.json.get('command')
    
    if command == 'toggle_follow':
        detector.following_mode = not detector.following_mode
        return jsonify({
            'success': True, 
            'following': detector.following_mode,
            'message': f"Following {'ON' if detector.following_mode else 'OFF'}"
        })
    
    elif command == 'stop' and ROBOT_AVAILABLE:
        detector.crawler.do_action('stop', 1, 0)
        detector.last_action = "Manual Stop"
        return jsonify({'success': True, 'message': 'Robot stopped'})
    
    elif command == 'speed_up':
        detector.speed = min(100, detector.speed + 10)
        return jsonify({'success': True, 'speed': detector.speed})
    
    elif command == 'speed_down':
        detector.speed = max(30, detector.speed - 10)
        return jsonify({'success': True, 'speed': detector.speed})
    
    # Manual control commands
    elif command in ['forward', 'backward', 'turn left', 'turn right'] and ROBOT_AVAILABLE:
        detector.crawler.do_action(command, 2, detector.speed)
        detector.last_action = f"Manual {command.title()}"
        return jsonify({'success': True, 'message': f'Executed {command}'})
    
    return jsonify({'success': False, 'message': 'Unknown command'})

@app.route('/status')
def status():
    """Get current status"""
    return jsonify({
        'following': detector.following_mode,
        'speed': detector.speed,
        'detections': detector.detection_count,
        'last_action': detector.last_action,
        'robot_available': ROBOT_AVAILABLE
    })

if __name__ == '__main__':
    try:
        print("Starting PiCrawler Web UI...")
        print("Access at: http://[your-pi-ip]:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        detector.cleanup()
