import cv2
import numpy as np
from ultralytics import YOLO
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def load_yolo_model():
    try:
        YOLO_MODEL_PATH = os.path.join(settings.BASE_DIR, 'app', 'yolov8n.pt')
        logger.info(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
        
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error("YOLO model file not found!")
            model = YOLO('yolov8n.pt')
            model.save(YOLO_MODEL_PATH)
            logger.info("Downloaded and saved YOLO model")
        else:
            model = YOLO(YOLO_MODEL_PATH)
        
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        raise

model = load_yolo_model()

import cv2
import os
import logging
from ultralytics import YOLO
from django.conf import settings

logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO(os.path.join(settings.BASE_DIR, 'app', 'yolov8n.pt'))

def process_video_with_yolo(input_video_path, output_dir):
    try:
        # Validate input
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        # Prepare output path with MP4 extension
        video_name = os.path.splitext(os.path.basename(input_video_path))[0] + '_processed.mp4'
        output_video_path = os.path.join(output_dir, video_name)
        frames_dir = os.path.join(settings.BASE_DIR, 'frames')
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer with MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"Could not initialize video writer for {output_video_path}")

        # Process frames
        detection_frames = []
        frame_count = 0
        detection_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model(frame, verbose=False)
            annotated_frame = frame.copy()
            has_detections = False

            # Draw detections
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Person class
                    detection_count += 1
                    has_detections = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Theif: {box.conf[0]:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Save frame if detections found
            if has_detections:
                frame_filename = f"detection_{frame_count:04d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                detection_frames.append(frame_filename)

            # Write to output video
            out.write(annotated_frame if has_detections else frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        # Verify output video
        if not os.path.exists(output_video_path):
            raise IOError("Output video file was not created")
            
        if os.path.getsize(output_video_path) == 0:
            os.remove(output_video_path)
            raise IOError("Output video file is empty")

        return {
            'output_video_path': output_video_path,  # Full system path
            'output_video_name': video_name,         # Just the filename
            'detection_frames': detection_frames,
            'frame_count': frame_count,
            'detection_count': detection_count
        }

    except Exception as e:
        logger.error(f"YOLO processing error: {str(e)}")
        if 'out' in locals() and out.isOpened():
            out.release()
        raise