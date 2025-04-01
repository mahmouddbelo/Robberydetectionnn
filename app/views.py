import os
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import torch
import shutil
import signal
import atexit
from django.conf import settings
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from functions import classify_video
from .yolo_utils import process_video_with_yolo

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
try:
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-short")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-short", num_labels=2
    )
    model_path = os.path.join(settings.BASE_DIR, 'app', 'shoplifting_detector.pt')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def home(request):
    return render(request, 'home.html')

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from .yolo_utils import process_video_with_yolo

def detect_shoplifter(request, video_path):
    try:
        # Validate input
        video_full_path = os.path.join(settings.MEDIA_ROOT, video_path)
        if not os.path.exists(video_full_path):
            raise FileNotFoundError(f"Video file not found: {video_full_path}")

        video_url = f"{settings.MEDIA_URL}{video_path}"
        
        # Process with YOLO
        results = process_video_with_yolo(video_full_path, settings.YOLO_ROOT)

        # Prepare context with both paths
        context = {
            'original_video': video_url,
            'processed_video_path': os.path.join(settings.YOLO_URL, results['output_video_name']),
            'processed_video_name': results['output_video_name'],
            'prediction': "Shoplifter" if results['detection_count'] > 0 else "Non-Shoplifter",
            'detection_count': results['detection_count'],
            'media_url': settings.MEDIA_URL,
            'frames_dir': 'frames/',
            'frames': results['detection_frames'],
            'yolo_url': settings.YOLO_URL
        }

        return render(request, 'detection.html', context)

    except Exception as e:
        return render(request, 'error.html', {
            'error': str(e),
            'message': "Video processing failed"
        })

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            # Save uploaded file
            fs = FileSystemStorage()
            video_file = request.FILES['video']
            
            # Ensure valid video extension
            valid_extensions = ['.mp4', '.avi', '.mov']
            file_ext = os.path.splitext(video_file.name)[1].lower()
            if file_ext not in valid_extensions:
                raise ValueError("Only MP4, AVI, or MOV files are allowed")
            
            filename = fs.save(video_file.name, video_file)
            
            # Redirect to processing
            return redirect('detect_shoplifter', video_path=filename)
        except Exception as e:
            return render(request, 'error.html', {
                'error': str(e),
                'message': "Upload failed"
            })
    return render(request, 'upload.html')

MEDIA_PATH = os.path.join(settings.BASE_DIR, 'media')
YOLO_PATH = os.path.join(settings.BASE_DIR, 'yolo')  
FRAMES_PATH = os.path.join(settings.BASE_DIR, 'frames') 

def cleanup_folder(folder_path):
    """Deletes all contents inside a folder but keeps the folder itself."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print(f"🧹 {folder_path} contents deleted.")

def cleanup_media():
    cleanup_folder(MEDIA_PATH)
    cleanup_folder(YOLO_PATH)
    cleanup_folder(FRAMES_PATH)

atexit.register(cleanup_media)

def handle_exit_signal(signum, frame):
    cleanup_media()
    print("Server stopped. Media folder contents removed.")
    exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)