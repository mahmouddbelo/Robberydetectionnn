from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_video, name='upload_video'),
    path('detect/<str:video_path>/', views.detect_shoplifter, name='detect_shoplifter'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
  + static(settings.YOLO_URL, document_root=settings.YOLO_ROOT) \
  + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)