from django.urls import path
from cctv import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream', views.video_stream1, name='video_stream1'),
    path('video_stream1', views.video_stream2, name='video_stream2'),
    path('cctv1/', views.cctv1),
    path('cctv2/', views.cctv2),
    path('ajax/', views.ajax_method),
]