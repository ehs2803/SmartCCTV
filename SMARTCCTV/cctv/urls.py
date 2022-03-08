from django.urls import path
from .import views
urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream', views.video_stream, name='video_stream'),
    path('video_stream1', views.video_stream1, name='video_stream1'),
    path('test1/', views.test11),
    path('test2/', views.test22),
    path('ajax/', views.ajax_method),
]