from django.shortcuts import render
from django.http.response import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from cctv.camera import VideoCamera
from django.contrib.auth.models import User
from threading import Thread

CAMERA  = None
check_cam = []

# Create your views here.
def index(request):
    global CAMERA
    user = None
    CAMERA = VideoCamera()
    t = Thread(target=CAMERA.run_server)
    t.start()
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }
    return render(request, 'cctv/live.html', context=context)


def gen1(camera):
    print('###')
    while len(camera.threads)==0:
        pass
    while True:
        frame = camera.threads[0].get_frame(0)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream1(request):
    return StreamingHttpResponse(gen1(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def gen2(camera):
    print('@@@')
    while len(camera.threads)>=2 and True:
        frame = camera.threads[1].get_frame(1)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream2(request):
    while len(CAMERA.threads)<=1:
        pass
    return StreamingHttpResponse(gen2(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def cctv1(request):
    global CAMERA
    user = None
    if CAMERA==None:
        CAMERA = VideoCamera()
        t = Thread(target=CAMERA.run_server)
        t.start()
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/cctv1.html', context=context)

def cctv2(request):
    global CAMERA
    user = None
    if CAMERA==None:
        CAMERA = VideoCamera()
        t = Thread(target=CAMERA.run_server)
        t.start()
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/cctv2.html', context=context)

@csrf_exempt
def ajax_method(request):
    sendmessage = ""
    print(check_cam)
    for i in check_cam: # 모든 라즈베리파이 연결에 대해서
        if i==False: # 평상시
            sendmessage = sendmessage+"0"
        else: # 특정 상황 감지 시
            sendmessage = sendmessage+"1"

    receive_message = request.POST.get('send_data')
    send_message = {'send_data' : sendmessage}
    return JsonResponse(send_message) # 감지 결과 전송