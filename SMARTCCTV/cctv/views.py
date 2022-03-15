'''
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from cctv.camera import VideoCamera
from django.contrib.auth.models import User

# Create your views here.
def index(request):
    user = None
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/live.html', context=context)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

'''

from django.shortcuts import render
from django.http.response import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from cctv.camera import VideoCamera
from django.contrib.auth.models import User
from threading import Thread

from playsound import playsound

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

def test1():
    while True:
        print(1)
def test2():
    while True:
        print(2)

def gen(camera):
    print('###')
    while len(camera.threads)==0:
        pass
    while True:
        print(22222222222)
        frame = camera.threads[0].get_frame(0)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream(request):
    print(1111111111111111111111111111111111111111111111111111)
    return StreamingHttpResponse(gen(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def gen1(camera):
    print('@@@')
    while len(camera.threads)>=2 and True:
        print(33333333333)
        frame = camera.threads[1].get_frame(1)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_stream1(request):
    #t2 = Thread(target=test2)
    #t2.start()
    while len(CAMERA.threads)<=1:
        pass
    print(22222222222222222222222222222222222222222222222222222222222)
    return StreamingHttpResponse(gen1(CAMERA),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def test11(request):
    #t1 = Thread(target=test1)
    #t1.start()
    global CAMERA
    user = None
    if CAMERA==None:
        CAMERA = VideoCamera()
        t = Thread(target=CAMERA.run_server)
        t.start()
        print(11111111111111111111111111111111111111111111111111111111111111111111111)
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/test1.html', context=context)

def test22(request):
    #t2 = Thread(target=test2)
    #t2.start()
    global CAMERA
    user = None
    if CAMERA==None:
        CAMERA = VideoCamera()
        t = Thread(target=CAMERA.run_server)
        t.start()
        print(2222222222222222222222222222222222222222222222222222222222222222222222222222)
    if request.session.get('id'):
        user = User.objects.get(id=request.session.get('id'))
    context = {
        'user': user
    }

    return render(request, 'cctv/test2.html', context=context)

@csrf_exempt
def ajax_method(request):
    sendmessage = "1"
    for i in check_cam:
        if i==False:
            sendmessage = sendmessage+"0"
        else:
            sendmessage = sendmessage+"1"

    receive_message = request.POST.get('send_data')
    send_message = {'send_data' : sendmessage}
    return JsonResponse(send_message)