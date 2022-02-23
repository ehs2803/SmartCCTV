import cv2
import numpy as np
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 9505

class VideoCamera(object):
    def __init__(self):
        #self.cap = cv2.VideoCapture(0)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))

        s = b''
    def __del__(self):
        pass
        #self.cap.release()
    def get_frame(self):
        data, addr = sock.recvfrom(46080)
        s += data

        if len(s) == (46080 * 20):
            frame = numpy.fromstring(s, dtype=numpy.uint8)
            frame = frame.reshape(480, 640, 3)
            cv2.imshow('test', frame)
            return frame.tobytes()
            '''
        ret, frame = self.cap.read()
        cv2.putText(frame, "test", (0, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
        frame_flip = cv2.flip(frame, 1)
        ret, frame = cv2.imencode('.jpg', frame_flip)
        return frame.tobytes()
        '''