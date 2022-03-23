import cv2
import numpy as np
import socket
import struct  # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle  # 바이트(bytes) 형식의 데이터 변환 모듈
import torch
import argparse
from playsound import playsound

from cctv.Detection.Utils import ResizePadding
from cctv.DetectorLoader import TinyYOLOv3_onecls
from cctv.PoseEstimateLoader import SPPE_FastPose
from cctv.fn import draw_single
from cctv.Track.Tracker import Detection, Tracker
from cctv.ActionsEstLoader import TSSTG

def preproc2(image):
    resize_fn = ResizePadding(384, 384) # 기본 384*384 size~
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def init_user():
    device = 'cuda'
    print("device : ",device)  # cuda
    # DETECTION MODEL.
    inp_dets = 384
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)
    # POSE MODEL.
    inp_pose = '224x160'.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1], device=device)
    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)
    # Actions Estimate.
    action_model = TSSTG()
    return detect_model,pose_model,tracker,action_model,inp_dets


def detect_human_algorithms(frame_img, init_args):
    # cap = cv2.VideoCapture(0)
    detect_model, pose_model, tracker, action_model, inp_dets = init_args

    # 새로 추가한 label classes
    classes = ["person", "bicycle", "car", " motorcycle",
               "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
               "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
               "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
               "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
               "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
               "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
               "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
               "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
               "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
               "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "wheelchair"]

    outvid = False

    # 여기가 그거였을 거여 아마 그 시작?
    frame = frame_img  # cam.getitem()

    # Detect humans bbox in the frame with detector model.

    detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

    # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
    tracker.predict()
    # Merge two source of predicted bbox together.
    for track in tracker.tracks:
        det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        detected = torch.cat([detected, det], dim=0) if detected is not None else det

    detections = []  # List of Detections object for tracking.
    if detected is not None:

        # Predict skeleton pose of each bboxs.
        poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
        # Create Detections object.
        detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                np.concatenate((ps['keypoints'].numpy(),
                                                ps['kp_score'].numpy()), axis=1),
                                ps['kp_score'].mean().numpy()) for ps in poses]

        # # 원래 있던 yolo VISUALIZE.
        for bb in detected[:, :]:
            detect_name = bb.type(torch.int64)[6]
            name = detect_name.tolist()
            if torch.equal(detect_name, torch.tensor(0)):  # 사람만 인식하여 그림 그려줌
                frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1)
                frame = cv2.putText(frame, classes[name], (int(bb[0]) + 5, int(bb[1]) - 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (0, 0, 255), 1)
            else:
                frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 1)
                frame = cv2.putText(frame, classes[name], (int(bb[0]) + 5, int(bb[1]) - 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (0, 255, 0), 1)

    # Update tracks by matching each track information of current and previous frame or
    # create a new track if no matched.
    tracker.update(detections)

    for i, track in enumerate(tracker.tracks):

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr().astype(int)
        center = track.get_center().astype(int)
        action = 'pending..'
        clr = (0, 255, 0)
        # Use 30 frames time-steps to prediction.
        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            if action_name == 'Fall Down':
                clr = (255, 0, 0)
            elif action_name == 'Lying Down':
                clr = (255, 200, 0)

            if action_name != 'Walking':  # action_name 이 핵심임 돌려보면 알거에요
                print("act", action_name)
        # VISUALIZE.
        if track.time_since_update == 0:
            if True:
                frame = draw_single(frame, track.keypoints_list[-1])
            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                0.4, (255, 0, 0), 2)
            frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.4, clr, 1)
    # Show Frame.
    frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)

    frame = frame[:, :, ::-1]

    return frame

'''
 tts_b_path = 'data/blink_count' + str(self.eye_count_min) + '.mp3'  # 알림 음성 파일
            playsound(tts_b_path)  # 음성으로 알림
'''
#from cctv.views import check_cam
import cctv.views

ip = '127.0.0.1'
port = 50002

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]
# 320×320 it’s small so less accuracy but better speed
# 609×609 it’s bigger so high accuracy and slow speed
# 416×416 it’s in the middle and you get a bit of both.


class VideoCamera(object):
    def __init__(self):
        self.threads = []

    def __del__(self):
        cv2.destroyAllWindows()

    def run_server(self):
        with socket.socket() as sock:
            sock.bind((ip, port))
            while True:
                # 소켓 스레드 통신
                sock.listen(5)
                conn, addr = sock.accept()
                f = Frame(conn)
                self.threads.append(f)
                cctv.views.check_cam.append(False)
            sock.close()
            print('server shutdown')
init_args_user =init_user()
class Frame:
    def __init__(self, client_socket):
        self.client_socket = client_socket
        self.data_buffer = b""
        self.data_size = struct.calcsize("L")

    def get_frame(self, n):
        # 설정한 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
        while len(self.data_buffer) <self.data_size:
            # 데이터 수신
            self.data_buffer += self.client_socket.recv(4096)

        # 버퍼의 저장된 데이터 분할
        packed_data_size = self.data_buffer[:self.data_size]
        self.data_buffer = self.data_buffer[self.data_size:]

        # struct.unpack : 변환된 바이트 객체를 원래의 데이터로 변환
        # - > : 빅 엔디안(big endian)
        #   - 엔디안(endian) : 컴퓨터의 메모리와 같은 1차원의 공간에 여러 개의 연속된 대상을 배열하는 방법
        #   - 빅 엔디안(big endian) : 최상위 바이트부터 차례대로 저장
        # - L : 부호없는 긴 정수(unsigned long) 4 bytes
        frame_size = struct.unpack(">L", packed_data_size)[0]

        # 프레임 데이터의 크기보다 버퍼에 저장된 데이터의 크기가 작은 경우
        while len(self.data_buffer) < frame_size:
            # 데이터 수신
            self.data_buffer += self.client_socket.recv(4096)

        # 프레임 데이터 분할
        frame_data = self.data_buffer[:frame_size]
        self.data_buffer = self.data_buffer[frame_size:]

        print("수신 프레임 크기 : {} bytes".format(frame_size))

        # loads : 직렬화된 데이터를 역직렬화
        # - 역직렬화(de-serialization) : 직렬화된 파일이나 바이트 객체를 원래의 데이터로 복원하는 것
        frame = pickle.loads(frame_data)

        # imdecode : 이미지(프레임) 디코딩
        # 1) 인코딩된 이미지 배열
        # 2) 이미지 파일을 읽을 때의 옵션
        #    - IMREAD_COLOR : 이미지를 COLOR로 읽음
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = preproc2(frame)
        frame = detect_human_algorithms(frame, init_args_user)

        #frame = yolo(frame=frame, size=size_list[0], score_threshold=0.4, nms_threshold=0.4, index=n)
        #frame_flip = cv2.flip(frame, 1)
        ret, frame = cv2.imencode('.jpg', frame)
        return frame.tobytes()
