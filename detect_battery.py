# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import statistics as st
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# 追記
from logging import getLogger
import time
import statistics
import serial

import socket

HOST = '100.75.254.126' # Raspberry PiのIPアドレス
PORT = 50007 # ポート番号

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        machine_speed=0.0667,  # speed m/s
        camera_coverage_length=22.5,  # camera coverage length cm
        com='COM4',
        camera_sensor_length=35,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 追記
    logger = getLogger(__name__)
    object_hantei="False"
    
    while True:
        try:
            print("値受け取り")
            data = ""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # ソケットをバインド
                s.bind(("", PORT))
                # 接続待ちを開始
                s.listen(1)
                print('Waiting for a connection...')
                # 接続を受け付ける
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        # データを受信する
                        data = conn.recv(1024)
                        if data != "":
                            break

            print(data)
            result = data.decode('utf-8')
            print(result)
            if data == b'':
                raise Exception('値なし')

            print(result)
            
            if result != None:
                machine_speed=0.0667
                camera_sensor_length=30
                object_hantei="False"
                print("camera_sensor_length = " + str(camera_sensor_length))
                print("machine_speed = " + str(machine_speed))
                print(str((camera_sensor_length/100)))
                time.sleep((camera_sensor_length/100)/machine_speed)
                # 計算
                point = (camera_coverage_length/100)/machine_speed + float(result)
                point1 = 0
                time_start = time.time()
                result_0 = list()
                result_2 = list()
                frame_count = 0
                for path, im, im0s, vid_cap, s in dataset:
                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                        for i, det in enumerate(pred):  # per image
                            frame_count += 1 # frame数
                            seen += 1
                            if webcam:  # batch_size >= 1
                                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                                s += f'{i}: '
                            else:
                                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                            p = Path(p)  # to Path
                            save_path = str(save_dir / p.name)  # im.jpg
                            txt_path = str(save_dir / 'labels' / p.stem) + (
                                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                            s += '%gx%g ' % im.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy() if save_crop else im0  # for save_crop
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, 5].unique():
                                    n = (det[:, 5] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                    result_0.append(names[int(c)])
                                    

                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # Write to file
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                            -1).tolist()  # normalized xywh
                                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                        with open(f'{txt_path}.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    if save_img or save_crop or view_img:  # Add bbox to image
                                        c = int(cls)  # integer class
                                        label = None if hide_labels else (
                                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                    if save_crop:
                                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                     BGR=True)

                            # Stream results
                            im0 = annotator.result()
                            if view_img:
                                if platform.system() == 'Linux' and p not in windows:
                                    windows.append(p)
                                    cv2.namedWindow(str(p),
                                                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                                cv2.imshow(str(p), im0)
                                cv2.waitKey(1)  # 1 millisecond

                        # Print time (inference-only)
                        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
                        time_end = time.time()

                        # if time_end - time_start > point+1:
                        if time_end - time_start > point:
                            if i:
                                print("{} 秒".format(time_end - time_start))
                                # print("リザルト　{}".format(result_0))
                                if len(result_0) > 0:
                                    print(result_0)
                                    print(statistics.mode(result_0))
                                    print("総合評価は {} です".format(statistics.multimode(result_0)))
                                    print("フレーム数は{}".format(frame_count))

                                    # 判定アルゴリズム
                                    hantei=0

                                    if len(statistics.multimode(result_0)) == 2 and frame_count * 0.01 < result_0.count(str(statistics.multimode(result_0[0])).strip("[' ']")):
                                        if 'Li_ion' in statistics.multimode(result_0) and 'Recycle symbol' in statistics.multimode(result_0):
                                            print("リチウムイオン電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif 'Ni_Cd' in statistics.multimode(result_0) and 'Recycle symbol' in statistics.multimode(result_0):
                                            print("ニカド電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif 'Ni_MH' in statistics.multimode(result_0) and 'Recycle symbol' in statistics.multimode(result_0):
                                            print("ニッケル水素電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        else:
                                            print("ERROR : 判定不可") # ex)['CR_Lithium', 'LR_Alkaline'],['Alkaline','Recycle symbol']


                                    elif len(statistics.multimode(result_0)) == 1 and frame_count * 0.01 < result_0.count(str(statistics.multimode(result_0)).strip("[' ']")):
                                        if ['Alkaline'] == statistics.multimode(result_0):
                                            print("アルカリ電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Alkaline_jp'] == statistics.multimode(result_0):
                                            print("アルカリ電池jp")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['LR_Alkaline'] == statistics.multimode(result_0):
                                            print("アルカリ電池LR")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['CR_Lithium'] == statistics.multimode(result_0):
                                            print("リチウム電池CR")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Lithium'] == statistics.multimode(result_0):
                                            print("リチウム電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Lithium_jp'] == statistics.multimode(result_0):
                                            print("リチウム電池jp")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Manganese'] == statistics.multimode(result_0):
                                            print("マンガン電池")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Manganese_jp'] == statistics.multimode(result_0):
                                            print("マンガン電池jp")
                                            #hantei=sensor_time_end - sensor_time_start
                                        elif ['Recycle symbol'] == statistics.multimode(result_0):
                                            print("2番目を見る")
                                            [result_2.append(s) for s in result_0 if s != str(statistics.multimode(result_0)).strip("[' ']")]
                                            if  statistics.multimode(result_2) == ['Li_ion']:
                                                print("リチウムイオン電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            elif statistics.multimode(result_2) == ['Ni_Cd']:
                                                print("ニカド電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            elif statistics.multimode(result_2) == ['Ni_MH']:
                                                print("ニッケル水素電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            else:
                                                print("ERROR : 判定不可")
                                        elif ['Li_ion'] == statistics.multimode(result_0):
                                            print("2番目を見る")
                                            [result_2.append(s) for s in result_0 if s != str(statistics.multimode(result_0)).strip("[' ']")]
                                            if statistics.multimode(result_2) == ['Recycle symbol']:
                                                print("リチウムイオン電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            else:
                                                print("ERROR : 判定不可")
                                        elif ['Ni_Cd'] == statistics.multimode(result_0):
                                            print("2番目を見る")
                                            [result_2.append(s) for s in result_0 if s != str(statistics.multimode(result_0)).strip("[' ']")]
                                            if statistics.multimode(result_2) == ['Recycle symbol']:
                                                print("ニカド電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            else:
                                                print("ERROR : 判定不可")
                                        elif ['Ni_MH'] == statistics.multimode(result_0):
                                            print("2番目を見る")
                                            [result_2.append(s) for s in result_0 if s != str(statistics.multimode(result_0)).strip("[' ']")]
                                            if statistics.multimode(result_2) == ['Recycle symbol']:
                                                print("ニッケル水素電池")
                                                #hantei=sensor_time_end - sensor_time_start
                                            else:
                                                print("ERROR : 判定不可")
                                    else:
                                        # 最頻値が3つ以上
                                        print("ERROR : 最頻値が3つ以上あります")

                                    print(hantei)
                                    remote_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    remote_server_socket.connect(("100.75.254.126", 50008))
                                    remote_server_socket.sendall(b'%0.2f\n'% float(result))
                                    print("ラズパイにデータを送信")
                                    remote_server_socket.close()
                                else:
                                    print("None")
                            break

        except KeyboardInterrupt:
            print("終了")
            break

        except Exception as e:
	        print(e) 


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    #追記
    parser.add_argument('--machine-speed', type=float, default=10, help='speed m/s')
    parser.add_argument('--camera-coverage-length', type=float, default=18.5, help='camera coverage length cm')
    parser.add_argument('--com', type=str, default='COM4', help='serial')
    parser.add_argument('--camera-sensor-length', type=int, default=60, help='destance between camera and sensor')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
