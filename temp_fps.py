import cv2
import time
import torch
import progressbar
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

input_file = '../videos/video.mp4'
cap = cv2.VideoCapture(input_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend('best.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
names = model.names

print('DEVICE:', device)

pTime = 0
while True:
    success, img = cap.read()
    ######

    im = letterbox(img, 640, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    #dataset = LoadImages(input_file, img_size=(640,640), stride=32, auto=True, vid_stride=1)

    im = torch.from_numpy(im).to(model.device)
    im = im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

    for j, det in enumerate(pred):
        im0 = img.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=3, example=str(names))

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

    ######

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime =cTime

    print('FPS:', fps)
    # cv2.putText(im0, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    #
    # cv2.imshow('Image', im0)
    cv2.waitKey(1)
