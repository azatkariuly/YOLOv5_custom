import argparse
import cv2
import torch
import progressbar
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import increment_path, non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator, colors

parser = argparse.ArgumentParser()

parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path or triton URL')
parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--input_file', type=str, default='', help='input file name')
parser.add_argument('--output_file', type=str, default='new_video.mp4', help='output file name')
parser.add_argument('--view_labels', action='store_true', help='show labels in a video')

def main():
    global args
    args = parser.parse_args()

    # Directories
    save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=False)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    output_file = args.output_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)

    dataset = LoadImages(args.input_file, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    vid_path, vid_writer = [None], [None]

    seconds = 3

    score_validation = 0
    score_detection_threshold = 5

    checked = False

    queue = []
    critical_point = 0

    # Create a video capture object for the input file
    cap = cv2.VideoCapture(args.input_file)

    # Get the frame dimensions and FPS of the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a video writer object for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # use appropriate codec for your output file format
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames:', total_frames)

    bar = progressbar.ProgressBar(maxval=total_frames, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()


    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        bar.update(i+1)

        # if critical_point > 0:
        #     out.write(im0s)
        #     critical_point -= 1
        # else:

        im = torch.from_numpy(im).to(model.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        for j, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))

            ########################
            if len(det):
                if args.view_labels:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = (f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    im0 = annotator.result()

                if critical_point > 0:
                    vid_writer[j].write(im0)
                    critical_point -= 1
                else:
                    if len(queue) > fps*seconds:
                        queue.pop(0)
                    queue.append(im0)

                # TODO: implement 3 seconds after critical_point

                if 3 in det[:, 5].unique():
                    score_validation += 1

                    if score_validation >= score_detection_threshold and not checked:
                        # Save results (image with detections)
                        # 'video' or 'stream'
                        if vid_path[j] != save_path:  # new video
                            vid_path[j] = save_path
                            if isinstance(vid_writer[j], cv2.VideoWriter):
                                vid_writer[j].release()  # release previous video writer

                            # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[j] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                        for v in queue:
                            vid_writer[j].write(v)
                        queue = []

                        critical_point = fps*seconds

                        # write 3 seconds after this point
                        checked = True
                else:
                    checked = False
                    score_validation = 0

    bar.finish()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
