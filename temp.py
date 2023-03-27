import argparse
import cv2
import torch
import progressbar
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes

parser = argparse.ArgumentParser()

parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path or triton URL')
parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--input_file', type=str, default='', help='input file name')
parser.add_argument('--output_file', type=str, default='new_video.mp4', help='output file name')

def main():
    global args
    args = parser.parse_args()

    output_file = args.output_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.data, fp16=False)
    dataset = LoadImages(args.input_file, img_size=(640,640), stride=32, auto=True, vid_stride=1)

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

        if critical_point > 0:
            out.write(im0s)
            critical_point -= 1
        else:
            if len(queue) > fps*seconds:
                queue.pop(0)
            queue.append(im0s)

        im = torch.from_numpy(im).to(model.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        for j, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if 3 in det[:, 5].unique():
                score_validation += 1

                if score_validation >= score_detection_threshold and not checked:
                    # cv2 write video
                    # write 3 seconds from buffer queue
                    for i in queue:
                        out.write(i)

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
