# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import sys
from utils import printProgressBar

#   Create config
cfg = get_cfg()
while True:
    mode = input("Choose model (c - classification, s - segmentation, k - keypoints): ")
    if mode == "c":
        cfg.merge_from_file("/Users/kristapsalmanis/projects/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        break
    elif mode == "s":
        cfg.merge_from_file("/Users/kristapsalmanis/projects/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
        break
    elif mode == "k":
        cfg.merge_from_file("/Users/kristapsalmanis/projects/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"
        break

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)

#   Get video and its metadata
input_vid = cv2.VideoCapture(sys.argv[1])
width = int(input_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_vid.get(cv2.CAP_PROP_FPS)
frames_total = int(input_vid.get(cv2.CAP_PROP_FRAME_COUNT))

#   Output video
output_vid = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*'MP4V'), fps, (width,height))

#   Read input frame by frame
counter = 0
printProgressBar(0, frames_total, prefix = 'Progress:', suffix = 'Complete', length = 50)
success,image = input_vid.read()
while success:
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = v.get_image()[:, :, ::-1]

    output_vid.write(image)
    success,image = input_vid.read()
    counter += 1
    printProgressBar(counter, frames_total, prefix = 'Progress:', suffix = 'Complete', length = 50)
output_vid.release()
cv2.destroyAllWindows()