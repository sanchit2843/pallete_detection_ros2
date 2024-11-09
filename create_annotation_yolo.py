import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import os
import cv2
import argparse
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours if len(contour) >= 3]
    return polygons

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--input_dir", type=str, default=".")
    args = parser.parse_args()

    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    CLASSES = ["pallets", "floor"]
    BOX_THRESHOLD = 0.25

    TEXT_THRESHOLD = 0.1
    NMS_THRESHOLD = 0.8

    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(os.listdir(args.input_dir)):
        SOURCE_IMAGE_PATH = os.path.join(args.input_dir, i)
        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)
        height, width = image.shape[:2]

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        txt_filename = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(SOURCE_IMAGE_PATH))[0]}.txt")

        with open(txt_filename, 'w') as f:    
            for idx, mask in enumerate(detections.mask):
                polygons = mask_to_polygon(mask)
                polygons_normalized = [[coord / width if i % 2 == 0 else coord / height for i, coord in enumerate(poly)] for poly in polygons]
                for poly in polygons_normalized:
                    # Assuming `detections.class_id[idx]` contains the class index
                    line = f"{detections.class_id[idx]} " + " ".join(map(str, poly)) + "\n"
                    f.write(line)
        