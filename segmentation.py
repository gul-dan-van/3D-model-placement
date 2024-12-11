
import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM


class PersonSegmentationExtractor:
    YOLO_MODEL = "./utils/yolov10n.pt"
    SAM2_MODEL = "./utils/sam2_s.pt"

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO(self.YOLO_MODEL).to(self.device)
        self.sam2_model = SAM(self.SAM2_MODEL).to(self.device)
        print("Initialized Person Segmentation Extraction Models...")

    def get_person_bounding_boxes(self, image):
        outputs = self.yolo_model(image)[0]
        person_bboxes = []
        for i in range(len(outputs.boxes)):
            if int(outputs.boxes.cls[i]) == 0:  # Class 0 is for persons
                bbox = outputs.boxes.xyxy[i].cpu().numpy()  # Convert bbox to numpy array
                person_bboxes.append(bbox)

        return person_bboxes

    def get_person_masks(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get person bounding boxes
        bboxes = self.get_person_bounding_boxes(image)
        
        # Prepare masks list
        person_masks = []
        
        # Ensure bboxes are passed in the correct format (list of lists)
        for bbox in bboxes:
            bbox = [bbox]  # Convert bbox to a list of single bounding box if needed
            
            # Run SAM model on the image for the given bounding box
            results = self.sam2_model.predict(image_rgb, bboxes=bbox)  # Use the correct SAM model function
            
            # Extract masks from the result
            for index, result in enumerate(results):
                if self.device != 'cpu':
                    mask = result.masks.cpu().numpy()
                else:
                    mask = result.masks.numpy()

                mask = mask.data.squeeze(0)
                bin_mask = (mask > 0).astype(np.uint8) * 255
                person_masks.append(bin_mask)
        
        # Return all masks after processing all bounding boxes
        return person_masks
