import cv2
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1)
mp_drawing = mp.solutions.drawing_utils


import numpy as np

def enhance_contrast(matrix, min_val=0.0, max_val=1.2, gamma=2.0):
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    contrasted_matrix = np.power(normalized_matrix, gamma)
    enhanced_matrix = contrasted_matrix * (max_val - min_val) + min_val

    return enhanced_matrix


def get_pose_coords(image, pose, mask_extractor, debug):
    mask = mask_extractor.get_person_masks(image)[0]

    contour = np.column_stack(np.where(mask == 255))
    
    if contour.size == 0:
        return None

    y_min, x_min = contour.min(axis=0)
    y_max, x_max = contour.max(axis=0)

    cropped_image = image[y_min:y_max, x_min:x_max]
    
    height, width, _ = cropped_image.shape
    rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks is None:
        results = pose.process(cropped_image)

    if results.pose_landmarks is None:
        return None
    
    points = (np.array([[point.x, point.y] for point in results.pose_landmarks.landmark]) * (width, height)).astype('int')
    
    points += np.array([x_min, y_min])
    if debug is None:
        debug = cropped_image.copy()
    mp_drawing.draw_landmarks(debug, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return points, debug

def align_images(target_image, reference_image, target_points, reference_points):
    target_points = np.array(target_points, dtype="float32")
    reference_points = np.array(reference_points, dtype="float32")
    
    matrix, _ = cv2.findHomography(target_points, reference_points, cv2.RANSAC)
    aligned_image = cv2.warpPerspective(target_image, matrix, (reference_image.shape[1], reference_image.shape[0]))

    return aligned_image

def transform_using_pose(shadow_info, alpha_image, mask_extractor, pose=POSE):

    alpha_points, debug = get_pose_coords(alpha_image, pose, mask_extractor, None)
    if alpha_points is None:
        return "No pose found"
    debug = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
    shadow_points, debug = get_pose_coords(shadow_info, pose, mask_extractor, debug)
    if shadow_points is None:
        return "No pose found"
    
    cv2.imwrite('positioning_junk/pose_debug.jpg', debug)
    

    positioned_model = align_images(shadow_info, alpha_image, shadow_points, alpha_points)
    return positioned_model

def transform_image(image, roi, target, reference):

    x1_1, y1_1, x2_1, y2_1 = roi      
    x1_2, y1_2, x2_2, y2_2 = target   
    x1_3, y1_3, x2_3, y2_3 = reference

    width2, height2 = x2_2 - x1_2, y2_2 - y1_2
    width3, height3 = x2_3 - x1_3, y2_3 - y1_3

    roi_image = image[y1_1:y2_1, x1_1:x2_1].copy()

    # Step 2: Calculate the scale factor to resize bbox2 to the size of bbox3
    scale_x = width3 / width2
    scale_y = scale_x*0.971

    # Step 3: Resize the entire ROI to match the scale of bbox3
    roi_resized = cv2.resize(roi_image, 
                             (int((x2_1 - x1_1) * scale_x), int((y2_1 - y1_1) * scale_y)),
                             interpolation=cv2.INTER_LINEAR)

    # Ensure resized dimensions stay within the image bounds
    roi_resized_height, roi_resized_width = roi_resized.shape[:2]

    # Step 4: Calculate the offset to align the resized target (bbox2) with the reference (bbox3)
    # Calculate the position where roi_resized should be placed to make bbox2 overlap bbox3
    offset_x = x1_3 - int((x1_2 - x1_1) * scale_x)  # X offset to align target with reference
    offset_y = y1_3 - int((y1_2 - y1_1) * scale_y)  # Y offset to align target with reference

    # Step 5: Ensure that only valid parts of the resized ROI are placed within the image bounds
    # Calculate the portion of the resized ROI that fits within the image boundaries
    x_start = max(0, -offset_x)  # Starting point inside roi_resized
    y_start = max(0, -offset_y)  # Starting point inside roi_resized

    x_end = min(roi_resized_width, image.shape[1] - offset_x)  # End point for placing in the image
    y_end = min(roi_resized_height, image.shape[0] - offset_y)  # End point for placing in the image

    # Clip the offset to ensure it fits within the image
    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)

    # Step 6: Create a blank canvas with the same dimensions as the original image
    transformed_image = np.ones_like(image).astype('uint8') * 255.  # White background

    # Place the valid part of the resized ROI on the new image at the calculated position
    transformed_image[offset_y: offset_y + (y_end - y_start), offset_x: offset_x + (x_end - x_start)] = roi_resized[y_start:y_end, x_start:x_end]

    return transformed_image

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)  
    return (x, y, x + w, y + h)
