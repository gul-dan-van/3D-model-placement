from time import time

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from helper import transform_image, get_bounding_box, enhance_contrast


start = time()

keyword = 'image10'
# enhance = True
enhance = False
angle='-45'
for angle in [135]:
    angle = str(angle)
    BLENDED_IMAGE = f'/Users/gauranshurathee/Desktop/shadow_positioning/nov26/hard_blended/{keyword}_colored_blended.png'
    ALPHA_IMAGE = f'/Users/gauranshurathee/Desktop/shadow_positioning/nov26/hard_alpha/{keyword}_colored_alpha.png'
    MODEL_MASK = f'{keyword}0001.jpg'
    SHADOW_INFO = f'{keyword}.png'
    RESULT_PATH = f'/Users/gauranshurathee/Desktop/shadow_positioning/results_f35/{keyword}.png'


    ############################## LOADING IMAGE ##############################
    blended_image = cv2.imread(BLENDED_IMAGE)
    height, width = blended_image.shape[:2]
    shadow_info = cv2.imread(SHADOW_INFO)
    model_mask = (cv2.imread(MODEL_MASK, 0)>128).astype('uint8')*255 # Single Channeled
    max_dimension = min(height, width)

    ############################## RESIZING SHADOW INFO ##############################
    shadow_h, shadow_w = shadow_info.shape[:2]
    scale_factor = max_dimension / max(shadow_h, shadow_w)
    new_shadow_w = int(shadow_w * scale_factor)
    new_shadow_h = int(shadow_h * scale_factor)
    resized_shadow_info = cv2.resize(shadow_info, (new_shadow_w, new_shadow_h), interpolation=cv2.INTER_LINEAR)
    scaled_shadow_info = 255. * resized_shadow_info / resized_shadow_info[0].min()
    resized_model_mask = cv2.resize(model_mask, (new_shadow_w, new_shadow_h), interpolation=cv2.INTER_LINEAR)

    ############################## PADDING SHADOW INFO ##############################
    empty_info = np.ones_like(blended_image) * 255.
    x_offset = (width - new_shadow_w) // 2
    y_offset = (height - new_shadow_h) // 2
    empty_info[y_offset:y_offset + new_shadow_h, x_offset:x_offset + new_shadow_w] = scaled_shadow_info
    shadow_info = empty_info


    empty_mask = np.zeros_like(blended_image)[:,:,0]
    x_offset = (width - new_shadow_w) // 2
    y_offset = (height - new_shadow_h) // 2
    empty_mask[y_offset:y_offset + new_shadow_h, x_offset:x_offset + new_shadow_w] = resized_model_mask
    model_mask = empty_mask
    model_mask = (model_mask>128).astype('uint8')*255

    # debug = np.zeros_like(blended_image)
    # # debug[:,:,0] = shadow_mask.mean(axis=2)
    # debug[:,:,1] = (composite_mask>128).astype('int')*255
    # debug[:,:,2] = ((positioned_model*255).mean(axis=2)<250).astype('uint8')*255
    # cv2.imwrite('junk/debug.png', debug)


    ############################## FETCHING MASKS AND NORMALIZING ##############################
    composite_mask = (cv2.imread(ALPHA_IMAGE, cv2.IMREAD_UNCHANGED)[:,:,3]>128).astype('uint8')*255 # Single channeled
    composite_bbox = get_bounding_box(composite_mask)
    shadow_bbox = (0, 0, width, height)
    model_bbox = get_bounding_box(model_mask)

    # debug = np.zeros_like(blended_image)
    # # debug[:,:,0] = shadow_mask.mean(axis=2)
    # debug[:,:,1] = model_mask
    # debug[:,:,2] = (scaled_shadow.mean(axis=2)<250).astype('uint8')*255
    # cv2.imwrite('positioning_junk/debug_edited.png', debug)


    ############################## REPOSITIONING AND SHADOW CREATION ##############################
    positioned_model = transform_image(shadow_info, shadow_bbox, model_bbox, composite_bbox) / 255
    positioned_mask = transform_image(model_mask, shadow_bbox, model_bbox, composite_bbox)
    positioned_model = positioned_model*1.1
    if enhance == True:
        positioned_model = enhance_contrast(positioned_model, min_val=0.0, max_val=1.1, gamma=1.3)
    blurred_model = cv2.GaussianBlur(positioned_model, (3, 3), sigmaX=0)

    composite_mask_3 = np.stack([(composite_mask>128).astype('int')]*3, axis=2)
    composite_image = composite_mask_3*blurred_model + (1-composite_mask_3)*positioned_model
    composite_image = np.clip(composite_image, 0., 1.)
    shadowed_image = (blended_image * composite_image)

    debug = np.zeros_like(blended_image)
    debug[:,:,0] = positioned_mask.astype('uint8')
    debug[:,:,1] = (composite_mask>128).astype('uint8')*255
    debug[:,:,2] = ((positioned_model*255).mean(axis=2)).astype('uint8')
    cv2.imwrite('junk/debug.png', debug)
    # cv2.imwrite('junk/debug_new.png', (positioned_model*255).clip(0,255).astype('uint8'))


    lam = 1.
    shadowed_image = shadowed_image*lam + blended_image*(1-lam)
    shadowed_image = np.clip(shadowed_image, 0, 255)

    cv2.imwrite(RESULT_PATH, shadowed_image.astype('uint8'))
    end = time()
    print(f'Total Time Taken: {end-start}')
