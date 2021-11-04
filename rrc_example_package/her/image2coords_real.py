# import sys
import numpy as np
import cv2
# import imutils
import time
import trifinger_cameras.py_tricamera_types as tricamera
from trifinger_cameras import utils
from rrc_example_package import rearrange_dice_env
# from rrc_example_package.example import PointAtDieGoalPositionsPolicy
# import trifinger_simulation.tasks.rearrange_dice as task
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

def process_sim_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def image2world(image_point, camera_parameters, z = 0.011):
    
    # get camera position and orientation separately
    tvec = camera_parameters.tf_world_to_camera[:3, 3]
    tvec = tvec[:, np.newaxis]
    rmat = camera_parameters.tf_world_to_camera[:3, :3]
    camMat = np.asarray(camera_parameters.camera_matrix)
    iRot = np.linalg.inv(rmat)
    iCam = np.linalg.inv(camMat)
    uvPoint = np.ones((3, 1))
    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]
    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)
    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
    wcPoint[2] = z #Hardcoded as z is always 0.011 if constrained to only push cube
    return tuple(map(float,wcPoint))
def get_2d_center(x, y, w, h):
    return (round((x + x + w) / 2), round((y+y+h) / 2))
    
def image2coords_real(camera_observation, camera_params, write_images=False, simulation=False):
    start = time.time()
    len_out = 0
    if simulation:
        convert_image=process_sim_image
    else:
        convert_image = utils.convert_image
    for i, c in enumerate(camera_observation.cameras):
        copy = convert_image(c.image.copy())
        seg_mask = segment_image(convert_image(c.image))
        contours = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        coord_out = []
        bbox_out = []
        for c in contours:
            # obtain the bounding rectangle coordinates for each square
            x, y, w, h = cv2.boundingRect(c)
            x_c, y_c = get_2d_center(x, y, w, h)
            world_point_c = image2world((x_c, y_c), camera_params[i], z = 0.011)
            coord_out.append(world_point_c)
            bbox_out.append((x, y, w, h))
            # out.append([(x, y, w, h), world_point_c]) # return bboxes and 3d point
            # With the bounding rectangle coordinates, draw a green bounding boxes and its centers for visualization purposes
            if write_images:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.circle(copy, (x_c, y_c), radius=0, color=(36, 255, 12), thickness=2)
        id = i + 10
        if write_images: 
            cv2.imwrite('test{}.png'.format(id), copy)
        #temporarilly keep the view with the highest number of detections
        if len_out < len(coord_out):
            coords = coord_out
            len_out = len(coord_out)
    end = time.time()
    print('Time to find coordinates: {}'.format(end - start))
    return coords
