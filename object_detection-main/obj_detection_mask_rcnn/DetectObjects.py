import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util

from common_scripts import CommonFunctions
from common_scripts import Visualisation
from common_scripts.CameraSetup import CameraSetup
from common_scripts.Variables import Variables


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect_from_image(image):
    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    image_np_with_detections = draw_boxes(image_np_with_detections, detections)
    image_np_with_detections = Image.fromarray(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    image_np_with_detections.show()


def draw_boxes(image, detections, detection_masks=False):
    masks = None
    if detection_masks:
        masks = CommonFunctions.resize_predictions_detection_masks(predictions=detections,
                                                                   image_shape=image.shape,
                                                                   verbose=0,
                                                                   debug=0)
    ''' 
    viz_utils.visualize_boxes_and_labels_on_image_array(
       image,
       detections['detection_boxes'],
       detections['detection_classes'] + 1,
       detections['detection_scores'],
       category_index,
       instance_masks=masks,
       use_normalized_coordinates=True,
       max_boxes_to_draw=5,
       min_score_thresh=.8,
       agnostic_mode=False)
    '''

    Visualisation.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,
        detections['detection_scores'],
        category_index,
        instance_masks=masks,
        mean_coordinates=detections['mean_cordinates'],
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    return image


def get_3d_distance(pt1, pt2):
    p1 = np.array(pt1)
    p2 = np.array(pt2)

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def compute_distance_btw_objects(labels):
    print(labels)
    pass


def get_world_cordinates_of_pixel(depth_object, color_intrin, pt):
    z, _, _, _ = cv2.mean(depth_object)
    point_z = np.round(z * 0.001, 2)  # distance in meters
    point_x = np.round(((pt[0] + color_intrin.width / 2) - color_intrin.ppx) * point_z * (1 / color_intrin.fx), 2)
    point_y = np.round(((pt[1] + color_intrin.height / 2) - color_intrin.ppy) * point_z * (1 / color_intrin.fy), 2)
    return [point_x, point_y, point_z]


def get_world_cordinates_of_pixel_without_intrinsics(depth_object, pt):
    focal_length = 55
    z, _, _, _ = cv2.mean(depth_object)
    z = np.round(z * 0.001, 2)
    x = np.round(pt[0] * z / focal_length, 2)
    y = np.round(pt[1] * z / focal_length, 2)
    return [x, y, z]


def get_distance_from_camera(image, detections, intrinsics):
    labels = []
    min_pixels = []
    for k in range(detections['detection_scores'].shape[0]):
        if detections['detection_scores'][k] > 0.8:
            # label = category_index[int((detections['detection_classes'] + 1)[k])]
            # print(label)
            box = (tuple(detections['detection_boxes'][k].tolist()))
            y_min = int(box[0] * 480)
            x_min = int(box[1] * 640)
            y_max = int(box[2] * 480)
            x_max = int(box[3] * 640)
            x_mid, y_mid = [int(x_min + (x_max - x_min) / 2), int(y_min + (y_max - y_min) / 2)]
            depth_object = image[y_min:y_max, x_min:x_max].astype(float)
            cordinates_b = get_world_cordinates_of_pixel_without_intrinsics(depth_object, [x_mid, y_mid])
            cordinates = get_world_cordinates_of_pixel(depth_object, intrinsics,
                                                       [x_mid, y_mid])

            label = {
                'name': category_index[(detections['detection_classes'] + 1)[k]]['name'],
                'coordinates': cordinates
            }

            labels.append(label)
            min_pixels.append([x_mid, y_mid])

    return labels, min_pixels


def detect_from_gis_sequence(depth_images, color_images):
    i = 0
    while True:
        depth_image = depth_images[i]
        colour_image = color_images[i]
        # colour_image = np.array(cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB))
        depth_colormap = np.array(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        input_tensor = tf.convert_to_tensor(np.expand_dims(colour_image, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        image_np_with_detections_color = colour_image.copy()
        image_np_with_detections_depth = depth_colormap.copy()

        image_np_with_detections_color = draw_boxes(image_np_with_detections_color, detections)
        image_np_with_detections_depth = draw_boxes(image_np_with_detections_depth, detections)

        image_np_with_detections_depth, labels = get_distance_from_camera(depth_image,
                                                                          image_np_with_detections_depth,
                                                                          detections)

        images = np.hstack((image_np_with_detections_color, image_np_with_detections_depth))
        cv2.imshow('Object_Detection', images)
        i += 1

        if i >= len(colour_image):
            i = 0
        cv2.waitKey(1)
        time.sleep(0.5)


def detect_live_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()
        image_np_with_detections = draw_boxes(image_np_with_detections, detections)

        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def detect_from_depth_camera():
    camera = CameraSetup()
    try:
        while True:
            depth_frame, color_frame = camera.capture_frame()

            if not depth_frame and not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #
            intrinsics = camera.get_intrinsics(color_frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(color_image, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            if VARIABLES.METHOD == 'mask':
                image_shape = detections.pop('image_shape')
                num_proposals = detections.pop('num_proposals')
            detections = {key: value[0].numpy()
                          for key, value in detections.items()}

            detections['num_detections'] = num_detections
            if VARIABLES.METHOD == 'mask':
                detections['num_proposals'] = num_proposals
                detections['image_shape'] = image_shape

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections_color = color_image.copy()
            image_np_with_detections_depth = depth_colormap.copy()
            labels, mid_pixels = get_distance_from_camera(depth_image, detections, intrinsics)
            if mid_pixels:
                for pt in mid_pixels:
                    Visualisation.draw_circle(pt, image_np_with_detections_color)

                if len(mid_pixels) >= 2:
                    Visualisation.draw_arrow(mid_pixels, image_np_with_detections_color)

            detections['mean_cordinates'] = []
            [detections['mean_cordinates'].append(label['coordinates']) for label in labels]
            image_np_with_detections_color = draw_boxes(image_np_with_detections_color, detections,
                                                        detection_masks=False)
            image_np_with_detections_depth = draw_boxes(image_np_with_detections_depth, detections)

            # image_np_with_detections_depth, labels = (depth_image,
            #                                                                  image_np_with_detections_depth,
            #                                                                  detections)
            print(labels)

            images = np.hstack((image_np_with_detections_color, image_np_with_detections_depth))

            cv2.imshow('object detection', images)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        camera.stop_stream()


if __name__ == "__main__":
    VARIABLES = Variables()

    if not os.path.exists(VARIABLES.TFOD_FILES['LABEL_MAP']):
        CommonFunctions.create_label_map(VARIABLES.LABELS, VARIABLES.TFOD_FILES['LABEL_MAP'])
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(VARIABLES.TFOD_FILES['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(VARIABLES.TFOD_PATHS['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(VARIABLES.TFOD_FILES['LABEL_MAP'])

    # Read from image
    # image_path = os.path.join(VARIABLES.PATHS['TEST_PATH'], '0.png')
    # img = cv2.imread(image_path)
    # detect_from_image(img)

    # Detect from gis sequence
    # image_path = os.path.join(VARIABLES.PATHS['TEST_PATH'], '0.png')
    # col_images, dpt_images = CommonFunctions.read_gip_images(image_path)
    # detect_from_gis_sequence(dpt_images, col_images)

    # detect from live video rgb only
    # detect_live_camera()

    # detect from depth camera
    detect_from_depth_camera()
