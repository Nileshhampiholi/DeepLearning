# Set headless-friendly backend.
import collections

import PIL.ImageColor as ImageColor
import cv2
# Set headless-friendly backend.
import matplotlib
import numpy as np
import six
from PIL import Image, ImageDraw, ImageFont
from six.moves import range

matplotlib.use('Agg')  # pylint: disable=multiple-statements

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """
    Draws mask on an image.
    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: A uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the key_points with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)
    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * (mask > 0))).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(np.uint8(pil_image).convert('RGB')))


def draw_bounding_box_on_image(image,
                               y_min,
                               x_min,
                               y_max,
                               x_max,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      y_min: y_min of bounding box.
      x_min: x_min of bounding box.
      y_max: y_max of bounding box.
      x_max: x_max of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        y_min, x_min, y_max, x_max as relative to the image. Otherwise, treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (x_min * im_width, x_max * im_width,
                                      y_min * im_height, y_max * im_height)
    else:
        (left, right, top, bottom) = (x_min, x_max, y_min, y_max)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width=thickness,
                  fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(((left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)), fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array(image,
                                     y_min,
                                     x_min,
                                     y_max,
                                     x_max,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    """
      Adds a bounding box to an image (numpy array).

      Bounding box coordinates can be specified in either absolute (pixel) or
      normalized coordinates by setting the use_normalized_coordinates argument.

      Args:
        image: a numpy array with shape [height, width, 3].
        y_min: y_min of bounding box.
        x_min: x_min of bounding box.
        y_max: y_max of bounding box.
        x_max: x_max of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                          (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
          y_min, x_min, y_max, x_max as relative to the image. Otherwise, treat
          coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, y_min, x_min, y_max, x_max, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_circle(pt, image):
    cv2.circle(image, (pt[0], pt[1]), 8, (0, 0, 255), -1)
    return image


def draw_arrow(points, image):
    cv2.arrowedLine(image, points[-1], points[-2], (0, 0, 255), 1)
    return image


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        mean_coordinates=None,
        instance_masks=None,
        instance_boundaries=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        mask_alpha=.4,
        groundtruth_box_visualization_color='black'):
    """
    Overlay labeled boxes on an image with formatted scores and label names.
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      mean_coordinates : coordinates
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: A uint8 numpy array of shape [N, image_height, image_width],
        can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      use_normalized_coordinates: whether boxes are to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box or keypoint to be
        visualized.
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      mask_alpha: transparency value between 0 and 1 (default: 0.4).
      groundtruth_box_visualization_color: box color for visualizing groundtruth box

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not agnostic_mode:
                    if classes[i] in six.viewkeys(category_index):
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)

                if not display_str:
                    display_str = '{}%'.format(round(100 * scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, round(100 * scores[i]))

                if mean_coordinates:
                    display_str = '{} \n{}: {}'.format(display_str, 'coordinate', mean_coordinates[i])

                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        y_min, x_min, y_max, x_max = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_masks_map[box],
                color=color,
                alpha=mask_alpha
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image,
                box_to_instance_boundaries_map[box],
                color='red',
                alpha=1.0
            )
        draw_bounding_box_on_image_array(
            image,
            y_min,
            x_min,
            y_max,
            x_max,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
    return image


def resize_predictions_detection_masks(predictions=None, image_shape=('height', 'width', 'channels'), mask_threshold=0.,
                                       verbose=1, debug=1):
    detection_masks = []

    for idx, detection_mask in enumerate(predictions['detection_masks']):
        height, width, _ = image_shape
        # background_mask with all black=0
        mask = np.zeros((height, width))

        # Get normalised bbox coordinates
        y_min, x_min, y_max, x_max = predictions['detection_boxes'][idx]

        # Convert to image size fixed bbox coordinates
        y_min, x_min, y_max, x_max = int(y_min * height), int(x_min * width), int(y_max * height), int(x_max * width)

        # Define bbox height and width
        bbox_height, bbox_width = y_max - y_min, x_max - x_min

        # Resize 'detection_mask' to bbox size
        bbox_mask = Image.fromarray(np.uint8(np.array(detection_mask) * 255), mode='L').resize(
            size=(bbox_width, bbox_height), resample=Image.NEAREST)

        # Insert detection_mask into image.size np.zeros((height, width)) background_mask
        mask[y_min:y_max, x_min:x_max] = bbox_mask
        mask_threshold = mask_threshold
        mask = np.where(mask > (mask_threshold * 255), 1, mask)
        # mask_threshold > 0.5 resulting mask seems too coarse, any value > 0 seems resulting in better masks
        # in case threshold is used to have other values (0)
        if mask_threshold > 0:
            mask = np.where(mask != 1, 0, mask)

        if debug:
            try:
                assert (
                        (np.unique(mask) == np.array([0., 1.])).all()  # in case, bbox_mask has (0,1)
                        or
                        (np.unique(mask) == np.array([0.])).all()  # in case, bbox_mask resulted in only (0)
                )
            except Exception as e:
                print(e)
                print('Mask Index:', idx)
                print('Mask unique values: ', np.unique(mask))
                print('Expected unique values: ', np.array([0., 1.]))
                break

        # Verbose first detection_mask
        if (verbose or debug) and idx == 0:
            print('Index (Example): ', idx)
            print('detection_mask shape():', np.array(detection_mask).shape)
            print('detection_boxes:', predictions['detection_boxes'][idx])
            print('detection_boxes@image.size (y_min, x_min, y_max, x_max) :', [y_min, x_min, y_max, x_max])
            print('detection_boxes (height, width):', (bbox_height, bbox_width))

        # Append mask to list() 'detection_masks'
        detection_masks.append(mask.astype(np.uint8))

    return detection_masks
