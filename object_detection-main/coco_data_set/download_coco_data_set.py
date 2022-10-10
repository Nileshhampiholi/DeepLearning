from pycocotools.coco import COCO
from urllib.request import urlopen
from PIL import Image, ImageDraw
import os
from pascal_voc_writer import Writer

# Show image
show_image = True

# Path to annotation files
data_type = 'val2017'
annotation_file = os.path.join('annotations', "instances_" + data_type + '.json')

# Path to store images and labels
path_to_image_dir = "images"
class_label = ["person"]
path_to_image_class = os.path.join(path_to_image_dir, class_label[0])

if not os.path.exists(path_to_image_class):
    os.mkdir(path_to_image_class)

# Load annotation file
coco = COCO(annotation_file)

# get all images containing given class labels
category_ids = coco.getCatIds(catNms=class_label)
image_ids = coco.getImgIds(catIds=category_ids)
print("Number of images to down_load : ", len(image_ids))
for im_id in range(10):  # len(image_ids)):
    try:
        print("Number of images downloaded  = %d" % im_id)
        # Read image data
        image_data = coco.loadImgs(image_ids[im_id])[0]
        image_url = image_data['coco_url']
        width = image_data["width"]
        height = image_data["height"]
        file_name = image_data["file_name"].split(".")[0]

        # Get annotations
        annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)

        # Load image
        image = Image.open(urlopen(image_url))
        draw = ImageDraw.Draw(image)

        # Write image to local folder
        path_to_image = os.path.join(path_to_image_class, file_name + ".jpg")

        # Write the annotations in Pascal Visual Object Classes(VOC) format
        path_to_xml = os.path.join(path_to_image_class, file_name + ".xml")
        writer = Writer(path_to_image, width, height)
        for object_annotation in annotations:
            bounding_box_coco_style = object_annotation['bbox']
            bounding_box_voc_style = [bounding_box_coco_style[0], bounding_box_coco_style[1],
                                      bounding_box_coco_style[0] + bounding_box_coco_style[2],
                                      bounding_box_coco_style[1] + bounding_box_coco_style[3]]
            writer.addObject('person', bounding_box_voc_style[0], bounding_box_voc_style[1],
                             bounding_box_voc_style[2], bounding_box_voc_style[3])
            if show_image:
                # Plot the bounding box
                draw.rectangle(bounding_box_voc_style, width=2)

        if show_image:
            image.show()

        if not os.path.exists(path_to_image):
            image.save(path_to_image)
            writer.save(path_to_xml)
    except Exception as e:
        print(e)
        pass
