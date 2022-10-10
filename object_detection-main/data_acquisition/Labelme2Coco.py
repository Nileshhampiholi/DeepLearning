import os
import argparse
import json
from labelme import utils
import numpy as np
import glob
from PIL import Image, ImageDraw


class Labelme2Coco(object):
    def __init__(self, list_json=None, save_json_path="./coco.json"):
        """
        param list_json: the list of all labelme json file paths
        param save_json_path: the path to save new json
        """
        self.data_coco = None
        self.labelme_json = list_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person"
            },
            {
                "supercategory": "detector",
                "id": 2,
                "name": "detector"
            },
            {
                "supercategory": "xray",
                "id": 3,
                "name": "xray"
            }
        ]
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            # print(num, json_file)
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.get_bbox(points)))

        annotation["category_id"] = self.getcat_id(label[0])
        annotation["id"] = self.annID
        return annotation

    def getcat_id(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def get_bbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask_2_box(mask)

    @staticmethod
    def mask_2_box(mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    @staticmethod
    def polygons_to_mask(img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {"images": self.images, "categories": self.categories, "annotations": self.annotations}
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    Labelme2Coco(labelme_json, args.output)
