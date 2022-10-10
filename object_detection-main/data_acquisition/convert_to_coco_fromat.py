import glob
import os
from Labelme2Coco import Labelme2Coco
from coco_data_set.download_coco_data_set import MergeCocoAnnotation
from Variables import Variables

if __name__ == "__main__":
    list_json = glob.glob(os.path.join(Variables.IMAGES_PATH, 'recorded_test', "*.json"))
    Labelme2Coco(list_json, save_json_path=os.path.join(Variables.IMAGES_PATH, 'recorded_test' + '.json'))

    list_json = glob.glob(os.path.join(Variables.IMAGES_PATH, 'recorded_train', "*.json"))
    Labelme2Coco(list_json, save_json_path=os.path.join(Variables.IMAGES_PATH, 'recorded_train' + '.json'))

    # Merge Coco Annotation and recorded Annotations
    MergeCocoAnnotation(os.path.join(Variables.IMAGES_PATH, "recorded_test.json"),
                        os.path.join(Variables.IMAGES_PATH, 'coco_test.json'),
                        path=os.path.join(Variables.IMAGES_PATH, 'merged_test.json'))

    MergeCocoAnnotation(os.path.join(Variables.IMAGES_PATH, "recorded_train.json"),
                        os.path.join(Variables.IMAGES_PATH, 'coco_train.json'),
                        path=os.path.join(Variables.IMAGES_PATH, 'merged_train.json'))
