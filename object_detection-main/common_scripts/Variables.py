import os


class Variables(object):
    LABELS = [{'name': 'person', 'id': 1}, {'name': 'detector', 'id': 2}, {'name': 'xray_tube', 'id': 3}]
    IMAGES_PATH = os.path.join('data_samples', 'collected_images')
    NUM_SAMPLES = 20
    TRAIN_PATH = os.path.join('data_samples', 'train_data')
    TEST_PATH = os.path.join('data_samples', 'test_data')
    SPLIT_PERCENTAGE = 0.4
    AUTOMATIC = True
    PATH_TO_ENV = r'C:\Users\320184682\PycharmProjects\venv\Scripts\python.exe'
    PARENT_DIR = os.path.abspath('..')
    METHOD = 'no_mask'
    if METHOD == 'mask':
        TFOD_MODEL = {
            'CUSTOM_MODEL_NAME': 'my_mask-rcnn',
            'PRETRAINED_MODEL_NAME': 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
            'PRETRAINED_MODEL_URL': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
                                    'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz'
        }
    else:
        TFOD_MODEL = {
            'CUSTOM_MODEL_NAME': 'my_ssd_mobnet_20000_tuned',
            'PRETRAINED_MODEL_NAME': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
            'PRETRAINED_MODEL_URL': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
                                    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        }

    TFOD_REQUIREMENTS = {
        'COMPILER_URL': "https://github.com/protocolbuffers/protobuf/releases/download/"
                        "v3.15.6/protoc-3.15.6-win64.zip",
        'TF_MODELS_URL': "https://github.com/tensorflow/models",
        'PORTO_BUF_URL': "https://github.com/protocolbuffers/protobuf/releases/"
                         "download/v3.15.6/protoc-3.15.6-win64.zip",
        'PATH_TO_TF_RECORD': 'https://github.com/nicknochnack/GenerateTFRecord',
        'PATH_TO_COCO_RECORD': "https://raw.githubusercontent.com/TannerGilbert/"
                             "Tensorflow-Object-Detection-API-train-custom-Mask-R-CNN-model/"
                             "master/create_coco_tf_record.py"
    }

    TFOD_SCRIPT_NAMES = {
        'TF_RECORD_SCRIPT_NAME': 'generate_tfrecord.py',
        'TF_RECORD_COCO_NAME': "create_coco_tf_record.py",
        'LABEL_MAP_NAME': 'label_map.pbtxt'
    }

    TFOD_PATHS = {
        'HOME': os.getcwd(),
        'WORKSPACE_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'scripts'),
        'API_MODEL_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'models'),
        'IMAGE_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'images'),
        'ANNOTATION_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace',
                                        'annotations'),
        'MODEL_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace',
                                              'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'models',
                                        TFOD_MODEL['CUSTOM_MODEL_NAME']),
        'OUTPUT_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'models',
                                    TFOD_MODEL['CUSTOM_MODEL_NAME'], 'export'),
        'TFJS_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'models',
                                  TFOD_MODEL['CUSTOM_MODEL_NAME'], 'tfjsexport'),
        'TFLITE_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'models',
                                    TFOD_MODEL['CUSTOM_MODEL_NAME'], 'tfliteexport'),
        'PROTOC_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'protoc'),
        'TRAIN_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'images', 'train'),
        'TEST_PATH': os.path.join(PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'workspace', 'images', 'test')
    }

    TFOD_FILES = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', TFOD_MODEL['CUSTOM_MODEL_NAME'],
                                        'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(TFOD_PATHS['SCRIPTS_PATH'], TFOD_SCRIPT_NAMES['TF_RECORD_SCRIPT_NAME']),
        'TF_COCO_RECORD': os.path.join(TFOD_PATHS['SCRIPTS_PATH'], TFOD_SCRIPT_NAMES['TF_RECORD_COCO_NAME']),
        'LABEL_MAP': os.path.join(TFOD_PATHS['ANNOTATION_PATH'], TFOD_SCRIPT_NAMES['LABEL_MAP_NAME'])
    }

    TFOD_SCRIPTS = {
        'TRAINING_SCRIPT': os.path.join(TFOD_PATHS['API_MODEL_PATH'], 'research', 'object_detection',
                                        'model_main_tf2.py')
    }
