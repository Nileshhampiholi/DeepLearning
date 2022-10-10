import os

from common_scripts import CommonFunctions
from common_scripts.Variables import Variables

if __name__ == "__main__":
    VARIABLES = Variables()

    # Create dirs
    for path in VARIABLES.TFOD_PATHS.values():
        CommonFunctions.make_dirs(path)

    # Clone TF Object detection models
    if not os.path.exists(os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'object_detection')):
        CommonFunctions.run_cmd(
            'git clone ' + VARIABLES.TFOD_REQUIREMENTS['TF_MODELS_URL'] + ' ' + VARIABLES.TFOD_PATHS['API_MODEL_PATH'])

    # Install TF Object detection
    if not os.listdir(VARIABLES.TFOD_PATHS['PROTOC_PATH']):
        CommonFunctions.download_files(VARIABLES.TFOD_REQUIREMENTS['PORTO_BUF_URL'],
                                       VARIABLES.TFOD_PATHS['PROTOC_PATH'])
        CommonFunctions.unzip_files(VARIABLES.TFOD_PATHS['PROTOC_PATH'])
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(VARIABLES.TFOD_PATHS['PROTOC_PATH'], 'bin'))
        CommonFunctions.run_cmd('cd {path} & protoc {destination_path} --python_out=.'.format(
            path=os.path.join(VARIABLES.PARENT_DIR, 'obj_detection_mask_rcnn', 'Tensorflow', 'models', 'research'),
            destination_path=os.path.join("object_detection", "protos", "*.proto")))
        CommonFunctions.copy_files(
            os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'object_detection', 'packages', 'tf2',
                         'setup.py'),
            os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research'))
        CommonFunctions.run_cmd('{python_path} {file_path} {build_path} build'.format(
            python_path=VARIABLES.PATH_TO_ENV,
            file_path=os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'setup.py'),
            build_path=os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research')))
        CommonFunctions.run_cmd('{python_path} {file_path} {install_path} install'.format(
            python_path=VARIABLES.PATH_TO_ENV,
            file_path=os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'setup.py'),
            install_path=os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research')))
        CommonFunctions.pip_install(os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'slim') + ' e')
        CommonFunctions.pip_install(os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research') + ' .')
    CommonFunctions.run_cmd(VARIABLES.PATH_TO_ENV + ' ' +
                            os.path.join(VARIABLES.TFOD_PATHS['API_MODEL_PATH'], 'research', 'object_detection',
                                         'builders',
                                         'model_builder_tf2_test.py'))

    # Download Pre-trained model
    if not os.listdir(VARIABLES.TFOD_PATHS['PRETRAINED_MODEL_PATH']):
        CommonFunctions.download_files(VARIABLES.TFOD_MODEL['PRETRAINED_MODEL_URL'],
                                       VARIABLES.TFOD_PATHS['PRETRAINED_MODEL_PATH'])
        CommonFunctions.unzip_files(VARIABLES.TFOD_PATHS['PRETRAINED_MODEL_PATH'])

    # Download script to create TF records
    if not os.path.exists(VARIABLES.TFOD_FILES['TF_RECORD_SCRIPT']):
        CommonFunctions.run_cmd(
            'git clone {url} {path_to_script}'.format(
                url=VARIABLES.TFOD_REQUIREMENTS['PATH_TO_TF_RECORD'],
                path_to_script=VARIABLES.TFOD_PATHS['SCRIPTS_PATH']))

    # Download the script to convert from json
    if not os.path.exists(VARIABLES.TFOD_FILES['TF_COCO_RECORD']):
        CommonFunctions.download_files(VARIABLES.TFOD_REQUIREMENTS['PATH_TO_COCO_RECORD'],
                                       VARIABLES.TFOD_PATHS['SCRIPTS_PATH'])
