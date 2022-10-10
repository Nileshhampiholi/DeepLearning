import glob
import os
from Variables import Variables
import CommonFunctions
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from labellme_to_coco import labelme2coco

if __name__ == "__main__":
    VARIABLES = Variables(mask_rcnn=False)
    #  Create Label map
    CommonFunctions.create_label_map(VARIABLES.LABELS, VARIABLES.FILES['LABEL_MAP'])

    # Convert from label me to coco format
    train_json = glob.glob(os.path.join(VARIABLES.PATHS['TRAIN_PATH'], "*.json"))
    labelme2coco(train_json, os.path.join(VARIABLES.PATHS['IMAGE_PATH'], 'train.json'))
    test_json = glob.glob(os.path.join(VARIABLES.PATHS['TEST_PATH'], "*.json"))
    labelme2coco(test_json, os.path.join(VARIABLES.PATHS['IMAGE_PATH'], 'test.json'))

    # Generate tf records from json files
    command = '{python} {script} --logtostderr \
                --train_image_dir={train_dir} \
                --test_image_dir= {test_dir} \
                --train_annotations_file={train_annotation} \
                --test_annotations_file={test_annotation}" \
                --output_dir={output_dir}'.format(
                python=VARIABLES.PATH_TO_ENV,
                script=VARIABLES.FILES['TF_COCO_RECORD'],
                train_dir=VARIABLES.PATHS['TRAIN_PATH'],
                test_dir=VARIABLES.PATHS['TEST_PATH'],
                train_annotation=os.path.join(VARIABLES.PATHS['IMAGE_PATH'], 'train.json'),
                test_annotation=os.path.join(VARIABLES.PATHS['IMAGE_PATH'], 'train.json'),
                output_dir=VARIABLES.PATHS['ANNOTATION_PATH'])
    CommonFunctions.run_cmd(command)

    # Generate TF records from xml
    CommonFunctions.run_cmd(
        '{python} {script} -x {train_path} -l {label_map} -o {annotation_path} train.record').format(
        python=VARIABLES.PATH_TO_ENV,
        train_path=VARIABLES.PATHS['TRAIN_PATH'],
        label_map=VARIABLES.FILES['LABEL_MAP'],
        annotation_path=VARIABLES.PATHS['ANNOTATION_PATH']
    )
    CommonFunctions.run_cmd(
        '{python} {script} -x {test_path} -l {label_map} -o {annotation_path} test.record').format(
        python=VARIABLES.PATH_TO_ENV,
        test_path=VARIABLES.PATHS['TEST_PATH'],
        label_map=VARIABLES.FILES['LABEL_MAP'],
        annotation_path=VARIABLES.PATHS['ANNOTATION_PATH']
    )

    # Copy config file and edit
    CommonFunctions.copy_files(
        os.path.join(VARIABLES.PATHS['PRETRAINED_MODEL_PATH'], VARIABLES.PRETRAINED_MODEL_NAME, 'pipeline.config'),
        VARIABLES.PATHS['CHECKPOINT_PATH'])

    # open file
    config = config_util.get_configs_from_pipeline_file(VARIABLES.FILES['PIPELINE_CONFIG'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(VARIABLES.FILES['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Change variables
    pipeline_config.model.ssd.num_classes = len(VARIABLES.LABELS)
    pipeline_config.train_config.batch_size = 10
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(VARIABLES.PATHS['PRETRAINED_MODEL_PATH'],
                                                                     VARIABLES.PRETRAINED_MODEL_NAME, 'checkpoint',
                                                                     'ckpt-11')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = VARIABLES.FILES['LABEL_MAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(VARIABLES.PATHS['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = VARIABLES.FILES['LABEL_MAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(VARIABLES.PATHS['ANNOTATION_PATH'], 'test.record')]

    # write file
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(VARIABLES.FILES['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)

    # train network
    '''
    command = "{} {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(
        VARIABLES.PATH_TO_ENV, VARIABLES.SCRIPTS['TRAINING_SCRIPT'], VARIABLES.PATHS['CHECKPOINT_PATH'],
        VARIABLES.FILES['PIPELINE_CONFIG'])

    CommonFunctions.run_cmd(command)

    # evaluate the network
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(
        VARIABLES.PATH_TO_ENV,
        VARIABLES.SCRIPTS['TRAINING_SCRIPT'],
        VARIABLES.FILES['PIPELINE_CONFIG'],
        VARIABLES.PATHS['CHECKPOINT_PATH'])
    '''
