import os
import time
import uuid
import cv2
import gip.gip_io.gip_image as gip
import numpy as np
from common_scripts.CameraSetup import CameraSetup
from common_scripts.Variables import Variables


class CaptureImages(Variables, CameraSetup):

    def capture_image_cv2(self):
        cap = cv2.VideoCapture(1)
        for label in self.LABELS:
            print('Collecting images for {}'.format(label))
            if self.AUTOMATIC:
                time.sleep(2)
            for img_num in range(self.NUM_SAMPLES):
                print('Collecting image {}'.format(img_num))

                ret, frame = cap.read()
                img_name = os.path.join(self.IMAGES_PATH, label,
                                        label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(img_name, frame)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', frame)

                if self.AUTOMATIC:
                    time.sleep(2)
                else:
                    input("Press enter to continue")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

    def capture_image_bib(self):
        for label in self.LABELS:
            gis_sequence = []
            print('Collecting images for {}'.format(label))
            if self.AUTOMATIC:
                time.sleep(2)
            for img_num in range(self.NUM_SAMPLES):
                print('Collecting image {}'.format(img_num))

                depth_frame, color_frame = self.capture_frame()

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                colour_gip_image = gip.GipImage(color_image, {})
                depth_gip_image = gip.GipImage(depth_image, {})

                gis_sequence.append(depth_gip_image)
                gis_sequence.append(colour_gip_image)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                images = np.hstack((color_image, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

                if self.AUTOMATIC:
                    time.sleep(2)
                else:
                    input("Press enter to continue")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            img_name = os.path.join(self.IMAGES_PATH, label,
                                    label + '.' + '{}.gis'.format(str(uuid.uuid1())))
            gip.write_sequence(gis_sequence, img_name)

        cv2.destroyAllWindows()
