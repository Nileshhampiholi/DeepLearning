from common_scripts import CommonFunctions
from common_scripts.Variables import Variables
from data_acquisition.CaptureImages import CaptureImages
import keyboard

if __name__ == "__main__":
    while True:
        VARIABLES = Variables()
        CommonFunctions.make_dirs()

        capture_images = CaptureImages()
        capture_images.capture_image_bib()

        if keyboard.is_pressed("q"):
            print("q pressed, ending loop")
            break
