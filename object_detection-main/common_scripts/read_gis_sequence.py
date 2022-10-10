import gip.gip_io.gip_image as gip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

color_image_path = r"path\X0.gip"

gip_image = gip.read_image(color_image_path)

img = Image.fromarray(np.uint8(gip_image.pixel_data))

img.show()

sequence_file = r"path\0.gis"


def read_sequence(sequence_file):
    sequence = gip.read_sequence(sequence_file)

    for i in range(0, len(sequence), 2):
        depth_image = sequence[i].pixel_data
        color_image = sequence[i + 1].pixel_data
        plt.figure(1)
        plt.imshow(color_image)

        plt.figure(2)
        plt.imshow(depth_image)
        plt.show()
