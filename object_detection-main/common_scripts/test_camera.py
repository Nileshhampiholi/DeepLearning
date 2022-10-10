import cv2
import numpy as np
import pyrealsense2 as rs


def get_3d_distance(pt1, pt2):
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2, axis=0))


def mouseRGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        points.append([x, y])


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
align_to_color = rs.align(rs.stream.color)

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense', mouseRGB)
points = []
coordinates = []
coordinates_b = []
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsic
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        # print(depth_intrin.ppx, depth_intrin.ppy, depth_intrin.fx, depth_intrin.fy, depth_intrin.width,
        #      depth_intrin.height)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        for pt in points[-2:]:
            point_z = np.round(depth_image[pt[1], pt[0]] * 0.001, 2)  # distance in meters
            point_x = np.round(((pt[0] + color_intrin.width / 2) - color_intrin.ppx) * point_z * (1 / color_intrin.fx),
                               2)
            point_y = np.round(((pt[1] + color_intrin.height / 2) - color_intrin.ppy) * point_z * (1 / color_intrin.fy),
                               2)
            a = [point_x, point_y, point_z]
            coordinates.append(a)
            z1 = depth_image[pt[1], pt[0]]
            x1 = np.round(pt[0] * z1 / 55, 2)
            y1 = np.round(pt[1] * z1 / 55, 2)
            cv2.circle(color_image, (pt[0], pt[1]), 8, (0, 0, 255), -1)
            coordinates_b.append([x1, y1, z1])
            cv2.putText(color_image, '{} mm'.format(a), (pt[0] - 10, pt[1] + 20), 0, 0.5, (0, 0, 255), 1)

        if len(points) >= 2:
            cv2.arrowedLine(color_image, points[-1], points[-2], (0, 0, 255), 1)
            dist = get_3d_distance(coordinates_b[-1], coordinates_b[-2])
            cv2.putText(color_image, 'dist = {} mm'.format(np.round(dist, 2)), (200, 100), 0, 0.5, (0, 0, 255), 1)

        if len(points) >= 10:
            points = []
            coordinates = []
            coordinates_b = []
        '''
        z1 = depth_image[pt[1], 200]
        x1 = np.round(50 * z1 / 55, 2)
        y1 = np.round(200 * z1 / 55, 2)
        cv2.circle(color_image, (50, 200), 8, (0, 0, 255), -1)
        cv2.putText(color_image, '{} mm'.format([x1, y1, z1]), (50 - 30, 200 + 30), 0, 0.5, (0, 0, 255), 1)
        
        z2 = depth_image[320, 200]
        x2 = np.round(350 * z2 / 55, 2)
        y2 = np.round(200 * z2 / 55, 2)
        cv2.circle(color_image, (350, 200), 8, (0, 0, 255), -1)
        cv2.putText(color_image, '{} mm'.format([x2, y2, z2]), (350 - 20, 200 - 30), 0, 0.5, (0, 0, 255), 1)
        # print([x1, y1, z1], [x2, y2, z2])
        dist = get_3d_distance([x1, y1, z1], [x2, y2, z2])
        cv2.putText(color_image, '{} mm'.format(np.round(dist, 2)), (200, 100), 0, 0.5, (0, 0, 255), 1)
        '''
        list_images = [color_image, depth_image]

        depth1 = depth_frame.get_distance(200, 50)

        depth_point1 = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [200, 50], depth1)

        depth2 = depth_frame.get_distance(200, 345)

        depth_point2 = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [200, 345], depth2)
        # print(get_3d_distance(depth_point1, depth_point2))

        # tags = {"tag_1": "colour_image", "tag_2": "depth_image"}
        # print(np.shape(color_image))
        # print(np.shape(depth_image))
        # gip_image = gip.GipImage(depth_image, {"tag": "depth"})
        # gip.write_image(gip_image,
        #                r"C:\Users\320184682\PycharmProjects\real_time_object_detection\data_acquisition\data_samples\depth_image.gip")
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        if cv2.waitKey(1) == 27:
            break


finally:

    # Stop streaming
    pipeline.stop()
