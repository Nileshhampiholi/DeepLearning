import cv2
import numpy as np
import pyrealsense2 as rs


class CameraSetup:
    def __init__(self):
        self.devices = None
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._check_connection()
        self._start_stream()
        # self.color_profile, self.depth_profile = self._get_profiles()

    def _check_connection(self):
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        pipeline_profile = self._config.resolve(pipeline_wrapper)
        self.devices = pipeline_profile.get_device()

    def _start_stream(self):
        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(self._config)

    def _get_profiles(self):
        color_profiles = []
        depth_profiles = []
        for device in self.devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print('Sensor: {}, {}'.format(name, serial))
            print('Supported video formats:')
            for sensor in device.query_sensors():
                for stream_profile in sensor.get_stream_profiles():
                    stream_type = str(stream_profile.stream_type())

                    if stream_type in ['stream.color', 'stream.depth']:
                        v_profile = stream_profile.as_video_stream_profile()
                        fmt = stream_profile.format()
                        w, h = v_profile.width(), v_profile.height()
                        fps = v_profile.fps()

                        video_type = stream_type.split('.')[-1]
                        print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                            video_type, w, h, fps, fmt))
                        if video_type == 'color':
                            color_profiles.append((w, h, fps, fmt))
                        else:
                            depth_profiles.append((w, h, fps, fmt))

        return color_profiles, depth_profiles

    @staticmethod
    def get_intrinsics(frame):
        return frame.profile.as_video_stream_profile().intrinsics

    @staticmethod
    def get_extrinsics(depth_frame, color_frame):
        return depth_frame.profile.get_extrinsics_to(color_frame.profile)

    def video_stream(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self._pipeline.wait_for_frames(timeout_ms=20000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Show images
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def capture_frame(self):
        frames = self._pipeline.wait_for_frames(timeout_ms=20000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    def stop_stream(self):
        self._pipeline.stop()
