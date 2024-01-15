import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import os
from cv_bridge import CvBridge

class ImagePublisher(Node):

    def __init__(self, folder_path):
        super().__init__('testImageSender')
        self.publisher = self.create_publisher(CompressedImage, 'ImageSend', 10)
        self.folder_path = folder_path
        self.bridge = CvBridge()

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        for filename in os.listdir(self.folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(self.folder_path, filename)
            cv_image = cv2.imread(image_path)

            if cv_image is not None:
                compressed_image = self.bridge.cv2_to_compressed_imgmsg(cv_image)
                self.publisher.publish(compressed_image)
                self.get_logger().info(f'Published image: {filename}')
            else:
                self.get_logger().error(f'Failed to read image: {filename}')

def main(args=None):
    rclpy.init(args=args)

    image_publisher = ImagePublisher(folder_path='/workspaces/vscode_ros2_workspace/src/rune/images')
    rclpy.spin(image_publisher)

    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
