from inspect import EndOfBlock
#帧率过低，后续修改
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
from cv_bridge import CvBridge

class ImagePublisher(Node):

    def __init__(self, folder_path):
        super().__init__('testImageSender')
        self.publisher = self.create_publisher(Image, 'ImageSend', 10)
        self.folder_path = folder_path
        self.bridge = CvBridge()
        self.timer_callback()
        #self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        for filename in os.listdir(self.folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(self.folder_path, filename)
            cv_image = cv2.imread(image_path)

            if cv_image is not None:
                msg = self.bridge.cv2_to_imgmsg(cv_image,encoding="bgr8")
                while True:
                    
                    self.publisher.publish(msg)
                    #self.get_logger().info(f'We are Publishing image: {filename}')
            else:
                self.get_logger().error(f'OHHHHHHHH!Failed to read image: {filename}')

def main(args=None):
    rclpy.init(args=args)

    image_publisher = ImagePublisher(folder_path='/workspaces/vscode_ros2_workspace/src/rune/images')
    rclpy.spin(image_publisher)

    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
