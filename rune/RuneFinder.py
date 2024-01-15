import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from rune.module.runeDetect import rune_detect 
from msg_pkg.msg import RuneCorner

class ImageSubscriber(Node):

    def __init__(self):

        super().__init__('RuneFinder')
        self.subscription = self.create_subscription(
            CompressedImage,
            'ImageSend',
            self.listener_callback,
            10)
        self.subscription
        # 创建一个发布器来发布IntList类型的消息
        self.publisher = self.create_publisher(RuneCorner, 'Receive_corner', 10)

        modelPath="/workspaces/vscode_ros2_workspace/src/rune/rune/module/best.engine"
        targetColor=0
        self.model = rune_detect(modelPath, targetColor)


    def listener_callback(self, msg):
        # 将接收到的数据转换为一个 NumPy 数组
        np_arr = np.frombuffer(msg.data, np.uint8)
        
        # 解码图像
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        ansList=self.model.detect(image_np)
        ansList2=[]
        for item in ansList:#二元tuple转成单个数据
            #print(type(item[0]))
            #print(type(item[1]))
            ansList2.append(int(item[0]))
            ansList2.append(int(item[1]))
        # 现在，image_np 是一个 OpenCV 格式的图像
        msg = RuneCorner()
    
        msg.data = ansList2  # 示例数据
        self.publisher.publish(msg)
        self.get_logger().info('Published IntList message')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
