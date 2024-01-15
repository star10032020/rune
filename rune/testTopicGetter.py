import rclpy
from rclpy.node import Node
from msg_pkg.msg import RuneCorner

class CornerSubscriber(Node):

    def __init__(self):
        super().__init__('testTopicGetter')
        self.subscription = self.create_subscription(
            RuneCorner,
            'Receive_corner',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'Received IntList: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    corner_subscriber = CornerSubscriber()
    rclpy.spin(corner_subscriber)

    corner_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
