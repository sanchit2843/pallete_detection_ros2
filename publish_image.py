import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse

class ImagePublisherNode(Node):
    def __init__(self, image_path, image_topic):
        super().__init__('image_publisher_node')
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, image_topic, 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.image_path = image_path
        self.cv_image = cv2.imread(self.image_path)
        
        if self.cv_image is None:
            self.get_logger().error(f'Failed to read image from {self.image_path}')
            rclpy.shutdown()
        
    def timer_callback(self):
        image_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        self.publisher_.publish(image_msg)
        self.get_logger().info(f'Published image from {self.image_path}')

def main(args=None):
    parser = argparse.ArgumentParser(description='Publish an image to a ROS topic')
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--image_topic", type=str, default='/camera/image_raw', help="ROS2 image topic to publish to")
    parsed_args = parser.parse_args(args=args)

    rclpy.init(args=args)
    node = ImagePublisherNode(parsed_args.image_path, parsed_args.image_topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
