import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import argparse

def infer(model, image):
    # Run inference on the image using the YOLO model
    results = model.predict(source=image, save=False)
    return results

class YoloInferenceNode(Node):
    def __init__(self, model_path, image_topic):
        super().__init__('yolo_inference_node')
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        
        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10)
        
        # Publishers for detections and masks
        self.detections_pub = self.create_publisher(String, 'detections', 10)
        self.mask_pub = self.create_publisher(Image, 'mask', 10)
    
    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run YOLO inference
        results = infer(self.model, cv_image)
        
        # Process and publish results
        for result in results:
            boxes = result.boxes
            masks = result.masks

            # Prepare detection information
            detection_str = ''
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    detection_str += f'Class: {class_id}, Confidence: {confidence}, Box: [{x1}, {y1}, {x2}, {y2}]\n'

                detections_msg = String()
                detections_msg.data = detection_str
                self.detections_pub.publish(detections_msg)
            # Render image with detections and publish as mask
            rendered_image = result.plot()
            mask_msg = self.bridge.cv2_to_imgmsg(rendered_image, encoding='bgr8')
            self.mask_pub.publish(mask_msg)

def main(args=None):
    parser = argparse.ArgumentParser(description='Run YOLO inference on ROS image topic')
    parser.add_argument("--model_path", type=str, default='yolov8n.pt', help="Path to YOLO model")
    parser.add_argument("--image_topic", type=str, default='/camera/image_raw', help="ROS2 image topic to subscribe to")
    parsed_args = parser.parse_args(args=args)

    rclpy.init(args=args)
    node = YoloInferenceNode(parsed_args.model_path, parsed_args.image_topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
