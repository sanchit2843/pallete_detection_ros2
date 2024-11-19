import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)
from cv_bridge import CvBridge
import argparse
import numpy as np


def infer(model, image):
    # Run inference on the image using the YOLO model
    results = model.predict(source=image, save=False)
    return results


class YoloInferenceNode(Node):
    def __init__(self, model_path, image_topic):
        super().__init__("yolo_inference_node")
        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        # Publishers for detections and masks
        self.detections_pub = self.create_publisher(Detection2DArray, "detections", 10)
        self.mask_pub = self.create_publisher(Image, "mask", 10)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Run YOLO inference
        results = infer(self.model, cv_image)

        # Initialize Detection2DArray message
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = (
            msg.header
        )  # Use the same header as the input image

        # Process and publish results
        for result in results:
            # Detection
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            confidences = result.boxes.conf.cpu().numpy()  # (N, 1)
            class_ids = result.boxes.cls.cpu().numpy()  # (N, 1)
            # print("class_ids",class_ids)
            # Image dimensions
            img_height, img_width = cv_image.shape[:2]
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                confidence = float(confidences[i])
                class_id = int(class_ids[i])

                # Create Detection2D message
                detection_msg = Detection2D()
                detection_msg.header = msg.header  # Use the same timestamp and frame ID

                # Fill in results
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = str(class_id)
                hypothesis.score = float(confidence)
                detection_msg.results.append(hypothesis)

                # Bounding box
                bbox = BoundingBox2D()
                bbox.center.x = (x1 + x2) / 2.0
                bbox.center.y = (y1 + y2) / 2.0
                bbox.size_x = float(x2 - x1)
                bbox.size_y = float(y2 - y1)
                detection_msg.bbox = bbox

                # Append detection to the array
                detection_array_msg.detections.append(detection_msg)

        # Publish detections
        self.detections_pub.publish(detection_array_msg)

        # Segmentation masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            # Apply masks to the image
            mask_overlay = self.apply_masks(cv_image, masks, class_ids)
        else:
            # If no masks, use the image with bounding boxes
            mask_overlay = result.plot()

        # Publish mask overlay image
        mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding="bgr8")
        mask_msg.header = msg.header  # Preserve the original message header
        self.mask_pub.publish(mask_msg)

    def apply_masks(self, image, masks, class_ids):
        # Apply masks to the image
        mask_overlay = image.copy()
        for mask, class_id in zip(masks, class_ids):
            # Resize mask to match image size if necessary
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            # Create a colored mask
            colored_mask = np.zeros_like(image)
            if class_id == 0:
                # Apply red mask for class ID 0
                colored_mask[:, :, 2] = mask * 255  # Red channel
            elif class_id == 1:
                # Apply blue mask for class ID 1
                colored_mask[:, :, 0] = mask * 255  # Blue channel
            # Overlay the mask on the image
            mask_overlay = cv2.addWeighted(
                mask_overlay, 1.0, colored_mask.astype(np.uint8), 0.5, 0
            )
        return mask_overlay

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV image (depth image)
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            # Optionally, process the depth image here or integrate it with object detection logic
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on ROS image topic"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./trained_weights.pt",
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--image_topic",
        type=str,
        default="robot1/zed2i/left/image_rect_color",
        help="ROS2 image topic to subscribe to",
    )
    parser.add_argument(
        "--depth_topic",
        type=str,
        default="/camera/depth/image_raw",
        help="ROS2 depth image topic to subscribe to",
    )
    parsed_args = parser.parse_args(args=args)

    rclpy.init(args=args)
    node = YoloInferenceNode(parsed_args.model_path, parsed_args.image_topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
