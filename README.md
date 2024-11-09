# pallete_detection_ros2


## Setup Grounded Sam using the documentation and create data or use the uploaded data:
1. https://github.com/IDEA-Research/Grounded-Segment-Anything
2. Move the create_annotation_yolo.py to Grounded-Segment-Anything.
3. Download grounded_dino weights and SAM weights into grounded-segment-anything folder. 
4. Run create_annotation_yolo.py with --output_dir where labels will be generated and input_dir where images are located. 

## Train the yolov11 model. 

### Run the image publisher for testing.
```
python publish_image.py ---model_path "" --image_topic "/camera/image_raw"
```

#### else

#### Run detector

```
python inference_yolo_ros.py ---model_path "" --image_topic "/camera/image_raw"
```
