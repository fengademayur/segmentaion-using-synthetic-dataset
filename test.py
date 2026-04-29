import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# load trained model
model = YOLO("/home/mayurf/synthetic_data/heavy_guy/runs/segment/heavy_guy_seg/run_1/weights/best.pt")

# configure realSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# get RGB stream 
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)



try:
    while True:
        # Wait for a coherent frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # convert images to numpy arrays
        img = np.asanyarray(color_frame.get_data())

        # run Inference
        
        results = model.predict(img, imgsz=640, conf=0.5)

        # draw results
        annotated_frame = results[0].plot()

        # display the frame
        cv2.imshow("Segmentation", annotated_frame)

        # break loop on 'e' key
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

finally:
    # stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


