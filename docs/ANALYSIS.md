# Analysis conventions

The supplied checkpoint should be trained specifically for the tennis ball: the current detector selects the highest-confidence bounding box rather than filtering a multi-class model by label.

MediaPipe tracks the player pose. The detector tracks ball centers, draws trajectories, and records detection counts, displacement statistics, approximate court zones, and player movement.

Speed values are pixel displacement between successive observations. They are not calibrated km/h or physical court measurements; gaps between detections and camera movement affect interpretation. The overlay's px/frame labels assume adjacent observations. Rally/shot segmentation and world-coordinate calibration are not implemented.

The side-by-side overlay requires a source of at least 640×480. Fractional frame rates are preserved. Downloads use a temporary directory; capture, writer, and pose resources are closed after processing. No real model or video run was validated during this repository pass.
