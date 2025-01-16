# Planning
**Important Note:** can’t do major pivots, so do a quick but thorough
feasibility check (pilot labeling, pilot inference) to confirm that the chosen
pipeline can realistically be accomplished.

## Implementation Constraints
* a small or quantized model if going deep learning
* pretrained models, or potentially minimal retraining.
* real-time or near-real-time detection and speed for accuracy trade-off.

## Data Preprocessing & Augmentation
* [later-iteration] Image Preprocessing:
  * consider if need to adjust brightness/contrast or reflection-correction due
    to metallic reflective surface
  * resize or crop them to speed up training or inference?
* [later-iteration] Point Cloud Preprocessing
  * noise or outliers removal
  * downsample (e.g., voxel grid) or segment the region of interest (ROI) to
    reduce computation.
* [later-iteration] Data Augmentation
  * basic augmentations (translation, rotation, flipping) for 2D training.
  * be cautious about 3D want to preserve physical constraints for pose
    estimation.

## System Design & Modeling
### 2D detection
* [first-iteration] Conventional CV: simple edge/shape-based or template
  matching, gets a decent functional baseline. Potential approach: detect
  circular/hexalobular patterns using classical methods (Canny edges,
  morphological operations, Hough transform, etc.), then refine or classify with
  simple heuristics.
* [later-iteration] Most of the screws of interest seem to be on top of the
  battery surfaces. In order to simplify the problem and avoiding detection of
  other screws (e.g. from the fixture/clamps), we could first detect the battery
  surface and its margin and use that as a guide.
* [later-iteration] DL models: lightweight architecture (e.g., MobileNet-based
  or YOLO Nano) that can run on CPU.

### 3D Pose Estimation
* [not-feasible] more robust 3D pose would require camera calibration.
* [first-iteration] Detecting the orientation of the screws just from their head
  and too little data might be challenging. Given that we get the 2D detection
  more reliably, we can project the detection as a ray (can't have 3D
  projection, depth is missing) and get an estimate of depth from the point
  cloud. For this purpose, we can make the assumption that screws, in this
  special case, are perpendicular to the top surface of the battery.
  Furthermore, we make the assumption that the battery surface is parallel to
  the image plane of the camera (can verify this by use of some plane detection
  in 3D pointcloud and compare with camera axis).
* [later-iteration] not really sure at this stage

### Post Detection2D processing
* [later-iteration] when all frames of the same battery pack have been processed
  and detection are available in global/robot coordinate, merge pointclouds from
  all frames into a single pointcloud, and cluster detections to for 1)
  improving detection and pose estimate results, and 2) pretty visualization.

## Evaluation & Testing
* 2D metrics are easy as 2D annotations are easily made;
  * Standard detection metrics (precision, recall, mAP),using minimum IoU
    threshold for bounding boxes.
  * Localization Accuracy in 2D, pixel or mm
* Pose Estimation Metrics: Orientation & Position Errors
  * this is not really feasible until I sort out the 3D annotation
* Cross-Pack Validation: train on pack 1 and see how it generalizes to pack 2
  (and vice versa). If combine battery packs for training, keep a few frames
  from each set for testing.

## Documentation & Deliverables
* Code or docker image, ready to deploy on any arbitrary machine
* A short presentation
* User Guide / Readme / Documentations
  * [first-iteration] how to run the system (dependencies, environment, etc.).
  * [later-iteration] data format, model usage, annotation guideline used
* Future Work Suggestions

