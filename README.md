[![black](https://github.com/saeedghsh/screw_detector/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/saeedghsh/screw_detector/actions/workflows/formatting.yml)
[![pylint](https://github.com/saeedghsh/screw_detector/actions/workflows/pylint.yml/badge.svg?branch=master)](https://github.com/saeedghsh/screw_detector/actions/workflows/pylint.yml)
[![mypy](https://github.com/saeedghsh/screw_detector/actions/workflows/type-check.yml/badge.svg?branch=master)](https://github.com/saeedghsh/screw_detector/actions/workflows/type-check.yml)
[![pytest](https://github.com/saeedghsh/screw_detector/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/saeedghsh/screw_detector/actions/workflows/pytest.yml)
[![pytest-cov](https://github.com/saeedghsh/screw_detector/actions/workflows/pytest-cov.yml/badge.svg?branch=master)](https://github.com/saeedghsh/screw_detector/actions/workflows/pytest-cov.yml)

# Screw Detection

## Cheat Sheet

### Usage examples

**NOTE:** For "most" of these operation to work,
`dataset/screw_detection_challenge/` with certain directory structure must be
available. Those entry points with `direct` mode could potentially work on
arbitrary paths (the directory structure should still be similar to the
structure below).

```bash
.
├── dataset
│   └── screw_detection_challenge
│       ├── battery_pack_1
│       │   ├── MAN_ImgCap_closer_zone_10
│       │   │   ├── MAN_ImgCap_closer_zone_10.json
│       │   │   ├── MAN_ImgCap_closer_zone_10.ply
│       │   │   └── MAN_ImgCap_closer_zone_10.png
│       │   ...
│       ├── battery_pack_1
│       │   ├── MAN_ImgCap_closer_zone_10
│       │   │   ├── MAN_ImgCap_closer_zone_10.json
│       │   │   ├── MAN_ImgCap_closer_zone_10.ply
│       │   │   └── MAN_ImgCap_closer_zone_10.png
│       │   ...
│       ├── battery_pack_1_annotations_datumaro.json
│       ├── battery_pack_2_annotations_datumaro.json
│       └── data_split_cache
│           ├── 20250112T232216_0.2_split.json
            ...
```

**visualizer:** Not very useful, just visualizes the date - it was starting point for integrating the visualizer into detector entry point for inspection during dev.
```bash
python -m entry_points.visualizer dataset
python -m entry_points.visualizer direct --input-path dataset/screw_detection_challenge/battery_pack_2
```

**Dataset split** for train and test. Every time it is run, a new data spit is performed and cached.
```bash
python -m entry_points.dataset_splitter
```

**2D detector**: with visualization options, mostly for dev purposes.
```bash
python -m entry_points.detector_2d dataset
python -m entry_points.detector_2d direct --input-path dataset/screw_detection_challenge/battery_pack_2
```

**Evaluation** of the specified detector over cached split data adn store the evaluation result under `evaluation_logs`.
```bash
python -m entry_points.evaluator
```

**Pose estimation**: runs the whole pipeline of 2D detector following by 3D processing and 3D pose estimation.
```bash
python -m entry_points.pose dataset
python -m entry_points.pose direct --input-path dataset/screw_detection_challenge/battery_pack_2
```

### Code quality tools

Tests, coverage, linter, formatter, static type check, ...
```bash
$ black . --check
$ isort . --check-only
$ mypy . --explicit-package-bases
$ pylint $(git ls-files '*.py')
$ xvfb-run --auto-servernum pytest
$ xvfb-run --auto-servernum pytest --cov=.
$ xvfb-run --auto-servernum pytest --cov=. --cov-report html; firefox htmlcov/index.html
$ coverage report -m # see which lines are missing coverage
```

**Profiling if needed**
```bash
python -m cProfile -o profile.out -m entry_points.dataset_visualizer
tuna profile.out
```

## TODOs and future work
* [x] add direct mode option to visualizer
* [x] rename `entry_detector` to `entry_detector_2d`
* [x] remove `entry` from all entry points scripts
* [ ] `_colors` function is a mess
  * the dict is not even used! use something like Enum for color name-idx
    mapping. the optional idx does not have default value. opencv and open3d use
    different order of RGB
* [ ] Classifier based 2D detector
  * [ ] add train-validation split, keep test separate
  * [ ] consider cropping annotated areas and store a training set instead of
    creating it on the fly every run! this way you run the split function once,
    store frames ids of test and train set and crops of annotated regions plus
    some TN examples. Test set can be fetched and evaluation can be on the fly.
  * [ ] since the circle annotations are very tight, consider padding the
    bounding boxes with 10% in each direction
  * [ ] data augmentation
  * [ ] negative sample selection, like considering content difference in
    selected negative samples to make sure enough variance.
  * [ ] Annotation tagging (system analysis and not just detector):
    * annotate screws not on top surface and add tags `on_top_surface`
    * partial visibility: 
    * those in shadow
* 3D:
  * [ ] template matching
  * [ ] plane detection
* Visualization
  * [ ] draw annotations also on the point cloud
  * [ ] `o3d` and `cv2` windows should be synchronized, so that I don't have to
    close two windows every time
  * [ ] add visualization of model prediction to visualization pipeline, 2D and
    3D!
  * [ ] replace visualizer with readily available anf off-the-shelf (e.g "rerun")
* [ ] Parameter optimizer

## Note
Portions of this code/project were developed with the assistance of ChatGPT (a product of OpenAI) and Copilot (A product of Microsoft).
