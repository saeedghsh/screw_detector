[stub-to-complete]

# Annotation Guideline:

## labeling
* screw heads are annotated with circle/ellipse, labeled `screw_head`
* ATM, only screws on top of the battery surface are annotated, not any other
  screws from the fixture or else. Should the need rises in the future, other
  screws will be annotation with the same or separate label depending on the
  need/context.
* empty screw holes are annotated with a separate class ("screw_hole"). This
  applies only to screw holes on top of the battery surface, i.e. holes that
  would have been annotated as "screw_head" should have there been any screw.
  similar to `screw_head` label
* different screw types are not distinguished at this stage, all have the same
  label ("screw_head")
* No particular tagging (such as lighting condition, occlusion, or else) is
  provided right now. Given that the data is seemingly collected in a fairly
  controlled environment and has fair quality, and the limited variety of data
  such tagging is skipped.

## Dataset directory structure
```bash
dataset/
└── screw_detection_challenge
    ├── battery_pack_1 # (task1) 
    │   ├── MAN_ImgCap_closer_zone_10
    │   ├── ...
    │
    ├── battery_pack_2 # (task2)
    │   ├── MAN_ImgCap_closer_zone_120
    │   ├── ...
    │
    ├── battery_pack_1_annotations_datumaro.json
    └── battery_pack_2_annotations_datumaro.json
```

## TODO:
* dataset organization, description and how-to
  * data organization: for simplicity followed the original directory structure
    of the provided dataset. used that as implicit data structure and coded
    against that
* strategy for Label Consistency & Quality Control 
* dataset metadata, i.e. per frame description (battery pack ID, transform
  matrix, whether a screw is present, etc)
* explain the two annotation tasks under the project!
* 3D?
