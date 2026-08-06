# Dataset split manifests

Each file contains one image stem per line:

- `train.txt`: 4555 images from the official VisDrone2019-DET training package.
- `val.txt`: 250 images from the official VisDrone2019-DET validation package.
- `test.txt`: 251 images from the official VisDrone2019-DET validation package.
- `all_stems.txt`: the union of the three partitions.

The `visdrone_train_` and `visdrone_val_` prefixes preserve the source package namespace and prevent sequences with the same numeric identifier from being conflated. Use `scripts/build_nmv3_from_visdrone.py` to reconstruct images and YOLO labels from the official distribution.

