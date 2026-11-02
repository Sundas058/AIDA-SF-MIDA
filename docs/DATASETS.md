# 📁 Dataset preparation

The training commands accept a common `--data-dir`. Each child directory uses a registered name: `market1501`, `dukemtmc`, `msmt17`, `cuhk03`, `personx`, or `unreal`.

## Folder-based datasets

Market-1501, DukeMTMC-ReID, PersonX, and Unreal readers accept:

```text
DATA_ROOT/DATASET_NAME/
├── bounding_box_train/
├── query/
└── bounding_box_test/
```

These three folders can also sit inside one extracted dataset directory immediately below `DATASET_NAME/`. Image filenames use `IDENTITY_cCAMERA_...`, for example `0007_c2_001.jpg`. The reader converts camera IDs to zero-based integers, ignores identity `-1`, and remaps source training identities to contiguous labels.

All stored image paths are absolute. Multi-source training offsets identity labels by domain so identical numeric IDs in separate datasets remain distinct.

## MSMT17

Place the official list files and image folders in `msmt17/MSMT17_V1/` or `msmt17/MSMT17_V2/`.

| Version | Training images | Query/gallery images |
|:--|:--|:--|
| V1 | `train/` | `test/` |
| V2 | `mask_train_v2/` | `mask_test_v2/` |

List files use the official `relative_image_path identity` format. Training combines `list_train.txt` and `list_val.txt` when the validation list is present. Query and gallery follow `list_query.txt` and `list_gallery.txt`. Cameras are read from the third underscore-separated filename field.

## CSV manifests and CUHK03

Any registered dataset accepts `train.csv`, `query.csv`, and `gallery.csv` inside its dataset directory. Each manifest has these columns:

```csv
path,pid,camid
images/person_0001_view_a.png,1,0
images/person_0001_view_b.png,1,1
images/person_0002_view_a.png,2,0
```

Image paths are relative to the manifest directory or absolute. Camera IDs in manifests are supplied directly, with consistent numbering across query and gallery.

For **CUHK03**, prepare manifests from the selected detected/labeled image release and its fixed train/query/gallery split. Preserve identity and camera information from that split. The default evaluator uses the all-gallery camera-filtered CMC/mAP protocol; it does not randomly resample a one-image-per-identity gallery.

For **PersonX**, preserve the chosen release and camera subset in the manifests when filenames use another naming scheme.

## Target-only training images

SF-MIDA reads only the target training split. It ignores identity labels in its training manifest or filenames and generates all training labels through clustering. Arbitrary target image names can be used through `train.csv`, with `pid=-1`.

DG and source pretraining request only source `train` splits. Evaluation independently requests `query` and `gallery`, so target test images are never needed by the DG training routine.

## Access and split tracking

Datasets are supplied by their original owners and are not redistributed here. Record the dataset version, image selection, and split manifests alongside each experiment. Input readers report missing paths and invalid filename formats before optimization.

