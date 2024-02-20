# FQDet

## :page_with_curl: Paper
- [FQDet: Fast-converging Query-based Detector](https://arxiv.org/abs/2210.02318) by [Cédric Picron](https://cedricpicron.github.io), [Punarjay Chakravarty](https://scholar.google.be/citations?user=AyXW9gYAAAAJ&hl) and [Tinne Tuytelaars](https://scholar.google.be/citations?user=EuFF9kUAAAAJ).

## :newspaper: News
- **`01 Jun, 2023`:** FQDet logs and checkpoints released.
- **`24 May, 2023`:** FQDet code released (with configs and training scripts).
- **`23 Oct, 2022`:** FQDet paper was accepted at NeurIPS 2022 (VTTA workshop).

## :dvd: Results
Object detection results on the 2017 COCO validation set. The inference FLOPs and FPS are measured on the first 100 images of the 2017 COCO validation set using an NVIDIA GeForce RTX 3060 Ti GPU.

| Backbone | Detector | Epochs |  AP  | Params | GFLOPs |  FPS  | Script | Log | Checkpoint |
|   :-:    |   :-:    |  :-:   | :-:  |  :-:   |  :-:   |  :-:  |  :-:   | :-: |    :-:     |
| R50+FPN  |  FQDet   |   12   | **43.3** | 33.9 M |  99.0  |  20.9 | [script](scripts/r50_fpn_fqdet_12e.sh) | [log](outputs/r50_fpn_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1rLhwkT1CwZdpiuWD0-a_9B5UnkA4D-e0?usp=sharing) |
| R50+TPN  |  FQDet   |   12   | **45.5** | 42.2 M |  107.8  |  13.6 | [script](scripts/r50_tpn_fqdet_12e.sh) | [log](outputs/r50_tpn_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1rLhwkT1CwZdpiuWD0-a_9B5UnkA4D-e0?usp=sharing) |
| R50+DefEnc  |  FQDet   |   12   | **47.2** | 44.1 M |  234.8  |  9.7 |  [script](scripts/r50_def_fqdet_12e.sh) | [log](outputs/r50_def_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1rLhwkT1CwZdpiuWD0-a_9B5UnkA4D-e0?usp=sharing) |

## :hammer_and_wrench: Installation
- **Environment**: 
  1. Install the `conda` package and environment management system if not already done.
  2. Execute `source setup_env.sh`.

- **Data preparation**:
  1. Download the desired datasets.
  2. Modify the paths in `setup_data.sh` to point to your installation directories.
  3. Execute `source setup_data.sh`.

## :seedling: Usage
- **Training**: Execute `python main.py` with the desired command-line arguments. Some example training scripts, which were used to obtain the results from above, are found in the `scripts` directory.

- **Evalutation**: Execute `python main.py --eval --eval_task $TASK` with the desired command-line arguments, with $TASK chosen from:
  1. analysis: Analyze the computional cost of the given model.
  2. comparison: Compare the results from two different models.
  3. performance: Compute the model performance on the desired benchmark.
  4. profile: Profile the given model.
  5. visualize: Visualize the model predictions.

## :bookmark: Citation
If you find FQDet useful in your research or wish to refer to the results published here, please use following BibTeX entry:
```BibTeX
@article{picron2022fqdet,
  title={FQDet: Fast-converging Query-based Detector},
  author={Picron, C{\'e}dric and Chakravarty, Punarjay and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:2210.02318},
  year={2022}
}
```
