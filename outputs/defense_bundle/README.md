# Final outputs

## Main presentation video

* [`videos/DEFENSE_DEMO_REEL_V2.mp4`](videos/DEFENSE_DEMO_REEL_V2.mp4): segmentation, oracle reference, and successful learned RGB control

## Individual videos

### Segmentation

* [`videos/segmentation_rgb_gt_prediction.mp4`](videos/segmentation_rgb_gt_prediction.mp4): RGB image, ground truth mask, and model prediction

### Control references and baselines

* [`videos/control_oracle_reference_success.mp4`](videos/control_oracle_reference_success.mp4): full state performance ceiling
* [`videos/control_heuristic_baseline_success.mp4`](videos/control_heuristic_baseline_success.mp4): rule based baseline success
* [`videos/control_random_baseline_failure.mp4`](videos/control_random_baseline_failure.mp4): random control failure baseline

### Vision bridges

* [`videos/control_synthetic_vision_bridge_failure.mp4`](videos/control_synthetic_vision_bridge_failure.mp4): HSV feature bridge; task failure
* [`videos/control_model_vision_bridge_failure.mp4`](videos/control_model_vision_bridge_failure.mp4): DeepLab centroid bridge; task failure

### Learned RGB controller

* [`videos/control_bc_rgb_v2_success.mp4`](videos/control_bc_rgb_v2_success.mp4): learned RGB only policy success
* [`videos/control_bc_rgb_v2_success_seed7.mp4`](videos/control_bc_rgb_v2_success_seed7.mp4): same policy, different initial state
* [`videos/control_bc_rgb_v2_success_seed53.mp4`](videos/control_bc_rgb_v2_success_seed53.mp4): same policy, different initial state

## Figures

### Segmentation panels

![Segmentation panels](polish/real_overlays/real_panels_contact.png)

### Per class IoU

![Per class IoU](plots/seg_eval_run3_ft_iou_sorted.png)

### Confusion matrix

![Confusion matrix](plots/seg_eval_run3_ft_confusion.png)

### Overlay contact sheet

![Overlay contact sheet](overlays/contact_sheet_run3_ft.png)

## Models and metrics

* `checkpoints/run3_ft/best.pt`: final segmentation model
* `checkpoints/bc_rgb_v2/best.pt`: final learned control model
* `metrics/seg_eval_run3_ft.json`: segmentation evaluation
* `metrics/bc_v2_summary.json`: learned controller evaluation
* `metrics/phase2_summary_defense.json`: controller comparison
