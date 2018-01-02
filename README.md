## Person_reID_baseline_pytorch



### Ablation Study

Input is resized to 256x128

| BatchSize | Dropout | Rank@1 | mAP | Reference|
| --------- | -------- | ----- | ---- | ---- |
| 32 | 0.5  | | | |
| 64 | 0.5  | 86.82 | 67.48 | |
| 64 | 0.5  | 85.42 | 65.29 | 0.4 color jitter|
| 64 | 0.75 | 84.86 | 66.06 | |
| 96 | 0.5  | 86.05 | 67.03 | |
| 96 | 0.75 | 85.66 | 66.44 | |
