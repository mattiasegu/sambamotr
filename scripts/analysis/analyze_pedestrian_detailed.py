import pandas as pd
import os.path as osp

exp_dir = "pretrained/sambamotr/sportsmot/sambamotr_residual_masking_sync_longer_sched2_lr_0.0002/val/det_0.5_track_0.5_miss_30_interval_1/checkpoint_17_tracker/"
# exp_dir = "pretrained/sambamotr/dancetrack/sambamotr_residual_masking_sync_longer_lr_0.0002/test/det_0.5_track_0.5_miss_35_interval_1/tracker"
df = pd.read_csv(osp.join(exp_dir, "pedestrian_detailed.csv"))
sorted_df = df.sort_values(by='HOTA___AUC', ascending=False)
sorted_df.to_csv(osp.join(exp_dir, "sorted_pedestrian_detailed.csv"), index=False)
print(sorted_df)