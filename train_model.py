import ssumo
from ssumo.params import read
import sys
from base_path import RESULTS_PATH
from pathlib import Path
import pickle
### Set/Load Parameters
analysis_key = sys.argv[1]

if len(sys.argv) > 2:
    z_path = Path(RESULTS_PATH + analysis_key)
    folders = [str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()]
    job_id = sys.argv[2]
    analysis_key = "{}/{}/".format(analysis_key, folders[int(job_id)])

config = read.config(RESULTS_PATH + analysis_key + "/model_config.yaml")

dataset, loader, model = ssumo.get.data_and_model(
    config,
    dataset_label="Train",
    data_keys=["x6d", "root", "offsets", "target_pose"]
    + config["disentangle"]["features"],
    shuffle=True,
)

model = ssumo.train.train(config, model, loader)
