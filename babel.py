import sys
from base_path import HUMAN_DATA_PATH

sys.path.append(HUMAN_DATA_PATH + "human_action/human_body_prior")
from human_body_prior.body_model.body_model import BodyModel
import torch
import numpy as np
import json
from tqdm import tqdm

babel = []
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/train.json")))
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/val.json")))
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/test.json")))

splits = ["train", "val", "test"]

allsets = [
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "DFaust67",
    "EKUT",
    "EyesJapanDataset",
    "HumanEva",
    "KIT",
    "MPIHDM05",
    "MPILimits",
    "MPImosh",
    "SFU",
    "SSMsynced",
    "TCDhandMocap",
    "TotalCapture",
    "Transitionsmocap",
]
bm_fname = HUMAN_DATA_PATH + "babelmodel.npz"
num_betas = 16  # number of body parameters


for set in range(len(babel)):
    meta = []
    data = []
    labels = []
    for k in tqdm(babel[set].keys()):
        meta.append(
            [
                k,
                # allsets.index(babel[set][k]["feat_p"].split("/")[0]),
                babel[set][k]["feat_p"].split("/")[0],
                babel[set][k]["url"],
            ]
        )  # babel id, dataset ID, url

        amass_npz_fname = HUMAN_DATA_PATH + "AMASS/" + babel[set][k]["feat_p"]
        bdata = np.load(amass_npz_fname)
        time_length = len(bdata["trans"])

        body_parms = {
            "root_orient": torch.Tensor(
                bdata["poses"][:, :3]
            ),  # controls the global root orientation
            "pose_body": torch.Tensor(bdata["poses"][:, 3:66]),  # controls the body
            "pose_hand": torch.Tensor(
                bdata["poses"][:, 66:]
            ),  # controls the finger articulation
            "trans": torch.Tensor(bdata["trans"]),  # controls the global body position
            "betas": torch.Tensor(
                np.repeat(
                    bdata["betas"][:num_betas][np.newaxis], repeats=time_length, axis=0
                )
            ),
        }

        bm = BodyModel(bm_path=bm_fname, num_betas=num_betas, model_type="smplh")

        bm.root_orient = torch.nn.Parameter(body_parms["root_orient"])
        bm.pose_body = torch.nn.Parameter(body_parms["pose_body"])
        bm.betas = torch.nn.Parameter(body_parms["betas"])
        bm.pose_hand = torch.nn.Parameter(body_parms["pose_hand"])

        body_pose_beta = bm(
            **{k: v for k, v in body_parms.items() if k in ["pose_body", "betas"]}
        )

        data.append(np.array(body_pose_beta.Jtr.detach()))

        if babel[set][k]["frame_ann"] == None:
            labels.append(
                babel[set][k]["seq_ann"]["labels"][0]["act_cat"]
            )  # gives list of act labels if no frame_ann
        else:
            seq = [
                [i["act_cat"], i["start_t"], i["end_t"]]
                for i in babel[set][k]["frame_ann"]["labels"]
            ]  # these 3 lines give ordered list of frame labels for lists of actions
            sorter = np.array([i[1:] for i in seq])
            sortedseq = [seq[i] for i in np.argsort(sorter[:, 0])]
            framelabels = [[] for i in range(len(data[-1]))]
            for segment in sortedseq:
                for i in range(
                    int(segment[1] * 120), min(int(segment[2] * 120), len(framelabels))
                ):
                    framelabels[i] += segment[0]
            labels.append(framelabels)

    np.save(
        HUMAN_DATA_PATH + "BABEL/data_" + splits[set] + ".npy",
        np.asarray(data, dtype="object"),
        allow_pickle=True,
    )
    np.save(
        HUMAN_DATA_PATH + "BABEL/meta_" + splits[set] + ".npy",
        np.asarray(meta, dtype="object"),
        allow_pickle=True,
    )
    np.save(
        HUMAN_DATA_PATH + "BABEL/labels_" + splits[set] + ".npy",
        np.asarray(labels, dtype="object"),
        allow_pickle=True,
    )

# data = np.load(HUMAN_DATA_PATH + "BABEL/data.npy", allow_pickle=True)
# labels = np.load(HUMAN_DATA_PATH + "BABEL/labels.npy", allow_pickle=True)
# meta = np.load(HUMAN_DATA_PATH + "BABEL/meta.npy", allow_pickle=True)
