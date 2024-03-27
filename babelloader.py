from dappy import read, preprocess
import numpy as np
import ssumo.data.quaternion as qtn
from typing import List
import torch
from ssumo.data.dataset import *
from torch.utils.data import DataLoader


def get_babel(
    data_config: dict,
    window: int = 51,
    train: bool = True,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
):
    """Read in BABEL dataset (cite)

    Args:
        data_config (dict): Dict of parameter options for loading in data.
        window (int, optional): # of frames per sample to be given by DataLoader class. Defaults to 51.
        train (bool, optional): Read in train or test set. Defaults to True.
        data_keys (List[str], optional): Data fields to be given by DataLoader class. Defaults to ["x6d", "root", "offsets"].
        shuffle (bool, optional): Whether DataLoader shuffles. Defaults to False.


    """
    ## TODO: Before starting this, set up your new mouse skeleton config
    # You can reference `ssumo/configs/mouse_skeleton.yaml``
    # Crucial elements are to identify your kinematic tree and offsets
    # Offsets will just be unit vectors in principle directions based off of kinematic tree
    # May also need to reorder keypoints
    skeleton_config = read.config(data_config["skeleton_path"])

    # TODO: Load in pose (Frames x keypoints x 3) and ids (1 per video) similar to dappy
    pose, ids = None, None  # Load or index train or test

    # Save raw pose in dataset if specified
    data = {"raw_pose": pose} if "raw_pose" in data_keys else {}

    # Get windowed indices (n_frames x window)
    window_inds = get_window_indices(ids, data_config["stride"], window)

    # Get root xyz position and center on (0,0,z)
    root = pose[..., 0, :][window_inds]
    root_center = np.zeros(root.shape)
    root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][:, None, :]

    # Get local 6D rotation representation -
    # Zhou, Yi, et al. "On the continuity of rotation representations in
    # neural networks." Proceedings of the IEEE/CVF Conference on
    # Computer Vision and Pattern Recognition. 2019.
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        # Getting local quaternions
        local_qtn = inv_kin(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[
                None,
                None,
            ],  # TODO: Change these to defined forward vector
        )
        # Converting quaternions to 6d rotation representations
        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Scale offsets by segment lengths
    if "offsets" in data_keys:
        data["offsets"] = get_segment_len(
            pose,
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
        )

    # Move everything to tensors
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    # Initialize Dataset and DataLoaders
    dataset = MouseDataset(
        data,
        window_inds,
        data_config["arena_size"],
        skeleton_config["KINEMATIC_TREE"],
        pose.shape[-2],
        label="Train" if train else "Test",
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return dataset, loader


# Use quaternions and fwd/inv kinematic functions to do the following:
# 1. Apply inv kin -> local 6D rotation -> fwd kinematics to reconstruct the original pose sequences
# 2. Once obtaining the decomposed local 6D rotations, visualize purely (0,0,0)-centered pose sequences
#    using only the fwd kinematic function, i.e. do not apply a translation anywhere.
# 3. Factor out yaw angle from the quaternions such that when you apply fwd kinematics, all poses will
#    face in the x+ direction, and be (0,0,0)-centered as in (2.)
# 4. Again using only the fwd kin function, plot pose sequences such that all segments are of length 1. Hint: look in ‘offsets’
# 5. Hardest one: rotate each window of pose sequences (length 51) such that the MIDDLE POSE is centered
#    on (0,0,Z) and rotated to face in the x+ direction. The other poses in the sequence should be rotated,
#    and translated accordingly.

# Models to train - a couple of the best VAE model architectures that you've found.
# Use pose representation which in which you only center the middle pose on (0,0,Z).
# Translate other poses according.
