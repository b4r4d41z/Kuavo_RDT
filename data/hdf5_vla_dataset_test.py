import os
import fnmatch
import json
import h5py
import yaml
import cv2
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5VLADataset:
    """
    Load episodes from HDF5 for RDT fine-tuning.
    """
    def __init__(self) -> None:
        # Path to your Kuavo dataset
        HDF5_DIR = "data/Kuavo/"
        self.DATASET_NAME = "Kuavo"
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        # Load basic configuration
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Compute sampling weights based on episode lengths
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def fill_in_state(self, values):
        UNI_STATE_INDICES = (
            [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)] +
            [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
            [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)] +
            [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
        )
        uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
        uni_vec[..., UNI_STATE_INDICES] = values
        return uni_vec

    def fill_in_action(self, a):
        uni_a = np.zeros(self.STATE_DIM)
        uni_a[80:83] = a[0:3]
        uni_a[83:86] = a[3:6]
        uni_a[86:89] = a[6:9]
        uni_a[60]    = a[9]
        uni_a[30:33] = a[10:13]
        uni_a[33:36] = a[13:16]
        uni_a[36:39] = a[16:19]
        uni_a[10]    = a[19]
        return uni_a

    def parse_hdf5_file(self, file_path):
        """
        Parse HDF5 file and generate a training sample.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]

            if num_steps < 128:
                return False, None

            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                return False, None

            step_id = np.random.randint(max(first_idx-1, 0), num_steps)

            instruction = f['instruction'][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            target_qpos = f['action'][step_id:step_id + self.CHUNK_SIZE]
            if target_qpos.shape[0] < self.CHUNK_SIZE:
                pad = np.tile(target_qpos[-1:], (self.CHUNK_SIZE - target_qpos.shape[0], 1))
                target_qpos = np.concatenate([target_qpos, pad], axis=0)

            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))

            actions = target_qpos.copy()

            state = self.fill_in_state(state)
            state_indicator = self.fill_in_state(np.ones_like(state_std))
            state_std = self.fill_in_state(state_std)
            state_mean = self.fill_in_state(state_mean)
            state_norm = self.fill_in_state(state_norm)

            if actions.ndim == 1:
                actions = self.fill_in_action(actions)
            else:
                uni_actions = np.zeros((actions.shape[0], self.STATE_DIM))
                for i in range(actions.shape[0]):
                    uni_actions[i] = self.fill_in_action(actions[i])
                actions = uni_actions

            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISTORY_SIZE+1, 0), step_id+1):
                    img = f['observations']['images'][key][i]
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    pad = np.tile(imgs[:1], (self.IMG_HISTORY_SIZE - imgs.shape[0], 1, 1, 1))
                    imgs = np.concatenate([pad, imgs], axis=0)
                return imgs

            cam_high = parse_img('cam_high')
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISTORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()

            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """
        Parse the full trajectory for statistics.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            if num_steps < 128:
                return False, None

            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                return False, None

            target_qpos = f['action'][:]
            state_traj = qpos[max(first_idx-1, 0):]
            action_traj = target_qpos[max(first_idx-1, 0):]

            state_unified = self.fill_in_state(state_traj)

            action_unified = np.zeros((action_traj.shape[0], self.STATE_DIM))
            for i in range(action_traj.shape[0]):
                action_unified[i] = self.fill_in_action(action_traj[i])

            return True, {"state": state_unified, "action": action_unified}
    