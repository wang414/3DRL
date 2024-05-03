# Import required packages
import argparse
import os.path as osp
from pathlib import Path
import os
import gymnasium as gym
import h5py
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
import glob


def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if th.is_tensor(x):
        return x.cpu().numpy()
    return x


def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images

    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    from mani_skill2.utils.common import flatten_state_dict

    state = np.hstack(
        [
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"]),
        ]
    )

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, rgb2], axis=-1)
    obs = dict(rgb=rgbd, state=state)
    return obs


def rescale_rgbd(rgbd, scale_rgb_only=False):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., :3] / 255.0
    rgb2 = rgbd[..., 3:] / 255.0
    # depth1 = rgbd[..., 3:4]
    # depth2 = rgbd[..., 7:8]
    # if not scale_rgb_only:
    #     depth1 = rgbd[..., 3:4] / (2**10)
    #     depth2 = rgbd[..., 7:8] / (2**10)
    return np.concatenate([rgb1, rgb2], axis=-1)

    # return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        # for details on how the code below works, see the
        # quick start tutorial
        # self.obs_state = []
        # self.obs_rgbd = []
        # self.actions = []
        self.files = glob.glob(os.path.join(dataset_dir,'*.rgbd.pd_ee_delta_pose.h5'))
        self.steps_per_obj = np.zeros(len(self.files), dtype=int)
        self.len_eps_objs = []
        for idx, file in enumerate(self.files):
            tt_st, st = self.load_single_obj_info(file)
            self.steps_per_obj[idx] = tt_st
            self.len_eps_objs.append(np.cumsum(np.array(st)))
        self.steps_per_obj = np.cumsum(np.array(self.steps_per_obj))

    def load_single_obj_info(self, dataset_file):
        import h5py

        from mani_skill2.utils.io_utils import load_json
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        # env_info = json_data["env_info"]
        # self.env_id = env_info["env_id"]
        # self.env_kwargs = env_info["env_kwargs"]
        total_steps = 0
        steps = []
        load_count = len(episodes)
        for eps_id in range(load_count):
            eps = episodes[eps_id]
            steps.append(eps['elapsed_steps'])
            total_steps+=eps['elapsed_steps']
            # trajectory = data[f"traj_{eps['episode_id']}"]
            # trajectory = load_h5_data(trajectory)
            # # convert the original raw observation with our batch-aware function
            # obs = convert_observation(trajectory["obs"])
            # # we use :-1 to ignore the last obs as terminal observations are included
            # # and they don't have actions
            # # self.obs_rgbd.append(obs["rgb"][:-1])
            # # self.obs_state.append(obs["state"][:-1])
            # # self.actions.append(trajectory["actions"])
            # print(obs["rgb"].shape)
            # exit()
        return total_steps, steps

    def __len__(self):
        return self.steps_per_obj[-1]
      
    def compute_idx(self, idx):
        obj_id = np.searchsorted(self.steps_per_obj, idx, side='right')
        if obj_id > 0:
            idx = idx-self.steps_per_obj[obj_id-1]
        trj_id = np.searchsorted(self.len_eps_objs[obj_id], idx, side='right')
        if trj_id > 0:
            idx = idx - self.len_eps_objs[obj_id][trj_id-1]
        return obj_id, trj_id, idx
    
    def load_step(self, obj_id, trj_id, step_id):
        import h5py
        dataset_file = self.files[obj_id]
        from mani_skill2.utils.io_utils import load_json
        data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        # env_info = json_data["env_info"]
        # self.env_id = env_info["env_id"]
        # self.env_kwargs = env_info["env_kwargs"]
        eps = episodes[trj_id]
 
        trajectory = data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)
            # # convert the original raw observation with our batch-aware function
        obs = convert_observation(trajectory["obs"])
            # # we use :-1 to ignore the last obs as terminal observations are included
            # # and they don't have actions
        rgb = obs["rgb"][step_id]
        state = obs["state"][step_id]
        action = trajectory["actions"][step_id]

        return rgb, state, action

    def __getitem__(self, idx):
        obj_id, trj_id, step_id = self.compute_idx(idx)
        rgb, state, action = self.load_step(obj_id, trj_id, step_id)
        action = th.from_numpy(action).float()
        # note that we rescale data on demand as opposed to storing the rescaled data directly
        # so we can save a ton of space at the cost of a little extra compute
        rgb = rescale_rgbd(rgb)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgb = th.from_numpy(rgb).float().permute((2, 0, 1))
        state = th.from_numpy(state).float()
        return dict(rgb=rgb, state=state), action


class NatureCNN(nn.Module):
    def __init__(self, image_size=(128, 128), in_channels=8, state_size=42):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        test_tensor = th.zeros([in_channels, image_size[0], image_size[1]])
        with th.no_grad():
            n_flatten = cnn(test_tensor[None]).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        # for state data we simply pass it through a single linear layer
        extractors["state"] = nn.Linear(state_size, 64)
        self.out_features += 64

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class Policy(nn.Module):
    def __init__(
        self,
        image_size=(128, 128),
        in_channels=8,
        state_size=42,
        hidden_units=[128, 128],
        act_dims=8,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.feature_extractor = NatureCNN(image_size, in_channels, state_size)
        mlp_layers = []
        prev_units = self.feature_extractor.out_features
        for h in hidden_units:
            mlp_layers += [nn.Linear(prev_units, h), activation()]
            prev_units = h
        mlp_layers += [nn.Linear(prev_units, act_dims), nn.Tanh()]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, observations) -> th.Tensor:
        features = self.feature_extractor(observations)
        return self.mlp(features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Simple script demonstrating how to train an agent with imitation learning (behavior cloning) using ManiSkill2 environmnets and demonstrations"
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-d", "--demos", type=str, help="path to demonstration dataset .h5py file"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/bc_rgbd",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--steps", type=int, help="number of training steps", default=8000
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env_id = args.env_id
    demo_path = args.demos
    log_dir = args.log_dir
    iterations = args.steps

    ckpt_dir = osp.join(log_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    obs_mode = "rgbd"
    control_mode = "pd_ee_delta_pose"
    env = gym.make(
        env_id, obs_mode=obs_mode, control_mode=control_mode, render_mode="cameras"
    )
    # RecordEpisode wrapper auto records a new video once an episode is completed
    env = RecordEpisode(
        env,
        output_dir=osp.join(log_dir, "eval_videos" if args.eval else "videos"),
        info_on_video=True,
    )
    # dataset = ManiSkill2Dataset(demo_path)
    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "checkpoints/ckpt_latest.pt")
        # Load the saved model
        policy = th.load(model_path)
    else:
        assert (
            demo_path is not None
        ), "Need to provide a demonstration dataset via --demos"
        dataset = ManiSkill2Dataset(demo_path)
        dataloader = DataLoader(
            dataset,
            batch_size=512,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        obs, action = dataset[0]
        print("RGBD:", obs["rgb"].shape)
        print("State:", obs["state"].shape)
        print("Action:", action.shape)
        # create our policy
        obs, action = dataset[0]
        rgbd_shape = obs["rgb"].shape
        th.manual_seed(0)
        policy = Policy(
            image_size=rgbd_shape[1:],
            in_channels=rgbd_shape[0],
            state_size=obs["state"].shape[0],
            act_dims=action.shape[0],
            hidden_units=[256, 256, 256],
        )
    # move model to gpu if possible
    device = "cuda" if th.cuda.is_available() else "cpu"
    policy = policy.to(device)
    print(policy)

    loss_fn = nn.MSELoss()

    # a short save function to save our model
    def save_model(policy, path):
        th.save(policy, path)

    def train_step(policy, obs, actions, optim, loss_fn):
        optim.zero_grad()
        # move data to appropriate device first
        obs_device = dict()
        for k in obs:
            obs_device[k] = obs[k].to(device)
        actions = actions.to(device)

        pred_actions = policy(obs_device)

        # compute loss and optimize
        loss = loss_fn(actions, pred_actions)
        loss.backward()
        optim.step()
        return loss.item()

    def evaluate_policy(env, policy, num_episodes=10):
        obs, _ = env.reset()
        successes = []
        i = 0
        pbar = tqdm(total=num_episodes)
        while i < num_episodes:
            # convert observation to our desired shape, rescale correctly, and move to appropriate device
            obs = convert_observation(obs)
            obs["rgb"] = rescale_rgbd(obs["rgb"], scale_rgb_only=True)
            obs_device = dict()
            # unsqueeze adds an extra batch dimension and we permute rgbd since PyTorch expects the channel dimension to be first
            obs_device["rgb"] = (
                th.from_numpy(obs["rgb"])
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            obs_device["state"] = (
                th.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
            )
            with th.no_grad():
                action = policy(obs_device).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(info["success"])
                i += 1
                obs, _ = env.reset(seed=i)
                pbar.update(1)
        success_rate = np.mean(successes)
        return success_rate

    if not args.eval:
        writer = SummaryWriter(log_dir)

        optim = th.optim.Adam(policy.parameters(), lr=1e-3)
        best_epoch_loss = np.inf
        pbar = tqdm(dataloader, total=iterations)
        epoch = 0
        steps = 0
        while steps < iterations:
            epoch_loss = 0
            for batch in dataloader:
                steps += 1
                obs, actions = batch
                loss_val = train_step(policy, obs, actions, optim, loss_fn)

                # track the loss and print it
                writer.add_scalar("train/mse_loss", loss_val, steps)
                epoch_loss += loss_val
                pbar.set_postfix(dict(loss=loss_val))
                pbar.update(1)

                # periodically save the policy
                if steps % 2000 == 0:
                    save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
                if steps >= iterations:
                    break

            epoch_loss = epoch_loss / len(dataloader)

            # save a new model if the average MSE loss in an epoch has improved
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
            if epoch % 50 == 0:
                print("Evaluating")
                success_rate = evaluate_policy(env, policy)
                writer.add_scalar("test/success_rate", success_rate, epoch)
            writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
            epoch += 1
        save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))

    # run a final evaluation
    success_rate = evaluate_policy(env, policy)
    print(f"Final Success Rate {success_rate}")


if __name__ == "__main__":
    main()