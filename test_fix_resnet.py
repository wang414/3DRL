# Import required packages
import argparse
import os.path as osp
from functools import partial
import json
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch as th
import torch.nn as nn
# from stable_baselines3 import PPO
from module.recurrent_ppo import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import glob
import re
import pickle, os
import mani_skill2.envs
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from torchvision.models import resnet18
from stable_baselines3.common.save_util import load_from_zip_file

# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


class ManiSkillRGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = self.init_observation_space(env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        # States include robot proprioception (agent) and task information (extra)
        # NOTE: SB3 does not support nested observation spaces, so we convert them to flat spaces
        state_spaces = []
        state_spaces.extend(flatten_dict_space_keys(obs_space["agent"]).spaces.values())
        state_spaces.extend(flatten_dict_space_keys(obs_space["extra"]).spaces.values())
        # Concatenate all the state spaces
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-np.inf, np.inf, shape=(state_size,))

        # Concatenate all the image spaces
        image_shapes = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            image_shapes.append(cam_space["rgb"].shape)
            # image_shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(image_shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(h, w, c))

        # Create the new observation space
        return spaces.Dict({"rgb": rgbd_space, "state": state_space})

    @staticmethod
    def convert_observation(observation):
        # Process images. RGB is normalized to [0, 1].
        images = []
        for cam_uid, cam_obs in observation["image"].items():
            rgb = cam_obs["rgb"] / 255.0
            depth = cam_obs["depth"]

            # NOTE: SB3 does not support GPU tensors, so we transfer them to CPU.
            # For other RL frameworks that natively support GPU tensors, this step is not necessary.
            if isinstance(rgb, th.Tensor):
                rgb = rgb.to(device="cpu", non_blocking=True)
            if isinstance(depth, th.Tensor):
                depth = depth.to(device="cpu", non_blocking=True)

            images.append(rgb)
            # images.append(depth)

        # Concatenate all the images
        rgbd = np.concatenate(images, axis=-1)

        # Concatenate all the states
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        return dict(rgb=rgbd, state=state)

    def observation(self, observation):
        return self.convert_observation(observation)


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, emb_module='resnet'):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        assert emb_module in ['CNN', 'resnet'], 'embedding module not define'
        total_concat_size = 0
        feature_size = 128

        for key, subspace in observation_space.spaces.items():
            # We go through all subspaces in the observation space.
            # We know there will only be "rgbd" and "state", so we handle those below
            if key == "rgb":
                # here we use a NatureCNN architecture to process images, but any architecture is permissble here
                in_channels = subspace.shape[-1]
                if emb_module == 'CNN':
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
                            in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2,
                            padding=0,
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                        ),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    # to easily figure out the dimensions after flattening, we pass a test tensor
                    test_tensor = th.zeros(
                        [subspace.shape[2], subspace.shape[0], subspace.shape[1]]
                    )
                    with th.no_grad():
                        n_flatten = cnn(test_tensor[None]).shape[1]
                    fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                    extractors["rgb"] = nn.Sequential(cnn, fc)
                if emb_module == 'resnet':
                    cnn1 = resnet18(pretrained=True)
                    cnn2 = resnet18(pretrained=True)
                    num_ftrs = cnn1.fc.in_features
                    cnn1.fc = nn.Linear(num_ftrs, feature_size//2)    
                    cnn2.fc = nn.Linear(num_ftrs, feature_size//2)    
                    cnn1.requires_grad_(False)
                    cnn2.requires_grad_(False)
                    cnn1.fc.requires_grad_(True)
                    cnn1.fc.requires_grad_(True)
                    extractors["rgb"] = nn.ModuleList([nn.Sequential(cnn1, nn.ReLU()), nn.Sequential(cnn2, nn.ReLU())])
                total_concat_size += feature_size
            elif key == "state":
                # for state data we simply pass it through a single linear layer
                state_size = subspace.shape[0]
                extractors["state"] = nn.Linear(state_size, 64)
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "rgb":
                observations[key] = observations[key].permute((0, 3, 1, 2))
                encoded_tensor_list.append(extractor[0](observations[key][:,:3,:,:]))
                encoded_tensor_list.append(extractor[1](observations[key][:,3:,:,:]))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def eval(model_path, model_id=None):

    env_id = "PickSingleYCB-v0"
    log_dir = None
    max_episode_steps = 50
    rollout_steps = 3200

    obs_mode = "rgbd"
    # NOTE: The end-effector space controller is usually more friendly to pick-and-place tasks
    control_mode = "pd_ee_delta_pose"
    use_ms2_vec_env = False

    # define a make_env function for Stable Baselines
    def make_env(
        env_id: str,
        max_episode_steps=None,
        record_dir: str = None,
    ):
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs

        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="cameras",
            max_episode_steps=max_episode_steps,
            model_ids = model_id,
        )
        env = ManiSkillRGBDWrapper(env)
        # For evaluation, we record videos
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env,
                record_dir,
                save_trajectory=False,
                info_on_video=True,
            )
        return env

    # Create an environment for evaluation
    record_dir = None
    
    env_fn = partial(
        make_env,
        env_id,
        record_dir=record_dir,
        max_episode_steps = max_episode_steps
    )
    eval_env = SubprocVecEnv([env_fn for _ in range(5)])
    eval_env = VecMonitor(eval_env)  # Attach a monitor to log episode info
    # eval_env.seed(int(time.time())%99991)
    # print(time.time())
    eval_env.reset()
    # return
    env = eval_env
    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor, net_arch=[256, 128], log_std_init=-0.5
    )
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        # n_steps=rollout_steps // num_envs,
        batch_size=400,
        n_epochs=5,
        gamma=0.8,
        target_kl=0.2,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model_path = model_path
    # Load the saved model
    model.set_parameters(model_path)
        # Evaluate the model
    returns, max_rewards = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=5,
    )
    # print("Returns", returns)
    # print("Episode Lengths", ep_lens)
    # print("max_rewards", max_rewards)
    # close all envs
    eval_env.close()
    return returns, max_rewards
   
with open('data/mani_skill2_ycb/info_pick_v0.json', 'r') as f:
    obj_dict = json.load(f)

def eval_all_obj(model_path):
    result = {}
    for key in obj_dict.keys():
        me, ma = eval(model_path=model_path, model_id=key)
        result[key] = [me, ma]
    return result

def extract_number(filename):
    match = re.search(r'rl_model_(\d+)_steps.zip', filename)
    if match:
        return int(match.group(1))
    return None 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    args = parser.parse_args()
    folder_path = args.input_folder
    pattern = f"{folder_path}/rl_model_*_steps.zip"
    matching_files = glob.glob(pattern)

    sorted_files = sorted(matching_files, key=extract_number)

    timesteps = []
    results = []
    for file in sorted_files:
        timesteps.append(extract_number(file))
        res = eval_all_obj(file)
        results.append(res)

    with open(os.path.join(folder_path, 'eval_data.pkl'), 'wb') as f:
        pickle.dump({'result':results, 'timesteps':timesteps}, f)