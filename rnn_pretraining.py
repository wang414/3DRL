import argparse
import os.path as osp
from functools import partial
import json, os, time
import sys
from icecream import ic
import mani_skill2.envs
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from stable_baselines3 import PPO
from module.pretrain_ppo import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from PIL import Image
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from torchvision.models import resnet18
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import open3d as o3d
import subprocess, threading
from PyTorchEMD import earth_mover_distance
# Defines a continuous, infinite horizon, task where terminated is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


class PretrainWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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
        rgb_space = spaces.Box(0, np.inf, shape=(h, w, c))
        
        # depth_space and mask space
        mask_space = spaces.Box(0, 255, shape=(h, w, 1), dtype=bool)
        depth_space = spaces.Box(-np.inf, np.inf, shape=(h, w, 1))

        # Create the new observation space
        return spaces.Dict({"rgb": rgb_space, "depth":depth_space, "mask":mask_space, "state": state_space})

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
        rgb = np.concatenate(images, axis=-1)

        # Concatenate all the states
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        depth = observation["image"]["hand_camera"]["depth"]
        mask = observation["image"]["hand_camera"]["obj_seg"]
        if isinstance(depth, th.Tensor):
            depth = depth.to(device="cpu", non_blocking=True)
        return dict(rgb=rgb, depth=depth, mask=mask, state=state)

    def observation(self, observation):
        return self.convert_observation(observation)

class ManiSkillRGBDVecEnvWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        # assert env.obs_mode == "rgbd"
        # we simply define the single env observation space. The inherited wrapper automatically computes the batched version
        single_observation_space = PretrainWrapper.init_observation_space(
            env.single_observation_space
        )
        super().__init__(env, single_observation_space)

    def observation(self, observation):
        return PretrainWrapper.convert_observation(observation)

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, emb_module='CNN'):
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
                if emb_module == 'resnet':
                    cnn = resnet18(pretrained=True)
                    cnn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    num_ftrs = cnn.fc.in_features
                    cnn.fc = nn.Linear(num_ftrs, feature_size)    
                # to easily figure out the dimensions after flattening, we pass a test tensor
                test_tensor = th.zeros(
                    [subspace.shape[2], subspace.shape[0], subspace.shape[1]]
                )
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors["rgb"] = nn.Sequential(cnn, fc)
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
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

class RewardCallback:
    def __init__(self, K, n_envs, max_episode_len, pointnet_path='/home/lab/ManiSkill2/Pointnet2_PyTorch/outputs/cls-ssg-xyz/epoch=16-val_loss=0.36-val_acc=0.885.ckpt', stride=5):
        os.system(f'rm -rf .tmp && mkdir .tmp')
        self.K = K
        self.stride = stride
        self.model = PointNet2ClassificationSSG({'model.use_xyz':True})
        self.model.load_from_checkpoint(pointnet_path)
        self.model = self.model.to('cuda')
        self.model.eval()
        with open('data/mani_skill2_ycb/info_pick_v0.json', 'r') as f:
            obj_dict = json.load(f)
        data_dir = 'data/mani_skill2_ycb/models'
        self.gt_pcds = {}
        for key in obj_dict.keys():
            pcd_dir = os.path.join(data_dir, key, 'pcd.ply')
            pcd = o3d.io.read_point_cloud(pcd_dir)
            pcd = np.asarray(pcd.points, dtype=np.float32)
            self.gt_pcds[key] = pcd
        self.n_envs = n_envs
        self.max_episode_len = max_episode_len
        self.rgb_buf = [[] for _ in range(n_envs)]
        self.masks_buf = [[] for _ in range(n_envs)]
        self.dep_buf = [[] for _ in range(n_envs)]
        self.obj_buf = [[] for _ in range(n_envs)]

    def cosine_similarity(self, vector1, vector2):
        # print(vector1.shape)
        vector1 = vector1.squeeze(-1)
        vector2 = vector2.squeeze(-1)
        similarity = F.cosine_similarity(vector1, vector2, dim=1)
        return similarity.item()

    def sample_idx(self, len):
        # print(len)
        cat_sam = np.random.choice(4, p=[0.3,0.3,0.3,0.1])
        if cat_sam == 0:
            st = 0
        elif cat_sam == 1:
            st = 10
        elif cat_sam == 2:
            st = 20   
        if cat_sam == 3:
            st = 0
            ed = len
        else:
            ed = min(len, st+30)
        l = np.arange(st, ed)
        np.random.shuffle(l)
        l = l[:10]
        l = np.sort(l)
        return l

    def compute(self, rgbs, depths, masks, obj, tmp_dir = './.tmp'):
        
        # start_time = time.time()
        os.system(f'rm -rf {tmp_dir} && mkdir {tmp_dir}')
        os.system(f"mkdir {os.path.join(tmp_dir, 'input')}")
        depths = depths.squeeze(-1)
        masks = masks.squeeze(-1)
        idxs = self.sample_idx(rgbs.shape[0])
        # print(idxs)
        rgbs = (rgbs[idxs,:,:,3:] * 255).astype(np.uint8)
        depths = depths[idxs]
        masks = masks[idxs].astype(np.uint8)
        np.savez(os.path.join(tmp_dir, 'input', 'input.npz'), rgbs=rgbs, depths=depths, masks=masks, K=self.K)
        # pcd = run_one_video(manireader, tmp_dir)
        # import time
        # time.sleep(0.5)
        command = f"python bundlesdf_runner.py --input_file {os.path.join(tmp_dir, 'input', 'input.npz')} --output_folder {os.path.join(tmp_dir, 'result')} --unit 1"
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        process.wait()
        # finish1_time = time.time()
        pcd_path = os.path.join(tmp_dir,'result','nerf_with_bundletrack_online', 'naive_fusion_biggest_cluster.ply')
        build_succ = True
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            gt_pcd = self.gt_pcds[obj]
            # chamfer_dis = pcd.compute_point_cloud_distance(gt_pcd)
            # chamfer_dis = np.asarray(chamfer_dis).sum().item()
            pcd = np.asarray(pcd.points, dtype=np.float32)
            pcd = th.tensor(pcd, device='cuda').unsqueeze(0)
            gt_pcd = th.tensor(np.asarray(self.gt_pcds[obj]), device='cuda', dtype=th.float32).unsqueeze(0)
            emd_dis = earth_mover_distance(pcd, gt_pcd)
            # ic(chamfer_dis)
            # ic(emd_dis)
            clip_dis = max(min(emd_dis*200, 10), 0)
        else:
            clip_dis = 10
            build_succ = False
        # finish2_time = time.time()
        # print(f'construct_time = {finish1_time-start_time}')
        # print(f'dis_time = {finish2_time-finish1_time}')
        return 10 - clip_dis, build_succ
    
    def reset_buf(self, idx):
        self.rgb_buf[idx] = []
        self.masks_buf[idx] = []
        self.dep_buf[idx] = []
        self.obj_buf[idx].pop(0)

    def add_obj(self, obj, idx):
        self.obj_buf[idx].append(obj)

    def add_buf(self, rgb, dep, mask, idx):
        self.rgb_buf[idx].append(rgb)
        self.masks_buf[idx].append(mask)
        self.dep_buf[idx].append(dep)

    def multi_thread_task(self, rollout_buffer, idx, stride, lock):
        for t in range(rollout_buffer.buffer_size):
            # ic(rollout_buffer.episode_starts[t])
            for i in range(idx*stride, idx*stride + stride):
                if rollout_buffer.episode_starts[t, i] and t > 0:
                    rw, bs = self.compute(np.array(self.rgb_buf[i]), np.array(self.dep_buf[i]),
                            np.array(self.masks_buf[i], dtype=np.uint8), self.obj_buf[i][0], tmp_dir=f'./.tmp/{idx}')
                    with lock:
                        self.reset_buf(i)
                        rollout_buffer.rewards[t-1, i] = rw
                        self.total_bs += bs
                self.add_buf(rollout_buffer.observations['rgb'][t,i], rollout_buffer.observations['depth'][t,i],
                              rollout_buffer.observations['mask'][t,i], i)
                if t+1 == rollout_buffer.buffer_size and len(self.rgb_buf[i]) == self.max_episode_len:
                    rw, bs = self.compute(np.array(self.rgb_buf[i]), np.array(self.dep_buf[i]), 
                            np.array(self.masks_buf[i], dtype=np.uint8), self.obj_buf[i][0], tmp_dir=f'./.tmp/{idx}')
                    with lock:
                        self.reset_buf(i)
                        rollout_buffer.rewards[t, i] = rw
                        self.total_bs += bs
                    

    def __call__(self, rollout_buffer):
        rollout_buffer.rewards *= 0
        self.total_bs = 0
        threads = []
        lock = threading.Lock()
        for i in range(2):
            t = threading.Thread(target=self.multi_thread_task, args=(rollout_buffer, i, 4, lock))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.total_bs

    def single_thread(self, rollout_buffer):  
        # ic(rollout_buffer.buffer_size)
        rollout_buffer.rewards *= 0
        self.total_bs = 0
        for t in range(rollout_buffer.buffer_size):
            # ic(rollout_buffer.episode_starts[t])
            for i in range(rollout_buffer.n_envs):
                if rollout_buffer.episode_starts[t, i] and t > 0:
                    rollout_buffer.rewards[t-1, i], bs = self.compute(np.array(self.rgb_buf[i]), np.array(self.dep_buf[i]),
                                        np.array(self.masks_buf[i], dtype=np.uint8), self.obj_buf[i][0])
                    self.reset_buf(i)
                    self.total_bs += bs
                self.add_buf(rollout_buffer.observations['rgb'][t,i], rollout_buffer.observations['depth'][t,i],
                              rollout_buffer.observations['mask'][t,i], i)
                if t+1 == rollout_buffer.buffer_size and len(self.rgb_buf[i]) == self.max_episode_len:
                    rollout_buffer.rewards[t, i], bs = self.compute(np.array(self.rgb_buf[i]), np.array(self.dep_buf[i]), 
                                        np.array(self.masks_buf[i], dtype=np.uint8), self.obj_buf[i][0])
                    self.reset_buf(i)
                    self.total_bs += bs
        return self.total_bs
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=256_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    parser.add_argument(
        '--finite', action='store_false', help="train with finity setting"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    env_id = args.env_id
    num_envs = args.n_envs
    log_dir = args.log_dir
    max_episode_steps = args.max_episode_steps
    total_timesteps = args.total_timesteps
    rollout_steps = 3200

    obs_mode = ["rgbd", "obj_seg"]
    # NOTE: The end-effector space controller is usually more friendly to pick-and-place tasks
    control_mode = "pd_ee_delta_pose"
    use_ms2_vec_env = False

    if args.seed is not None:
        set_random_seed(args.seed)

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
            camera_cfgs={'add_segmentation': True},
        )
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        env = PretrainWrapper(env)
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
    if args.eval:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")
    env_fn = partial(
        make_env,
        env_id,
        record_dir=record_dir,
        max_episode_steps = max_episode_steps
    )
    eval_env = SubprocVecEnv([env_fn for _ in range(1)])
    eval_env = VecMonitor(eval_env)  # Attach a monitor to log episode info
    eval_env.seed(seed=args.seed)
    eval_env.reset()

    if args.eval:
        env = eval_env
    else:
        # Create vectorized environments for training
        if use_ms2_vec_env:
            env: VecEnv = make_vec_env(
                env_id,
                num_envs,
                obs_mode=obs_mode,
                control_mode=control_mode,
                wrappers=[partial(ContinuousTaskWrapper)],
                max_episode_steps=max_episode_steps,
            )
            env = ManiSkillRGBDVecEnvWrapper(env)
            env = SB3VecEnvWrapper(
                env
            )  # makes MS2VecEnvs compatible with SB3. It's equivalent to SubprocVecEnv
        else:
            env_fn = partial(
                make_env,
                env_id,
                max_episode_steps=max_episode_steps,
            )
            env = SubprocVecEnv([env_fn for _ in range(num_envs)])
        # Attach a monitor to log episode info
        env = VecMonitor(env)
        env.seed(seed=args.seed)  # Note SB3 vec envs don't use the gymnasium API
        env.reset()
    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor, net_arch=[256, 128], log_std_init=-0.5
    )
    ic('prepare to setup PPO')
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=rollout_steps // num_envs,
        batch_size=400,
        n_epochs=5,
        gamma=0.8,
        target_kl=0.2,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        # ent_coef=
        verbose=1,
    )

    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path)
    else:
        # Define callbacks to periodically save our model and evaluate it to help monitor training
        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
            verbose=2
        )
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=10 * rollout_steps // num_envs,
            log_path=log_dir,
            best_model_save_path=log_dir,
            deterministic=True,
            render=False,
        )
        K = np.array([[64,0,64],[0,64,64],[0,0,1]], dtype=np.float32)
        reward_callback = RewardCallback(K, num_envs, max_episode_steps)
        # reward_callback = None
        # Train an agent with PPO
        model.learn(total_timesteps, log_interval=10, callback=[checkpoint_callback, eval_callback], reward_callback=reward_callback)
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=5,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)

    # close all envs
    eval_env.close()
    if not args.eval:
        env.close()


if __name__ == "__main__":
    main()
