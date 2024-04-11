import argparse
import os.path as osp
from functools import partial
import json, os, re
import sys
from PIL import Image
from icecream import ic
import numpy as np
sys.path.append('../BundleSDF')
from run_nerf import ManiImageReader, run_one_video

# from torchvision.models import resnet18
# from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
# import open3d as o3d

if __name__=="__main__":
#   set_seed(0)
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', type=str, default="./.tmp/input.npz")
  parser.add_argument('--output_folder', type=str, default="./.tmp")
  parser.add_argument('--unit', default=1, help='default depth unit 1 mm')
  args = parser.parse_args()

  load_data = np.load(args.input_file)

  # rgbs = []
  # depth = []
  # masks = []
  # files = sorted(os.listdir(os.path.join(args.video_dir,'rgb')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  # for file in files:
  #   if file.endswith('.png'):
  #     rgbs.append(np.array(Image.open(os.path.join(args.video_dir, 'rgb', file))))

  # files = sorted(os.listdir(os.path.join(args.video_dir,'masks')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  # for file in files:
  #   if file.endswith('.png'):
  #     masks.append(np.array(Image.open(os.path.join(args.video_dir, 'masks', file))))

  # files = sorted(os.listdir(os.path.join(args.video_dir,'depth')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  # for file in files:
  #   if file.endswith('.png'):
  #     depth.append(np.array(Image.open(os.path.join(args.video_dir, 'depth', file))))
  # rgbs = np.stack(rgbs, axis=0)[:,:,:,::-1]
  # masks = np.stack(masks, axis=0)
  # depth = np.stack(depth, axis=0)
  # K = np.loadtxt(os.path.join(args.video_dir, 'cam_K.txt')).reshape(3,3)
  rgbs = load_data['rgbs'].copy()
  depth = load_data['depths'].copy()
  masks = load_data['masks'].copy()
  K = load_data['K'].copy()
  # print(rgbs.shape)
  # print(depth.shape)
  # print(masks.shape)
  # print(K.shape)
  # exit()
  manireader = ManiImageReader(rgbs, depth, masks, K, unit=1)
  run_one_video(manireader, out_folder=args.output_folder, stride=1, debug_level=1)