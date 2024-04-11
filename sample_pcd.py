import json
import numpy as np
import open3d as o3d
import os

with open('data/mani_skill2_ycb/info_pick_v0.json', 'r') as f:
    obj_dict = json.load(f)
data_dir = 'data/mani_skill2_ycb/models'
for key in obj_dict.keys():
    mesh_dir = os.path.join(data_dir, key, 'textured.obj')
    mesh = o3d.io.read_triangle_mesh(mesh_dir)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    # pcd = np.asarray(pcd.points)
    o3d.io.write_point_cloud(os.path.join(data_dir, key, 'pcd.ply'), pcd)
    