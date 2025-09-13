# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

'''
from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
'''
import yaml
#from foundationpose.Utils import NestDict, make_yaml_dumpable

from foundationpose.datareader import LinemodReader
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd
from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime
#from metrics import *


def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid



def run_pose_estimation_worker(reader, i_frames, est:Any6D=None, debug=0, ob_id=None, device='cuda'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id() # '000002'
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H,W = color.shape[:2]

    debug_dir =est.debug_dir

    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")

    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result


def run_pose_estimation():
  wp.force_load(device='cuda')
  reader = LinemodReader(f'{opt.linemod_dir}/test/000002', split=None)

  debug = save_path
  use_reconstructed_mesh = opt.use_reconstructed_mesh

  res = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
  mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
  est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_path, debug=0, glctx=glctx)
#  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
#  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  for ob_id in reader.ob_ids:  # ids: [1,2,4,5,6,8,9,10,11,12,13,14,15]
    ob_id = int(ob_id)
    if use_reconstructed_mesh:
      mesh = trimesh.load(reader.get_reference_view_1_mesh(opt.anchor_path))
      ######## Needs to be defined after the anchor method is created ########
    else:
      mesh = reader.get_gt_mesh(ob_id)
    symmetry_tfs = reader.symmetry_tfs[ob_id]

    args = []

    #video_dir = f'{opt.linemod_dir}/test/{ob_id:06d}'
    #reader = LinemodReader(video_dir, split=None)
    #video_id = reader.get_video_id()
    video_id = '000002'
    est.reset_object(mesh=mesh, symmetry_tfs=None)
    #est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)

    for i in range(len(reader.color_files)):
      args.append((reader, [i], est, debug, ob_id, "cuda"))

    outs = []
    for arg in args:
      out = run_pose_estimation_worker(*arg)
      outs.append(out)

    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          for ob_id in out[video_id][id_str]:
            res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

  with open(f'{debug}/linemod_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)


if __name__=='__main__':
  
  seed_everything(0)
  
  parser = argparse.ArgumentParser(description="Set experiment name and paths")
  
  #parser.add_argument("--name", type=str, default="any6d", help="Experiment name")
  parser.add_argument("--anchor_path", type=str, default="/home/../Experiments/simonep01/linemod_anchors/", help="Path to anchor files")
  parser.add_argument('--linemod_dir', type=str, default="/home/../DATA/bop_classic/lmo", help="linemod root dir")
#  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  opt = parser.parse_args()

  detect_type = 'mask'   # mask / box / detected

  date_str = f'{datetime.now():%Y-%m-%d_%H-%M}'
  save_root = f"./results/linemod_results/{date_str}"
  save_path = f'{save_root}'

  os.makedirs(save_path, exist_ok=True)

  run_pose_estimation()
