import os
import trimesh
import numpy as np
import cv2
import time

import nvdiffrast.torch as dr
import argparse
import pandas as pd

from estimater import Any6D
from foundationpose.Utils import visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
from tqdm import tqdm
from sam2_instantmesh import *


# python run_ho3d_anchor_timed.py --anchor_folder anchor_results/dexycb_reference_view_ours --ycb_model_path ~/../../Experiments/simonep01/ho3d --img_to_3d

def print_timing(stage_name, elapsed_time, obj_name=None):
    """Helper function to print timing information"""
    obj_str = f" [{obj_name}]" if obj_name else ""
    print(f"⏱️  {stage_name}{obj_str}: {elapsed_time:.2f} seconds")

if __name__=='__main__':
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--anchor_folder", type=str, default="demo_data/light_ho3d/make_anchors", help="Path to anchor files")
    parser.add_argument("--ycb_model_path", type=str, default="/home/../Experiments/simonep01/demo_data/light_ho3d", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    args = parser.parse_args()

    anchor_folder = args.anchor_folder
    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d


    results = []
    obj_list = [f for f in os.listdir(anchor_folder) if not f.endswith('.xlsx')]
    glctx = dr.RasterizeCudaContext()

    # Track timing for each stage across all objects
    timing_stats = {
        'sam2_instantmesh': [],
        'any6d': [],
        'results': []
    }

    for obj in tqdm(obj_list, desc='Object'):
        obj_start_time = time.time()
        print(f"\n🔄 Processing object: {obj}")

        if obj == '006_mustard_bottle':
            obj_num = 5
        elif obj == '021_bleach_cleanser':
            obj_num = 12
        elif obj == '019_pitcher_base':
            obj_num = 11
        #elif obj == '004_sugar_box':
        #    obj_num = 3
        #elif obj == '005_tomato_soup_can':
        #    obj_num = 4
        elif obj == '010_potted_meat_can':
            obj_num = 9

        save_path = f'{anchor_folder}/{obj}'
        mesh_path = os.path.join(f'{anchor_folder}/{obj}/mesh_{obj}.obj')


        color = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'color.png')), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(save_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        mask = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'mask.png')),cv2.COLOR_BGR2RGB)[...,0].astype(np.bool_)

        # 2. SAM2 + InstantMesh Processing (if enabled)
        if img_to_3d:
            stage_start = time.time()
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            mask_refine = running_sam_box(color, input_box)

            input_image = preprocess_image(color, mask_refine, save_path, obj)
            images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
            instant_mesh_process(images, save_path, obj)
            
            sam_time = time.time() - stage_start
            timing_stats['sam2_instantmesh'].append(sam_time)
            print_timing("SAM2 + InstantMesh Processing", sam_time, obj)

            mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        else:
            timing_stats['sam2_instantmesh'].append(0)
            mesh = trimesh.load(mesh_path)

        # 3. Any6D
        stage_start = time.time()
        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))


        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=0)
        intrinsic = np.loadtxt(f'{anchor_folder}/{obj}/K.txt')

        pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')
        elapsed = time.time() - stage_start
        timing_stats['any6d'].append(elapsed)
        print_timing("Any6D", elapsed, obj)

        # 6. Results
        stage_start = time.time()
        gt_pose = np.loadtxt(os.path.join(save_path, f'{obj}_gt_pose.txt'))
        gt_mesh = trimesh.load(f'{ycb_model_path}/models/{obj}/textured_simple.obj')


        visualize_frame_results(color=color, gt_mesh=gt_mesh, est=est, K=intrinsic, gt_pose=gt_pose, pred_pose=pred_pose,
                                metric=None, obj_f=obj, frame_idx=0, save_path=save_path, glctx=glctx,
                                name=f'demo_data', mesh_index=0, init=False, save_on_folder=True)



        chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh)
        print("CF: ", chamfer_dis)


        np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
        est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))
        np.savetxt(os.path.join(save_path, f'{obj}_cd.txt'), [chamfer_dis])
        elapsed = time.time() - stage_start
        timing_stats['results'].append(elapsed)
        print_timing("Results", elapsed, obj)

        results.append({
            'Object': obj,
            'Object_Number': obj_num,
            'Chamfer_Distance': float(chamfer_dis)
            })

        obj_total_time = time.time() - obj_start_time
        print_timing(f"TOTAL for {obj}", obj_total_time)

    df = pd.DataFrame(results)
    df = df.sort_values('Object')
    excel_path = os.path.join(anchor_folder, 'chamfer_distances.xlsx')
    df.to_excel(excel_path, index=False)

    # Print timing summary statistics
    total_time = time.time() - total_start_time
    print_timing("TOTAL EXECUTION TIME", total_time)
    
    print("\n📊 TIMING SUMMARY STATISTICS:")
    print("="*50)
    for stage, times in timing_stats.items():
        if times and any(t > 0 for t in times):
            times_array = np.array(times)
            print(f"{stage.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(times_array):.2f}s | Total: {np.sum(times_array):.2f}s")
            print(f"  Min: {np.min(times_array):.2f}s | Max: {np.max(times_array):.2f}s")
            print(f"  % of total: {(np.sum(times_array)/total_time)*100:.1f}%")
            print()

    print("\nChamfer Distance Summary Statistics:")
    print(df['Chamfer_Distance'].describe())
    print(f"\nResults saved to: {excel_path}")