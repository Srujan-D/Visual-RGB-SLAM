import os
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.ma_solver import Solver

from vggt.models.vggt import VGGT
import pickle

import sys
sys.path.append('/home/srujan/work/map-anything')

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

import rerun as rr

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="mapanything_slam_results/gr_hq_living_room/poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=8, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=125, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out") # 0.0 means no filtering for GROUND TRUTH DEPTH

parser.add_argument(
    "--memory_efficient_inference",
    action="store_true",
    default=False,
    help="Use memory efficient inference for reconstruction (trades off speed)",
)
parser.add_argument(
    "--apache",
    action="store_true",
    help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
)
parser.add_argument(
    "--viz",
    action="store_true",
    default=False,
    help="Enable visualization with Rerun",
)
parser.add_argument(
    "--save_glb",
    action="store_true",
    default=False,
    help="Save reconstruction as GLB file",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="output.glb",
    help="Output path for GLB file (default: output.glb)",
)

def extract_frame_data(solver):
    """
    Extract frame data from VGGT-SLAM results.

    Args:
        solver: Completed VGGT-SLAM solver

    Returns:
        frame_data: List of frame dictionaries with poses, point clouds, etc.
    """
    frame_data = []

    for submap in solver.map.get_submaps():
        submap_id = submap.get_id()
    
        # Get local poses (relative to submap reference frame)
        local_poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
        frame_ids = submap.get_frame_ids()
        images = submap.get_all_frames()
        
        # Get point clouds and confidence masks
        point_clouds, _, conf_masks = submap.get_points_list_in_world_frame(
            ignore_loop_closure_frames=True
        )
        
        # Get intrinsics (same for all frames in submap)
        intrinsics = submap.vggt_intrinscs
        
        for i, (local_pose, frame_id, image, points, conf_mask) in enumerate(
            zip(local_poses, frame_ids, images, point_clouds, conf_masks)
        ):
            # Filter points by confidence
            points_flat = points.reshape(-1, 3)
            conf_flat = conf_mask.reshape(-1)
            confident_points = points_flat[conf_flat]
            
            frame_data.append({
                'frame_id': frame_id,
                'submap_id': submap_id,
                'local_pose': local_pose,  # Keep local for reference
                'image': image.contiguous().cpu().numpy(),
                'points_3d': confident_points,
                'intrinsics': intrinsics,
                'conf_mask': conf_mask,
            })

    # # plot all global poses for all frames of all submaps
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot trajectory as a connected line
    # poses = np.array([frame['pose'] for frame in frame_data])
    # positions = poses[:, :3, 3]  # Extract translation components
    
    # # Plot 3D trajectory
    # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    # ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
    # ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='s', label='End')
    
    # # Plot global pose of each frame to get continious global trajectory
    # for frame in frame_data:
    #     pose = frame['pose']
    #     ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], c='b', s=10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.title('Global 3D Trajectory (All Frames)')
    # plt.show()
        

    # # plot trajectory of local poses of all frames of all submaps
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot trajectory as a connected line
    # local_poses = np.array([frame['local_pose'] for frame in frame_data])
    # local_positions = local_poses[:, :3, 3]  # Extract translation components
    # ax.plot(local_positions[:, 0], local_positions[:, 1], local_positions[:, 2], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    # ax.scatter(local_positions[0, 0], local_positions[0, 1], local_positions[0, 2], c='g', s=100, marker='o', label='Start')
    # ax.scatter(local_positions[-1, 0], local_positions[-1, 1], local_positions[-1, 2], c='r', s=100, marker='s', label='End')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.title('Local 3D Trajectory (Submap Frames)')
    # plt.show()

    # # 2d projection of trajectory
    # plt.figure()
    # xs = [frame['pose'][0, 3] for frame in frame_data]
    # ys = [frame['pose'][1, 3] for frame in frame_data]
    # plt.plot(xs, ys, marker='o')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Trajectory Projection')
    # plt.grid()
    # plt.show()

    # # 2d projection of trajectory colored by submap_id
    # plt.figure()
    # submap_ids = [frame['submap_id'] for frame in frame_data]
    # scatter = plt.scatter(xs, ys, c=submap_ids, cmap='tab10', marker='o')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Trajectory Projection Colored by Submap ID')
    # plt.colorbar(scatter, label='Submap ID')
    # plt.grid()
    # plt.show()

    return frame_data


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):

    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
        )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )



def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()
    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")


    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    t = 0
    
    predictions = None
    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, args)

            # Prepare lists for GLB export if needed
            world_points_list = []
            images_list = []
            masks_list = []
            # Initialize Rerun if visualization is enabled
            if args.viz:
                print("Starting visualization...")
                viz_string = "MapAnything_Visualization"
                rr.script_setup(args, viz_string)
                rr.set_time("stable_time", sequence=t)
                rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

                # Loop through the outputs for visualization only
                for view_idx, pred in enumerate(predictions["outputs"]):
                    # Extract data from predictions
                    depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
                    intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
                    camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

                    # Compute new pts3d using depth, intrinsics, and camera pose
                    pts3d_computed, valid_mask = depthmap_to_world_frame(
                        depthmap_torch, intrinsics_torch, camera_pose_torch
                    )

                    # Convert to numpy arrays
                    mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
                    mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
                    pts3d_np = pts3d_computed.cpu().numpy()
                    image_np = pred["img_no_norm"][0].cpu().numpy()

                    # Store data for GLB export if needed
                    if args.save_glb:
                        world_points_list.append(pts3d_np)
                        images_list.append(image_np)
                        masks_list.append(mask)

                    # Log to Rerun
                    log_data_to_rerun(
                        image=image_np,
                        depthmap=depthmap_torch.cpu().numpy(),
                        pose=camera_pose_torch.cpu().numpy(),
                        intrinsics=intrinsics_torch.cpu().numpy(),
                        pts3d=pts3d_np,
                        mask=mask,
                        base_name=f"mapanything/view_{view_idx}",
                        pts_name=f"mapanything/pointcloud_view_{view_idx}",
                        viz_mask=mask,
                    )

            t += 1

            # Add points to solver ONCE per submap (not per view)
            solver.add_points(predictions)
            # print("=== BEFORE OPTIMIZATION ===")
            # print(f"Number of submaps: {solver.map.get_num_submaps()}")
            # for i, submap in enumerate(solver.map.get_submaps()):
            #     print(f"Submap {submap.get_id()} reference homography translation: {submap.get_reference_homography()[:3, 3]}")

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            # print("=== AFTER OPTIMIZATION ===")
            # for i, submap in enumerate(solver.map.get_submaps()):
            #     print(f"Submap {submap.get_id()} reference homography translation: {submap.get_reference_homography()[:3, 3]}")

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    frame_data = extract_frame_data(solver)
    # save frame data, solver, depthmaps so that we dont need to run vggt-slam again while debugging graf
    os.makedirs("mapanything_slam_results/gr_hq_living_room/", exist_ok=True)
    with open(os.path.join("mapanything_slam_results/gr_hq_living_room/", "mapanything_frame_data.pkl"), "wb") as f:
        pickle.dump(frame_data, f)

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        solver.map.write_poses_to_file(args.log_path)

        # Log the full point cloud as one file, used for visualization.
        solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))


if __name__ == "__main__":
    main()
