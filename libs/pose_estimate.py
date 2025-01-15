"""
This module provides classes and functions to handle 2D detections in an image
and map them to the corresponding 3D points in a point cloud. We rely on a
'**direct indexing**' approach rather than a traditional camera projection:

1. We store a global index for each 3D point, referencing its position in the
   original point cloud (row-major order, possibly scaled).
2. Operations like Z-filtering, voxel downsampling, or clustering preserve
   an updated subset of these global indices.
3. We then map each final 3D point back to an (u, v) position in the 2D image
   using a simple row/column calculation, assuming a known scale factor.

If proper camera calibration becomes available in the future, we can switch to a
'pinhole projection' model, removing the need to maintain global indices.
"""

from typing import List, Optional, Tuple, cast

import numpy as np
import open3d as o3d

from libs.dataset.data_structure import Detection2D, Detection3D, Frame, Pose3D


class PointCloudProcessor:
    """
    Point cloud processing pipeline that filters, downsamples, and clusters
    a point cloud while preserving global indices for direct 2D-3D mapping.

    We maintain a 'global_indices' array that references each local point
    in the current point cloud to the original, full-resolution point cloud.
    """

    def __init__(self, configuration: dict) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> dict:
        """Return the configuration dictionary."""
        return self._configuration

    def processes(
        self, pcd: o3d.geometry.PointCloud, global_indices: Optional[np.ndarray] = None
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Pipeline applying Z-filter, voxel downsample, and N-largest clustering,
        preserving a global index map back to the original cloud.
        """
        # If global_indices is None, create a fresh array referencing the entire original pcd
        if global_indices is None:
            global_indices = np.arange(len(pcd.points))

        if self.configuration["use_z_distance_filter"]:
            pcd, global_indices = PointCloudProcessor.filter_points_by_z(
                pcd,
                self.configuration["z_filter_threshold_min"],
                self.configuration["z_filter_threshold_max"],
                global_indices,
            )

        if self.configuration["use_voxel_downsample"]:
            pcd_down, downsample_lists = PointCloudProcessor.voxel_downsample_with_indices(
                pcd, self.configuration["voxel_size_mm"], global_indices
            )
            pcd = pcd_down
            # Create a new global index array, picking e.g. the FIRST original index in each voxel
            new_global_indices = []
            for voxel_global_list in downsample_lists:
                # You could pick the first or average or store them all
                new_global_indices.append(voxel_global_list[0])
            global_indices = np.array(new_global_indices, dtype=int)

        # N-largest clustering
        if self.configuration["use_dbscan"]:
            pcd, global_indices = PointCloudProcessor.cluster_n_largest(
                pcd,
                global_indices,
                n_largest=self.configuration["n_largest_clusters"],
                eps=self.configuration["dbscan_eps"],
                min_points=self.configuration["dbscan_min_points"],
            )

        return pcd, global_indices

    @staticmethod
    def filter_points_by_z(
        pcd: o3d.geometry.PointCloud, z_min: float, z_max: float, global_indices: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Filters pcd by z-range. 'global_indices' references the original cloud’s indices
        (i.e., each row in pcd corresponds to global_indices[i]).
        Returns (filtered_pcd, new_global_indices).
        """
        points = np.asarray(pcd.points)
        z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        # local indices of pcd to keep
        local_keep = np.where(z_mask)[0]
        # build new pcd
        pcd_filtered = pcd.select_by_index(local_keep)
        # update global indices
        new_global_indices = global_indices[local_keep]
        return pcd_filtered, new_global_indices

    @staticmethod
    def voxel_downsample_with_indices(
        pcd: o3d.geometry.PointCloud, voxel_size: float, global_indices: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, List[List[int]]]:
        """
        Custom voxel downsampling that returns a new pcd + list-of-lists, each containing
        the global indices of the points that merged into one downsampled voxel.
        """
        # pylint: disable=too-many-locals
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.colors else None

        min_pt = points.min(axis=0)
        voxel_coords = np.floor((points - min_pt) / voxel_size).astype(np.int64)

        voxel_dict: dict = {}
        for local_i in range(len(points)):
            key = tuple(voxel_coords[local_i])
            voxel_dict.setdefault(key, []).append(local_i)

        downsampled_pts = []
        downsampled_cols = []
        downsampled_idx_lists = []  # each element is a list of global indices

        for _, local_indices_list in voxel_dict.items():
            pts_in_voxel = points[local_indices_list]
            if colors is not None:
                cols_in_voxel = colors[local_indices_list]
                avg_col = cols_in_voxel.mean(axis=0)
            else:
                avg_col = [0.5, 0.5, 0.5]

            avg_pt = pts_in_voxel.mean(axis=0)

            downsampled_pts.append(avg_pt)
            downsampled_cols.append(avg_col)

            # collect the global indices for all points that merged into this voxel
            merged_globals = global_indices[local_indices_list]
            downsampled_idx_lists.append(list(merged_globals))

        pcd_downsampled = o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(np.array(downsampled_pts))
        pcd_downsampled.colors = o3d.utility.Vector3dVector(np.array(downsampled_cols))

        return pcd_downsampled, downsampled_idx_lists

    @staticmethod
    def cluster_n_largest(
        pcd: o3d.geometry.PointCloud,
        global_indices: np.ndarray,
        n_largest: int,
        eps: float,
        min_points: int,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Reruns DBSCAN clustering and keeps the N largest clusters."""
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        if len(unique_labels) == 0:  # Return empty pcd
            return pcd.select_by_index([]), np.array([], dtype=int)

        cluster_sizes = [(lbl, np.sum(labels == lbl)) for lbl in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        chosen_labels = [lbl for lbl, _ in cluster_sizes[:n_largest]]

        valid_mask = np.isin(labels, chosen_labels)
        local_keep = np.where(valid_mask)[0]
        pcd_clustered = pcd.select_by_index(local_keep)
        new_global_indices = global_indices[local_keep]

        return pcd_clustered, new_global_indices


class PoseEstimator:
    """PoseEstimator class that, among other tasks, can map 2D detections onto
    3D points using direct indexing (no camera calibration)."""

    POINTCLOUD_TO_IMAGE_SCALE_FACTOR = 2
    CAMERA_AXIS = np.array([0.0, 0.0, 1.0])

    def __init__(self, configuration: dict) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> dict:
        """Return the configuration dictionary."""
        return self._configuration

    @staticmethod
    def map_points_to_image_coords(
        pcd: o3d.geometry.PointCloud,
        image_shape: Tuple[int, int],
        selected_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes (u, v) image coords for each point in 'pcd' by interpreting
        'selected_indices' as the row-major ordering of the original point cloud.

        If 'selected_indices' is None, we assume no filtering/downsampling was done,
        and so the local index (0..n_points-1) matches the original image's row/col.
        """
        scale_factor = PoseEstimator.POINTCLOUD_TO_IMAGE_SCALE_FACTOR
        n_points = len(pcd.points)
        if selected_indices is None:
            # If no filtering occurred, assume direct indexing from 0..n_points-1
            selected_indices = np.arange(n_points)

        uv_coords = np.zeros((n_points, 2), dtype=np.int32)
        original_width = image_shape[1] // scale_factor
        for i, original_idx in enumerate(selected_indices):
            row = original_idx // original_width
            col = original_idx % original_width
            uv_coords[i, 0] = col * scale_factor
            uv_coords[i, 1] = row * scale_factor

        return uv_coords

    @staticmethod
    def get_points_in_bbox(
        pcd: o3d.geometry.PointCloud, uv_coords: np.ndarray, detection: Detection2D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the subset of 3D points within the 2D bounding box plus their
        centroid."""
        (x, y, w, h) = detection.get_bbox()
        x_max, y_max = x + w, y + h

        points = np.asarray(pcd.points)
        mask = (
            (uv_coords[:, 0] >= x)
            & (uv_coords[:, 0] < x_max)
            & (uv_coords[:, 1] >= y)
            & (uv_coords[:, 1] < y_max)
        )
        points_in_bbox = points[mask]
        if len(points_in_bbox) == 0:
            return points_in_bbox, np.array([0.0, 0.0, 0.0])
        centroid = points_in_bbox.mean(axis=0)
        return points_in_bbox, centroid

    @staticmethod
    def build_detections_3d(
        detections: List[Detection2D],
        points_for_detections: List[Tuple[np.ndarray, np.ndarray]],
        camera_to_robot: Optional[np.ndarray] = None,
    ) -> List[Detection3D]:
        """
        Creates a list of Detection3D objects from 2D detections and
        their (points_in_bbox, centroid). Also estimates a naive 6D pose:
          - translation: the centroid
          - orientation: same as camera (i.e. identity or a known camera rotation)

        If 'camera_to_robot' is provided (4x4), we transform each pose into the
        robot coordinate frame. If no points are found for a detection, the
        resulting Detection3D has None fields for points, centroid, and pose.

        Args:
            detections: 2D detections from which we derived 3D points.
            points_for_detections: list of (points_in_bbox, centroid) per detection.
            camera_to_robot: 4x4 transform from camera frame to robot frame.
                             If provided, the final pose_6d is expressed in the robot frame.

        Returns:
            A list of Detection3D objects. The i-th item corresponds to detections[i].
        """
        output = []
        # TODO: This needs to be fixed!  # pylint: disable=fixme
        #       This currently provide visually correct results, but the orientation seems wrong.
        camera_quat = np.array([-1.0, 0.0, 0.0, 0.0])  # x,y,z,w

        for det, (pts_3d, ctr_3d) in zip(detections, points_for_detections):
            if pts_3d.size == 0:
                output.append(Detection3D(detection_2d=det))
                continue

            # Create a naive 6D pose with centroid and the "camera" orientation
            pose_3d = Pose3D(translation=ctr_3d, quaternion=camera_quat)
            if camera_to_robot is not None:
                pose_3d = pose_3d.transform(camera_to_robot)

            detection_3d = Detection3D(
                detection_2d=det, points_3d=pts_3d, centroid_3d=ctr_3d, pose_3d=pose_3d
            )
            output.append(detection_3d)

        return output

    @staticmethod
    def find_points_for_detections(
        pcd: o3d.geometry.PointCloud,
        frame: Frame,
        selected_indices: Optional[np.ndarray] = None,
        transform_to_robot_frame: bool = False,
    ) -> Optional[List[Detection3D]]:
        """For each 2D detection, collects the corresponding 3D points and
        centroid  via direct indexing and returns them as a list of (points,
        centroid)."""
        if frame.detections is None:
            return None
        # Explicit cast to suppress mypy error
        image_shape = cast(Tuple[int, int], frame.image.shape[:2])
        uv_coords = PoseEstimator.map_points_to_image_coords(pcd, image_shape, selected_indices)
        points_for_detections = [
            PoseEstimator.get_points_in_bbox(pcd, uv_coords, det) for det in frame.detections
        ]
        camera_to_robot = None
        if transform_to_robot_frame and frame.camera_transform is not None:
            camera_to_robot = frame.camera_transform
        detections_3d = PoseEstimator.build_detections_3d(
            frame.detections, points_for_detections, camera_to_robot
        )
        return detections_3d
