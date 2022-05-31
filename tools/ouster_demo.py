import argparse
import glob, time
from pathlib import Path

import cv2
from visualization.KittiUtils import BBox2D, BBox3D, KittiObject, KittiCalibration, label_to_class_name, model_output_to_kitti_objects
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiDataset import KittiDataset


try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    # Ouster
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar_ouster.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/kitti/dataset/kitti/training/velodyne',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar_ouster/default/ckpt/checkpoint_epoch_300.pth', help='')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # dataset
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    kitti_dataset = KittiDataset(args.data_path[:-8]) # remove "/velodyne" from the path
    # visualizer
    visualizer = KittiVisualizer()
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    #  model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx in range(len(kitti_dataset)):
            image, pointcloud, labels, calib = kitti_dataset[idx]
            pointcloud = pointcloud.reshape(-1,4)
            logger.info(f'Visualized sample index: \t{idx + 1}')

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # ========================== inference ==========================

            # preprocessing
            data_dict = {
                "points": pointcloud,
                "frame_id": idx
            }
            data_dict = demo_dataset.prepare_data(data_dict)

            start.record()
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            # inference
            pred_dicts, _ = model.forward(data_dict)

            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            print("inference time = ", start.elapsed_time(end))

            # convert to kitti format objects 
            kitti_objects =model_output_to_kitti_objects(pred_dicts)

            # scroe filter
            score_threshold = 0.7
            filtered_objects  = []
            for object in kitti_objects:
                if object.score > score_threshold:
                    filtered_objects.append(object)
            kitti_objects = filtered_objects
            print([x.score for x in kitti_objects])

            # visalization
            visualizer.visualize_scene_bev(pointcloud, kitti_objects, labels, calib)
            if visualizer.user_press == 27:
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()
