import os
import cv2
import numpy as np
import torch
import json
import h5py
import argparse
from tqdm import tqdm
from utils import load_tiff, load_flow_png, depth2pc, project_pc2image, flow_warp_numpy, get_occu_mask_bidirection
from event_utils import eventsToVoxel, load_events_h5


class KubricData(torch.utils.data.Dataset):
    def __init__(self, root_dir, event_bins, event_polarity, n_points=8192, max_flow=250.):

        self.root_dir = str(root_dir)

        self.event_dir = os.path.join(self.root_dir, 'events_i50_c0.15')
        self.event_bins = event_bins
        self.event_polarity = event_polarity
        self.n_points = n_points
        self.max_flow = max_flow

        self.is_preprocess = False
        self.preprocess_dir = os.path.join(self.root_dir, 'sf_preprocess')
        if not os.path.isdir(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

        self.indices = []

        seq_num = len(os.listdir(os.path.join(self.root_dir, 'rgba')))
        self.valid_seq = np.arange(seq_num)

        for seq_idx, seqname in enumerate(sorted(os.listdir(os.path.join(self.root_dir, 'rgba')))):
            if not seq_idx in self.valid_seq:
                continue
            if seqname in ['staticcamera_8']: # valid check
                continue
            seq_path = os.path.join(self.root_dir, 'rgba', seqname)
            images = sorted([f for f in os.listdir(seq_path)])
            for index in range(len(images) - 1):
                self.indices.append([seqname, int(images[index].split('.')[0])])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        root = self.root_dir
        seq = self.indices[i][0]
        idx1 = self.indices[i][1]
        idx2 = idx1 + 1

        # load camera intrinsics
        metadata_path = os.path.join(root, 'metadata', seq, 'metadata.json')
        metadata = json.load(open(metadata_path, 'r'))
        width = metadata['flags']['resolution'][0]
        height = metadata['flags']['resolution'][1]
        focal_length = metadata['camera']['focal_length']
        sensor_width = metadata['camera']['sensor_width']

        sensor_height = sensor_width / width * height
        fx = focal_length / sensor_width * width
        fy = focal_length / sensor_height * height
        f = fx
        cx = width / 2.
        cy = height / 2.

        # load images
        image1_path = os.path.join(root, 'rgba', seq, '{0:05d}.png'.format(idx1))
        image2_path = os.path.join(root, 'rgba', seq, '{0:05d}.png'.format(idx2))
        assert os.path.isfile(image1_path)
        image1 = cv2.imread(image1_path)[..., ::-1]
        image2 = cv2.imread(image2_path)[..., ::-1]

        # load 2d flow
        flow_forward_path = os.path.join(root, 'forward_flow', seq, '{0:05d}.png'.format(idx1))
        flow_backward_path = os.path.join(root, 'backward_flow', seq, '{0:05d}.png'.format(idx2))

        flow_2d, flow_2d_mask = load_flow_png(flow_forward_path)
        flow_2d_mask = np.logical_and(np.sqrt(
            flow_2d[:, :, 0] ** 2 + flow_2d[:, :, 1] ** 2) < self.max_flow, flow_2d_mask)

        flow_2d_backward, _ = load_flow_png(flow_backward_path)
        flow_2d_nooccmask = get_occu_mask_bidirection(flow_2d, flow_2d_backward) < 0.5

        # load fgmask
        seg1_path = os.path.join(root, 'segmentation', seq, '{0:05d}.png'.format(idx1))
        seg2_path = os.path.join(root, 'segmentation', seq, '{0:05d}.png'.format(idx2))
        fgmask1 = np.sum(cv2.imread(seg1_path), axis=-1) != 0
        fgmask2 = np.sum(cv2.imread(seg2_path), axis=-1) != 0

        # load depth maps
        depth1_path = os.path.join(root, 'depth', seq, '{0:05d}.tiff'.format(idx1))
        depth2_path = os.path.join(root, 'depth', seq, '{0:05d}.tiff'.format(idx2))
        depth1 = load_tiff(depth1_path)
        depth2 = load_tiff(depth2_path)
        depth12 = flow_warp_numpy(depth2[..., None], flow_2d, filling_value=0, interpolate_mode='bilinear')[:, :, 0]
        fgmask12 = flow_warp_numpy(fgmask2[..., None], flow_2d, filling_value=0, interpolate_mode='bilinear')[:, :, 0]

        mask = np.logical_and(depth12 != 0, flow_2d_mask)
        mask = np.logical_and(mask, fgmask1)
        depth12[mask == 0] = 1e6
        depth1[mask == 0] = 1e6

        nooccmask = np.logical_and(mask, fgmask12)
        nooccmask = np.logical_and(nooccmask, flow_2d_nooccmask)

        # lift depth maps into point clouds
        pc1 = depth2pc(depth1, f, cx, cy)[mask]
        pc2 = depth2pc(depth12, f, cx, cy, flow_2d)[mask]
        nooccmask_3d = nooccmask[mask]
        nooccmask_2d = nooccmask
        flow_3d = pc2 - pc1

        height, width = image1.shape[:2]
        event_voxel = load_events_h5(os.path.join(self.event_dir, seq, '{0:05d}_event.hdf5'.format(idx1)))
        event_voxel = eventsToVoxel(event_voxel, num_bins=self.event_bins, height=height, width=width, \
            event_polarity=bool(self.event_polarity), temporal_bilinear=True)
        event_voxel = event_voxel.transpose(1, 2, 0)

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=min(self.n_points, pc1.shape[0]), replace=False)
        indices2 = np.random.choice(pc2.shape[0], size=min(self.n_points, pc2.shape[0]), replace=False)
        pc1, pc2, flow_3d, nooccmask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], nooccmask_3d[indices1]

        filename = os.path.join(self.preprocess_dir, seq, '{0:05d}_preprocessed.hdf5'.format(idx1))
        assert not os.path.isfile(filename)

        self.write_hdf5(filename, image1, image2, event_voxel, flow_2d, flow_2d_mask, flow_3d, \
                        nooccmask_2d, nooccmask_3d, pc1, pc2, metadata)
        return filename

    def write_hdf5(self, filename, image1, image2, event_voxel, flow_2d, flow_2d_mask, flow_3d, \
        nooccmask_2d, nooccmask_3d, pc1, pc2, metadata):

        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        h5file = h5py.File(filename, 'w')
        h5file.create_dataset("image1", data=np.array(image1), compression="gzip")
        h5file.create_dataset("image2", data=np.array(image2), compression="gzip")
        h5file.create_dataset("event_voxel", data=np.array(event_voxel), compression="gzip")
        h5file.create_dataset("flow_2d", data=np.array(flow_2d), compression="gzip")
        h5file.create_dataset("flow_2d_mask", data=np.array(flow_2d_mask), compression="gzip")
        h5file.create_dataset("flow_3d", data=np.array(flow_3d), compression="gzip")
        h5file.create_dataset("nooccmask_2d", data=np.array(nooccmask_2d), compression="gzip")
        h5file.create_dataset("nooccmask_3d", data=np.array(nooccmask_3d), compression="gzip")
        h5file.create_dataset("pc1", data=np.array(pc1), compression="gzip")
        h5file.create_dataset("pc2", data=np.array(pc2), compression="gzip")

        width = metadata['flags']['resolution'][0]
        height = metadata['flags']['resolution'][1]
        focal_length = metadata['camera']['focal_length']
        sensor_width = metadata['camera']['sensor_width']
        sensor_fov = metadata['camera']['field_of_view']

        sensor_height = sensor_width / width * height
        fx = focal_length / sensor_width * width
        fy = focal_length / sensor_height * height
        cx = width / 2.
        cy = height / 2.

        camrea_param = np.array([(fx, fy, cx, cy, sensor_width, sensor_height, sensor_fov), ],
                                 dtype=np.dtype([('fx', np.float32), ('fy', np.float32), ('cx', np.float32), ('cy', np.float32),
                                                ('sensor_width', np.float32), ('sensor_height', np.float32), ('sensor_fov', np.float32)]))

        h5file.create_dataset('metadata', data=camrea_param, compression="gzip")

        h5file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='Path to the flyingthings3d_subset_pc subset')
    parser.add_argument('--event_bins', type=int, default=10)
    parser.add_argument('--event_polarity', type=int, default=1)
    args = parser.parse_args()

    print('Processing')

    preprocessor = KubricData(
        args.input_dir,
        args.event_bins,
        args.event_polarity,
        n_points = 8192 * 2
    )
    preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=4)

    bar = tqdm(preprocessor)
    for filename in bar:
        bar.set_description(filename[0])
