import logging
import os
import cv2
import json
import h5py
import numpy as np
from glob import glob
import torch.utils.data
from utils import load_tiff, load_flow_png, depth2pc, project_pc2image, flow_warp_numpy, get_occu_mask_bidirection
from augmentation import joint_augmentation
from event_utils import eventsToVoxel, load_events_h5


class KubricData(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        if hasattr(cfgs, 'data_seq'):
            seqnames = cfgs.data_seq
        else:
            seqnames = None

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        assert self.split in ['train', 'full', 'val']
        self.cfgs = cfgs

        self.is_event = False
        if hasattr(self.cfgs, 'event_bins'):
            self.event_dir = os.path.join(self.root_dir, 'events_i50_c0.15')
            self.event_bins = cfgs.event_bins
            self.event_polarity = cfgs.event_polarity
            self.is_event = True

        self.is_preprocess = False
        self.preprocess_dir = os.path.join(self.root_dir, 'sf_preprocess')
        if os.path.isdir(self.preprocess_dir):
            self.is_preprocess = True

        self.indices = []

        if self.is_preprocess:
            ls_folder = os.path.join(self.root_dir, 'sf_preprocess')
        else:
            ls_folder = os.path.join(self.root_dir, 'rgba')

        seq_num = len(os.listdir(ls_folder))
        if self.split == 'full':
            self.valid_seq = np.arange(seq_num)
        elif self.split == 'train':
            self.valid_seq = [i for i in range(seq_num) if i % 5 != 0]
        elif self.split == 'val':
            self.valid_seq = [i for i in range(seq_num) if i % 5 == 0]

        if seqnames is None:
            for seq_idx, seqname in enumerate(sorted(os.listdir(ls_folder))):
                if not seq_idx in self.valid_seq:
                    continue
                seq_path = os.path.join(ls_folder, seqname)
                images = sorted([f for f in os.listdir(seq_path)])
                total_length = len(images) if self.is_preprocess else len(images) - 1
                for index in range(total_length):
                    id = images[index].split('.')[0]
                    if '_' in id:
                        id = id.split('_')[0]
                    self.indices.append([seqname, int(id)])
        else:
            logging.info('for {} seqs only'.format(seqnames))
            for seqname in seqnames:
                seq_path = os.path.join(ls_folder, seqname)
                assert os.path.isdir(seq_path)
                images = sorted([f for f in os.listdir(seq_path)])
                for index in range(len(images) - 1):
                    id = images[index].split('.')[0]
                    if '_' in id:
                        id = id.split('_')[0]
                    self.indices.append([seqname, int(id)])

    def __len__(self):
        return len(self.indices)

    def open_hdf5(self, filename, is_event=True):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        h5file = h5py.File(filename, 'r')
        image1 = np.array(h5file["image1"])
        image2 = np.array(h5file["image2"])
        flow_2d = np.array(h5file["flow_2d"])
        flow_2d_mask = np.array(h5file["flow_2d_mask"])
        flow_3d = np.array(h5file["flow_3d"])
        nooccmask_2d = np.array(h5file["nooccmask_2d"])
        nooccmask_3d = np.array(h5file["nooccmask_3d"])
        pc1 = np.array(h5file["pc1"])
        pc2 = np.array(h5file["pc2"])
        metadata = np.array(h5file["metadata"])[0]

        if is_event:
            event_voxel = np.array(h5file["event_voxel"])
            return image1, image2, event_voxel, flow_2d, flow_2d_mask, flow_3d, \
                nooccmask_2d, nooccmask_3d, pc1, pc2, metadata
        else:
            return image1, image2, flow_2d, flow_2d_mask, flow_3d, \
                nooccmask_2d, nooccmask_3d, pc1, pc2, metadata

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        root = self.root_dir
        seq = self.indices[i][0]
        idx1 = self.indices[i][1]
        idx2 = idx1 + 1
        data_dict = {'seq': seq, 'index': idx1}

        preprocess_file = os.path.join(self.preprocess_dir, seq, '{0:05d}_preprocessed.hdf5'.format(idx1))
        if self.is_preprocess and os.path.isfile(preprocess_file):
            if self.is_event:
                image1, image2, event_voxel, flow_2d, flow_2d_mask, flow_3d, \
                    nooccmask_2d, nooccmask_3d, pc1, pc2, metadata = \
                        self.open_hdf5(preprocess_file)
            else:
                image1, image2, flow_2d, flow_2d_mask, flow_3d, \
                    nooccmask_2d, nooccmask_3d, pc1, pc2, metadata = \
                        self.open_hdf5(preprocess_file, is_event=False)
            fx = metadata[0]
            fy = metadata[1]
            cx = metadata[2]
            cy = metadata[3]
            f = fx

            image1_path, image2_path = None, None
            depth1, depth2, depth12 = None, None, None
        else:
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
                flow_2d[:, :, 0] ** 2 + flow_2d[:, :, 1] ** 2) < self.cfgs.max_flow, flow_2d_mask)

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

            if self.is_event:
                height, width = image1.shape[:2]
                event_voxel = load_events_h5(os.path.join(self.event_dir, seq, '{0:05d}_event.hdf5'.format(idx1)))
                event_voxel = eventsToVoxel(event_voxel, num_bins=self.event_bins, height=height, width=width, \
                    event_polarity=bool(self.event_polarity), temporal_bilinear=True)
                event_voxel = event_voxel.transpose(1, 2, 0)
            else:
                event_voxel = None

        # apply depth mask
        mask1 = pc1[..., -1] < self.cfgs.max_depth
        mask1 = pc1[..., -1] < self.cfgs.max_depth
        mask2 = pc2[..., -1] < self.cfgs.max_depth
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]
        nooccmask_3d = nooccmask_3d[mask1]
        mask1 = np.sqrt(flow_3d[:, 0]**2 + flow_3d[:, 1]**2 + flow_3d[:, 2]**2) < self.cfgs.max_3dflow
        pc1, flow_3d = pc1[mask1], flow_3d[mask1]
        nooccmask_3d = nooccmask_3d[mask1]

        # NaN check
        mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(flow_3d, axis=-1)))
        mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]
        nooccmask_3d = nooccmask_3d[mask1]
        # inf check
        mask1 = np.logical_not(np.isinf(np.sum(pc1, axis=-1) + np.sum(flow_3d, axis=-1)))
        mask2 = np.logical_not(np.isinf(np.sum(pc2, axis=-1)))
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]
        nooccmask_3d = nooccmask_3d[mask1]

        # remove out-of-boundary regions of pc2 to create occlusion
        height, width = image1.shape[:2]
        xy2 = project_pc2image(pc2, height, width, f, cx, cy, clip=False)
        boundary_mask = np.logical_and(
            np.logical_and(xy2[..., 0] >= 0, xy2[..., 0] < width),
            np.logical_and(xy2[..., 1] >= 0, xy2[..., 1] < height)
        )
        pc2 = pc2[boundary_mask]

        # data augmentation
        # Note that nooccmask is not need to change if self.cfgs.augmentation is None / eval
        # or not need to use if self.cfgs.augmentation is not None / train
        if self.is_event:
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, event_voxel = joint_augmentation(
                image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation, event_voxel
            )
        else:
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
                image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation,
            )

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]
        nooccmask_3d = nooccmask_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)
        images = np.concatenate([image1, image2], axis=-1)

        if image1_path is not None:
            data_dict['image_names'] = [image1_path, image2_path]
        if depth1 is not None:
            data_dict['depths'] = np.stack([np.array(depth1), np.array(depth2), np.array(depth12)])

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])
        data_dict['occ_mask_2d'] = np.array(nooccmask_2d).astype(np.float32)
        data_dict['occ_mask_3d'] = 1.0 - np.array(nooccmask_3d).astype(np.float32)

        if self.is_event:
            data_dict['event_voxel'] = event_voxel.transpose([2, 0, 1])

        return data_dict

    def get_image1_path(self, i):
        root = self.root_dir
        seq = self.indices[i][0]
        idx1 = self.indices[i][1]
        path = os.path.join(root, 'rgba', seq, '{0:05d}.png'.format(idx1))
        return path

    def get_raw_events(self, i):
        seq = self.indices[i][0]
        idx1 = self.indices[i][1]
        assert self.is_event
        events = load_events_h5(os.path.join(self.event_dir, seq, '{0:05d}_event.hdf5'.format(idx1)))
        return events
