import os
import cv2
import numpy as np
import h5py
import torch.utils.data
from utils import load_flow_png
from augmentation import joint_augmentation
from event_utils import eventsToVoxel, load_events_h5


class FlyingThings3D(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.cfgs = cfgs

        self.is_preprocess = False
        self.preprocess_dir = os.path.join(self.root_dir, \
            self.split + '_preprocess_ev10_1', 'left')
        if os.path.isdir(self.preprocess_dir):
            self.is_preprocess = True

        self.indices = []
        if self.is_preprocess:
            for filename in os.listdir(self.preprocess_dir):
                self.indices.append(int(filename.split('_')[0]))
        else:
            for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
                self.indices.append(int(filename.split('.')[0]))

    def __len__(self):
        return len(self.indices)

    def open_hdf5(self, filename):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        h5file = h5py.File(filename, 'r')
        image1 = np.array(h5file["image1"])
        image2 = np.array(h5file["image2"])
        flow_2d = np.array(h5file["flow_2d"])
        flow_mask_2d = np.array(h5file["flow_mask_2d"])
        flow_3d = np.array(h5file["flow_3d"])
        occ_mask_3d = np.array(h5file["occ_mask_3d"])
        pc1 = np.array(h5file["pc1"])
        pc2 = np.array(h5file["pc2"])
        return image1, image2, flow_2d, flow_mask_2d, \
            flow_3d, occ_mask_3d, pc1, pc2

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
        preprocess_file = os.path.join(self.preprocess_dir, '%07d_preprocessed.hdf5' % idx1)
        if self.is_preprocess and os.path.isfile(preprocess_file):
            image1, image2, flow_2d, flow_mask_2d, \
                flow_3d, occ_mask_3d, pc1, pc2 = \
                    self.open_hdf5(preprocess_file)
        else:
            pcs = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
            pc1, pc2 = pcs['pc1'], pcs['pc2']

            flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))
            flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))

            occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
            occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))

            image1 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx1))[..., ::-1]
            image2 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx2))[..., ::-1]

        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)

        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation
        )

        # random sampling during training
        if self.split == 'train':
            indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
            indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
            pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], occ_mask_3d[indices1]

        images = np.concatenate([image1, image2], axis=-1)
        pcs = np.concatenate([pc1, pc2], axis=1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['occ_mask_3d'] = occ_mask_3d
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict

    def get_image1_path(self, i):
        idx1 = self.indices[i]
        path = os.path.join(self.split_dir, 'image', '%07d.png' % idx1)
        return path


class FlyingThings3DEvent(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.event_dir = os.path.join(cfgs.root_dir, self.split + '_events_h5', 'left')
        self.event_bins = cfgs.event_bins
        self.event_polarity = cfgs.event_polarity

        self.is_preprocess = False
        self.preprocess_dir = os.path.join(self.root_dir, \
            self.split + '_preprocess_ev{}_{}'.format(self.event_bins, int(self.event_polarity)), 'left')
        if os.path.isdir(self.preprocess_dir):
            self.is_preprocess = True

        self.cfgs = cfgs

        self.indices = []

        if self.is_preprocess:
            for filename in os.listdir(self.preprocess_dir):
                self.indices.append(int(filename.split('_')[0]))
        else:
            for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
                if os.path.isfile(os.path.join(self.event_dir, filename.split('.')[0] + '_event.hdf5')):
                    self.indices.append(int(filename.split('.')[0]))

    def __len__(self):
        return len(self.indices)

    def open_hdf5(self, filename):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        h5file = h5py.File(filename, 'r')
        image1 = np.array(h5file["image1"])
        image2 = np.array(h5file["image2"])
        event_voxel = np.array(h5file["event_voxel"])
        flow_2d = np.array(h5file["flow_2d"])
        flow_mask_2d = np.array(h5file["flow_mask_2d"])
        flow_3d = np.array(h5file["flow_3d"])
        occ_mask_3d = np.array(h5file["occ_mask_3d"])
        pc1 = np.array(h5file["pc1"])
        pc2 = np.array(h5file["pc2"])
        return image1, image2, event_voxel, flow_2d, \
            flow_mask_2d, flow_3d, occ_mask_3d, pc1, pc2

    def open_npz(self, filename):
        assert os.path.isfile(filename), '{} not exist!'.format(filename)
        npzfile = np.load(filename)
        image1 = np.array(npzfile["image1"])
        image2 = np.array(npzfile["image2"])
        event_voxel = np.array(npzfile["event_voxel"])
        flow_2d = np.array(npzfile["flow_2d"])
        flow_mask_2d = np.array(npzfile["flow_mask_2d"])
        flow_3d = np.array(npzfile["flow_3d"])
        occ_mask_3d = np.array(npzfile["occ_mask_3d"])
        pc1 = np.array(npzfile["pc1"])
        pc2 = np.array(npzfile["pc2"])
        return image1, image2, event_voxel, flow_2d, \
            flow_mask_2d, flow_3d, occ_mask_3d, pc1, pc2

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
        preprocess_file = os.path.join(self.preprocess_dir, '%07d_preprocessed.hdf5' % idx1)
        if self.is_preprocess and os.path.isfile(preprocess_file):
            image1, image2, event_voxel, flow_2d, flow_mask_2d, \
                flow_3d, occ_mask_3d, pc1, pc2 = \
                    self.open_hdf5(preprocess_file)
        else:
            pcs = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
            pc1, pc2 = pcs['pc1'], pcs['pc2']

            flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))
            flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))

            occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
            occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))

            image1 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx1))[..., ::-1]
            image2 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx2))[..., ::-1]

            height, width = image1.shape[:2]
            event_voxel = load_events_h5(os.path.join(self.event_dir, '%07d_event.hdf5' % idx1))
            event_voxel = eventsToVoxel(event_voxel, num_bins=self.event_bins, height=height, width=width, \
                event_polarity=self.event_polarity, temporal_bilinear=True)
            event_voxel = event_voxel.transpose(1, 2, 0)

        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)

        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, event_voxel = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation, event=event_voxel
        )

        # random sampling during training
        if self.split == 'train':
            indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
            indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
            pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], occ_mask_3d[indices1]

        images = np.concatenate([image1, image2], axis=-1)
        pcs = np.concatenate([pc1, pc2], axis=1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['event_voxel'] = event_voxel.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['occ_mask_3d'] = occ_mask_3d
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict

    def get_image1_path(self, i):
        idx1 = self.indices[i]
        path = os.path.join(self.split_dir, 'image', '%07d.png' % idx1)
        return path

    def get_raw_events(self, i):

        idx1 = self.indices[i]
        path = os.path.join(self.event_dir, '%07d_event.hdf5' % idx1)
        events = load_events_h5(path)
        return events
