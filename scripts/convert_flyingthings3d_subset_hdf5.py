import os
import cv2
import numpy as np
import torch
import h5py
import argparse
from utils import load_flow_png
from event_utils import eventsToVoxel, load_events_h5
from tqdm import tqdm


class FlyingThings3DEvent(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, event_bins, event_polarity):
        assert os.path.isdir(root_dir)

        self.root_dir = str(root_dir)
        self.split = str(split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.event_dir = os.path.join(self.root_dir, self.split + '_events_h5', 'left')
        self.event_bins = event_bins
        self.event_polarity = event_polarity

        self.preprocess_dir = os.path.join(self.root_dir, \
            self.split + '_preprocess_ev{}_{}'.format(event_bins, int(event_polarity)), 'left')
        if not os.path.isdir(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

        self.indices = []
        for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
            if os.path.isfile(os.path.join(self.event_dir, filename.split('.')[0] + '_event.hdf5')):
                self.indices.append(int(filename.split('.')[0]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx1 = self.indices[i]
        idx2 = idx1 + 1

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
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
            event_polarity=bool(self.event_polarity), temporal_bilinear=True)
        event_voxel = event_voxel.transpose(1, 2, 0)

        filename = os.path.join(self.preprocess_dir, '%07d_preprocessed.hdf5' % idx1)
        self.write_hdf5(filename, image1, image2, event_voxel, flow_2d, \
            flow_mask_2d, flow_3d, occ_mask_3d, pc1, pc2)

        return filename

    def write_hdf5(self, filename, image1, image2, event_voxel, flow_2d, \
        flow_mask_2d, flow_3d, occ_mask_3d, pc1, pc2):

        h5file = h5py.File(filename, 'w')
        h5file.create_dataset("image1", data=np.array(image1), compression="gzip")
        h5file.create_dataset("image2", data=np.array(image2), compression="gzip")
        h5file.create_dataset("event_voxel", data=np.array(event_voxel), compression="gzip")
        h5file.create_dataset("flow_2d", data=np.array(flow_2d), compression="gzip")
        h5file.create_dataset("flow_mask_2d", data=np.array(flow_mask_2d), compression="gzip")
        h5file.create_dataset("flow_3d", data=np.array(flow_3d), compression="gzip")
        h5file.create_dataset("occ_mask_3d", data=np.array(occ_mask_3d), compression="gzip")
        h5file.create_dataset("pc1", data=np.array(pc1), compression="gzip")
        h5file.create_dataset("pc2", data=np.array(pc2), compression="gzip")
        h5file.close()

    def write_npz(self, filename, image1, image2, event_voxel, flow_2d, \
        flow_mask_2d, flow_3d, occ_mask_3d, pc1, pc2):
        np.savez_compressed(filename, image1=image1, image2=image2, event_voxel=event_voxel, \
            flow_2d=flow_2d, flow_mask_2d=flow_mask_2d, flow_3d=flow_3d, occ_mask_3d=occ_mask_3d, \
                pc1=pc1, pc2=pc2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='Path to the flyingthings3d_subset_pc subset')
    parser.add_argument('--event_bins', type=int, default=10)
    parser.add_argument('--event_polarity', type=int, default=1)
    args = parser.parse_args()

    for split_idx, split in enumerate(['train', 'val']):
        if not os.path.exists(os.path.join(args.input_dir, split)):
            continue

        print('Processing "%s" split...' % split)

        preprocessor = FlyingThings3DEvent(
            args.input_dir,
            split,
            args.event_bins,
            args.event_polarity
        )
        preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=8)

        bar = tqdm(preprocessor)
        for filename in bar:
            bar.set_description(filename[0])
