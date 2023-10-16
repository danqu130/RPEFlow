import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
torch.set_num_threads(1)
import torch.utils.data
import logging
import numpy as np
import hdf5plugin
import h5py
import math
import imageio
import skimage
from PIL import Image
from numba import jit
from omegaconf import OmegaConf
from typing import Dict, Tuple

from utils import depth2pc, project_pc2image, flow_warp_numpy
from augmentation import joint_augmentation


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (
        flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (
        flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(
            t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(
            self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(
                self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        # return events, self.ms_to_idx[t_start_ms:t_end_ms] - t_start_ms_idx
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]

    def close(self):
        self.h5f.close()


TRAIN_SEQUENCE = {
    'thun_00_a': True,
    'zurich_city_01_a': False,
    'zurich_city_02_a': False,
    'zurich_city_02_c': True,
    'zurich_city_02_d': True,
    'zurich_city_02_e': True,
    'zurich_city_03_a': True,
    'zurich_city_05_a': True,
    'zurich_city_05_b': False,
    'zurich_city_06_a': True,
    'zurich_city_07_a': True,
    'zurich_city_08_a': True,
    'zurich_city_09_a': False,
    'zurich_city_10_a': True,
    'zurich_city_10_b': True,
    'zurich_city_11_a': False,
    'zurich_city_11_b': True,
    'zurich_city_11_c': True,
}


class DSECTrain(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        super().__init__()
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['train', 'val', 'full']

        self.cfgs = cfgs

        self.root_dir = os.path.join(cfgs.root_dir, 'train')
        self.split = cfgs.split
        self.isbi = cfgs.isbi

        if hasattr(cfgs, 'data_seq'):
            self.data_seqs = cfgs.data_seq
        else:
            self.data_seqs = None

        self.event_bins = cfgs.event_bins
        self.event_polarity = cfgs.event_polarity
        self.is_preprocess = cfgs.use_preprocess
        self.preprocess_root = self.root_dir + '_preprocess_pc'

        self.height = 480
        self.width = 640

        self.left_image1_filenames = []
        self.right_image1_filenames = []
        self.left_image2_filenames = []
        self.right_image2_filenames = []

        self.forward_flow_ts = []
        self.forward_flow_filenames = []
        self.backward_flow_filenames = []
        self.disparity_filenames = []
        self.calibration_filenames = []

        self.event_filenames = []
        self.event_slices = {}
        self.event_rectifys = {}

        self.preprocess_list = []

        self.data_length = 0
        self.fetch_valids()
        if self.is_preprocess is True:
            if len(self.preprocess_list) == 0:
                raise RuntimeError(
                    "please check the preprocess {} has valid format data".format(self.preprocess_root))
        else:
            if self.data_length == 0:
                raise RuntimeError(
                    "please check the root {} has valid format data".format(self.root_dir))

    def get_item_events(self, index, rectifyed=True):
        event_names = self.event_filenames[index]
        start_ts, end_ts = self.forward_flow_ts[index]
        if rectifyed:
            events = self.load_rectifyed_events(event_names, start_ts, end_ts)
        else:
            events = self.load_raw_events(event_names, start_ts, end_ts)
        return events

    def __rmul__(self, v):
        self.h5file_names = v * self.h5file_names
        self.h5file_index = v * self.h5file_index
        return self

    def __len__(self):
        return self.data_length

    def fetch_valids(self):

        if self.data_seqs is None or self.data_seqs == 'full' or self.data_seqs == ['full']:
            base_seqs = sorted([f for f in os.listdir(
                self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))])
            if self.split == 'full':
                logging.info('using DSEC train seqs')
            elif self.split == 'train':
                logging.info('using DSEC train seqs')
                base_seqs = [seq for seq in base_seqs if seq in TRAIN_SEQUENCE.keys()\
                    and TRAIN_SEQUENCE[seq] is True]
            elif self.split == 'val':
                logging.info('using DSEC val seqs')
                base_seqs = [seq for seq in base_seqs if seq in TRAIN_SEQUENCE.keys()\
                    and TRAIN_SEQUENCE[seq] is False]
        else:
            logging.info('using DSEC seqs' + "".join(self.data_seqs))
            base_seqs = [self.data_seqs] if isinstance(
                self.data_seqs, str) else self.data_seqs

        for seq_index, seq in enumerate(base_seqs):
            full_seq = os.path.join(self.root_dir, seq)
            assert os.path.isdir(full_seq)
            assert os.path.isdir(os.path.join(full_seq, 'flow'))

            if self.is_preprocess:
                preprocess_dir = os.path.join(self.preprocess_root, seq)
                if not os.path.exists(preprocess_dir):
                    os.makedirs(preprocess_dir)

            cam_to_cam_yaml = os.path.join(
                full_seq, 'calibration', 'cam_to_cam.yaml')

            forward_flow_folder = os.path.join(full_seq, 'flow', 'forward')
            forward_flow_ts = np.genfromtxt(os.path.join(
                full_seq, 'flow', 'forward_timestamps.txt'), delimiter=',', dtype='int64')
            forward_flow_filenames = sorted([f for f in os.listdir(forward_flow_folder)
                                             if os.path.isfile(os.path.join(forward_flow_folder, f))])

            backward_flow_folder = os.path.join(full_seq, 'flow', 'backward')
            backward_flow_ts = np.genfromtxt(os.path.join(
                full_seq, 'flow', 'backward_timestamps.txt'), delimiter=',', dtype='int64')
            backward_flow_filenames = sorted([f for f in os.listdir(backward_flow_folder)
                                              if os.path.isfile(os.path.join(backward_flow_folder, f))])
            assert len(forward_flow_filenames) == len(backward_flow_filenames)

            event_disparity_folder = os.path.join(
                full_seq, 'disparity', 'event')
            event_disparity_filenames = sorted([f for f in os.listdir(event_disparity_folder)
                                                if os.path.isfile(os.path.join(event_disparity_folder, f)) and f.endswith(".png")])
            event_disparity_filenames = [os.path.join(
                event_disparity_folder, f) for f in event_disparity_filenames]
            image_disparity_folder = os.path.join(
                full_seq, 'disparity', 'image')
            disparity_ts = np.loadtxt(os.path.join(
                full_seq, 'disparity', 'timestamps.txt'), dtype='int64')

            left_image_folder = os.path.join(
                full_seq, 'images', 'left', 'ev_inf')
            left_image_filenames = sorted([f for f in os.listdir(left_image_folder)
                                           if os.path.isfile(os.path.join(left_image_folder, f)) and f.endswith(".png")])
            left_image_filenames = [os.path.join(
                left_image_folder, f) for f in left_image_filenames]
            image_ts = np.loadtxt(os.path.join(
                full_seq, 'images', 'timestamps.txt'), dtype='int64')

            left_event_file_name = os.path.join(
                full_seq, 'events', 'left', 'events.h5')
            left_event_rectify = os.path.join(
                full_seq, 'events', 'left', 'rectify_map.h5')

            seq_length = len(forward_flow_filenames) if not self.isbi \
                else len(forward_flow_filenames) - 1

            for index in range(seq_length):
                forward_flow_file = os.path.join(forward_flow_folder,
                                                 forward_flow_filenames[index])
                ts_single = forward_flow_ts[index]
                if self.isbi:
                    backward_flow_file = os.path.join(backward_flow_folder,
                                                      backward_flow_filenames[index+1])
                    backward_flow_ts_single = backward_flow_ts[index+1]
                    if backward_flow_ts_single[0] != ts_single[1] or \
                            backward_flow_ts_single[1] != ts_single[0]:
                        # print(index, 'delete', ts_single, backward_flow_ts_single, full_seq)
                        continue

                    self.forward_flow_ts.append(ts_single)
                    self.forward_flow_filenames.append(forward_flow_file)
                    self.backward_flow_filenames.append(backward_flow_file)
                else:
                    self.forward_flow_ts.append(ts_single)
                    self.forward_flow_filenames.append(forward_flow_file)

                image1_index = np.searchsorted(
                    image_ts, ts_single[0], side='left')
                image2_index = np.searchsorted(
                    image_ts, ts_single[1], side='left')
                assert image_ts[image1_index] == ts_single[0]
                assert image_ts[image2_index] == ts_single[1]
                image1_file = left_image_filenames[image1_index]
                image2_file = left_image_filenames[image2_index]
                image1_id = image1_file.split('/')[-1][:-4]
                assert os.path.isfile(image1_file)
                assert os.path.isfile(image2_file)
                self.left_image1_filenames.append(image1_file)
                self.left_image2_filenames.append(image2_file)

                disparity1_index = np.searchsorted(
                    disparity_ts, ts_single[0], side='left')
                disparity2_index = np.searchsorted(
                    disparity_ts, ts_single[1], side='left')
                assert disparity_ts[disparity1_index] == ts_single[0]
                assert disparity_ts[disparity2_index] == ts_single[1]
                disparity1_file = event_disparity_filenames[disparity1_index]
                disparity2_file = event_disparity_filenames[disparity2_index]
                assert os.path.isfile(disparity1_file)
                assert os.path.isfile(disparity2_file)

                self.disparity_filenames.append(
                    [disparity1_file, disparity2_file])
                self.event_filenames.append(
                    [seq_index, left_event_file_name, left_event_rectify])
                self.calibration_filenames.append(cam_to_cam_yaml)

                if self.is_preprocess:
                    preprocess_name = os.path.join(preprocess_dir, image1_id + '.hdf5')
                    if not os.path.exists(os.path.dirname(preprocess_name)):
                        os.makedirs(os.path.dirname(preprocess_name))
                    self.preprocess_list.append(preprocess_name)

        self.data_length = len(self.forward_flow_ts)

    @staticmethod
    def load_flow(flowfile: str):
        assert os.path.exists(flowfile)
        assert flowfile.endswith(".png")
        flow_16bit = imageio.imread(flowfile, format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def load_disparity(filepath: str):
        assert os.path.exists(filepath), '{} not exist!'.format(filepath)
        assert filepath.endswith(".png")
        # disp_16bit = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        # return disp_16bit.astype('float32')/256
        disp_16bit = skimage.io.imread(filepath)
        return disp_16bit.astype(np.uint16)/256.0

    @staticmethod
    def load_image(filepath: str):
        assert os.path.exists(filepath)
        return np.array(Image.open(filepath))

    def rectify_events(self, event_data, rectify_map):
        assert rectify_map.shape == (
            self.height, self.width, 2), rectify_map.shape

        x = event_data['x']
        y = event_data['y']

        assert x.max() < self.width
        assert y.max() < self.height
        xy_rect = rectify_map[y, x]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        x_mask = (x_rect >= 0) & (x_rect < self.width)
        y_mask = (y_rect >= 0) & (y_rect < self.height)
        mask_combined = x_mask & y_mask

        return dict(
            x=x_rect[mask_combined],
            y=y_rect[mask_combined],
            p=event_data['p'][mask_combined],
            t=event_data['t'][mask_combined],
        )

    def load_raw_events(self, event_names, start_ts, end_ts):
        seq_index = event_names[0]
        if str(seq_index) not in self.event_slices.keys():
            event_filename = event_names[1]
            rectify_filename = event_names[2]

            event_file = h5py.File(event_filename, 'r')
            with h5py.File(rectify_filename, 'r') as h5_rect:
                events_rectify_map = h5_rect['rectify_map'][()]
            self.event_slices[str(seq_index)] = EventSlicer(event_file)
            self.event_rectifys[str(seq_index)] = events_rectify_map

        return self.event_slices[str(seq_index)].get_events(start_ts, end_ts)

    def load_rectifyed_events(self, event_names, start_ts, end_ts):
        seq_index = event_names[0]
        if str(seq_index) not in self.event_slices.keys():
            event_filename = event_names[1]
            rectify_filename = event_names[2]

            event_file = h5py.File(event_filename, 'r')
            with h5py.File(rectify_filename, 'r') as h5_rect:
                events_rectify_map = h5_rect['rectify_map'][()]
            self.event_slices[str(seq_index)] = EventSlicer(event_file)
            self.event_rectifys[str(seq_index)] = events_rectify_map

        raw_events = self.event_slices[str(
            seq_index)].get_events(start_ts, end_ts)
        return self.rectify_events(raw_events, self.event_rectifys[str(seq_index)])

    def load_data_by_index(self, index):
        start_ts, end_ts = self.forward_flow_ts[index]

        im1_filename = self.left_image1_filenames[index]
        im2_filename = self.left_image2_filenames[index]
        im1 = self.load_image(im1_filename)
        im2 = self.load_image(im2_filename)

        disparity1_filename = self.disparity_filenames[index][0]
        disparity2_filename = self.disparity_filenames[index][1]
        disp1 = self.load_disparity(disparity1_filename)
        disp2 = self.load_disparity(disparity2_filename)

        event_names = self.event_filenames[index]
        events = self.load_rectifyed_events(event_names, start_ts, end_ts)

        flow12_filename = self.forward_flow_filenames[index]
        flow12, flow12_valid = self.load_flow(flow12_filename)

        calibration_filename = self.calibration_filenames[index]
        calib_conf = OmegaConf.load(calibration_filename)
        intrinsics = np.array(calib_conf['intrinsics']['camRect0']['camera_matrix'])
        # perspective transformation matrices
        perspectives = np.array(calib_conf['disparity_to_depth']['cams_03'])

        return im1, im2, events, flow12, flow12_valid, disp1, disp2, \
            intrinsics, perspectives

    def eventsToVoxelInterTorch(self, xs, ys, ts, ps, num_bins, height, width):
        input_size = [num_bins, height, width]
        voxel_grid = torch.zeros(
            (input_size), dtype=torch.float32, requires_grad=False)
        C, H, W = voxel_grid.shape

        t_norm = ts
        t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

        x0 = xs.int()
        y0 = ys.int()
        t0 = t_norm.int()

        value = 2*ps-1

        for xlim in [x0, x0+1]:
            for ylim in [y0, y0+1]:
                for tlim in [t0, t0+1]:

                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                        ylim >= 0) & (tlim >= 0) & (tlim < C)
                    interp_weights = value * \
                        (1 - (xlim-xs).abs()) * (1 - (ylim-ys).abs()) * \
                        (1 - (tlim - t_norm).abs())

                    index = H * W * tlim.long() + \
                        W * ylim.long() + \
                        xlim.long()

                    voxel_grid.put_(
                        index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid

    def eventsToVoxelInter(self, events, num_bins, height, width, event_polarity=False):

        xs = events['x']
        ys = events['y']
        ts = events['t']
        ps = events['p']

        ts = (ts - ts[0]).astype('float32')
        ts = (ts/ts[-1])
        xs = xs.astype('float32')
        ys = ys.astype('float32')
        ps = ps.astype('float32')

        if isinstance(events['x'], np.ndarray):
            xs = torch.from_numpy(xs)
            ys = torch.from_numpy(ys)
            ts = torch.from_numpy(ts)
            ps = torch.from_numpy(ps)

        if not event_polarity:
            # generate voxel grid which has size num_bins x H x W
            voxel_grid = self.eventsToVoxelInterTorch(
                xs, ys, ts, ps, num_bins, height, width)
        else:
            # generate voxel grid which has size 2*num_bins x H x W
            pos_weights = ps[ps > 0]
            neg_weights = ps[ps <= 0]
            neg_weights = 1
            voxel_pos = self.eventsToVoxelInterTorch(
                xs[ps > 0], ys[ps > 0], ts[ps > 0], pos_weights, num_bins, height, width)
            voxel_neg = self.eventsToVoxelInterTorch(
                xs[ps <= 0], ys[ps <= 0], ts[ps <= 0], neg_weights, num_bins, height, width)
            voxel_grid = torch.cat([voxel_pos, voxel_neg], 0)

        return voxel_grid.numpy()

    def open_hdf5(self, filename):

        h5file = h5py.File(filename, 'r')

        events = {}
        events['x'] = np.array(h5file["events_x"])
        events['y'] = np.array(h5file["events_y"])
        events['t'] = np.array(h5file["events_t"])
        events['p'] = np.array(h5file["events_p"])

        event_voxel = np.array(h5file["event_voxel"])

        image1 = np.array(h5file["image1"])
        image2 = np.array(h5file["image2"])
        flow12 = np.array(h5file["flow12"])
        flow12_valid = np.array(h5file["flow12_valid"])

        disp1 = None
        disp2 = None
        if 'disp1' in h5file.keys():
            disp1 = np.array(h5file["disp1"])
        if 'disp2' in h5file.keys():
            disp2 = np.array(h5file["disp2"])

        intrinsics = None
        perspectives = None
        if 'intrinsics' in h5file.keys():
            intrinsics = np.array(h5file["intrinsics"])
        if 'perspectives' in h5file.keys():
            perspectives = np.array(h5file["perspectives"])

        return image1, image2, events, event_voxel, flow12, flow12_valid, \
                disp1, disp2, intrinsics, perspectives

    def write_hdf5(self, filename, image1, image2, events, event_voxel, \
        flow12, flow12_valid, disp1, disp2, intrinsics, perspectives):

        h5file = h5py.File(filename, 'w')
        h5file.create_dataset("events_x", data=np.array(
            events['x']), compression="gzip")
        h5file.create_dataset("events_y", data=np.array(
            events['y']), compression="gzip")
        h5file.create_dataset("events_p", data=np.array(
            events['p']), compression="gzip")
        h5file.create_dataset("events_t", data=np.array(
            events['t']), compression="gzip")

        h5file.create_dataset("event_voxel", data=np.array(
            event_voxel), compression="gzip")

        h5file.create_dataset("image1", data=np.array(
            image1), compression="gzip")
        h5file.create_dataset("image2", data=np.array(
            image2), compression="gzip")
        h5file.create_dataset("flow12", data=np.array(
            flow12), compression="gzip")
        h5file.create_dataset("flow12_valid", data=np.array(
            flow12_valid), compression="gzip")
        h5file.create_dataset("disp1", data=np.array(
            disp1), compression="gzip")
        h5file.create_dataset("disp2", data=np.array(
            disp2), compression="gzip")
        h5file.create_dataset("intrinsics", data=np.array(
            intrinsics), compression="gzip")
        h5file.create_dataset("perspectives", data=np.array(
            perspectives), compression="gzip")

        h5file.close()

    def append_hdf5(self, filename, dsname, data):
        h5file = h5py.File(filename, 'a')
        h5file.create_dataset(dsname, data=np.array(
            data), compression="gzip")
        h5file.close()

    def __getitem__(self, index):

        if not self.cfgs.augmentation.enabled:
            np.random.seed(23333)

        if self.is_preprocess:
            baseid = self.preprocess_list[index].split('/')
            baseid = baseid[-1].split('.')[0]
            seq_name = self.preprocess_list[index].split('/')[-2]
        else:
            baseid = self.left_image1_filenames[index].split('/')
            baseid = baseid[-1].split('.')[0]
            seq_name = self.left_image1_filenames[index].split('/')[-5]

        data_dict = {}
        data_dict['index'] = index
        data_dict['baseid'] = baseid
        data_dict['seq_name'] = seq_name

        # start_ts, end_ts = self.forward_flow_ts[index]
        if self.is_preprocess and os.path.isfile(self.preprocess_list[index]):
            image1, image2, events, event_voxel, flow_2d, flow_2d_mask, \
                disp1, disp2, intrinsics, perspectives = \
                    self.open_hdf5(self.preprocess_list[index])
            image_h, image_w = image1.shape[:2]

        else:
            image1, image2, events, flow_2d, flow_2d_mask, \
                disp1, disp2, intrinsics, perspectives = \
                    self.load_data_by_index(index)
            image_h, image_w = image1.shape[:2]

            event_voxel = self.eventsToVoxelInter(events, num_bins=self.event_bins, \
                height=image_h, width=image_w, event_polarity=self.event_polarity)

            if self.is_preprocess:
                self.write_hdf5(self.preprocess_list[index], image1, image2, events, event_voxel, \
                    flow_2d, flow_2d_mask, disp1, disp2, intrinsics, perspectives)

        data_dict['input_h'] = image_h
        data_dict['input_w'] = image_w

        # K_r0 = np.eye(3)
        # K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = intrinsics
        f = intrinsics[0]
        cx = intrinsics[2]
        cy = intrinsics[3]
        baseline = 1.0 / perspectives[3][2]

        depth1 = baseline * f / (disp1 + 1e-6)
        depth2 = baseline * f / (disp2 + 1e-6)
        mask1 = np.logical_and(np.logical_and(disp1 != np.inf, depth1 < self.cfgs.max_depth), disp1 != 0)
        mask2 = np.logical_and(np.logical_and(disp2 != np.inf, depth2 < self.cfgs.max_depth), disp2 != 0)

        depth12 = flow_warp_numpy(depth2[..., None], flow_2d, filling_value=0, interpolate_mode='bilinear')[:, :, 0]
        mask12 = np.logical_and(np.logical_and(depth12 != np.inf, depth12 < self.cfgs.max_depth), depth12 != 0)

        depth1[mask1 == 0] = 1e6
        depth2[mask2 == 0] = 1e6
        depth12[mask12 == 0] = 1e6

        mask = np.logical_and(np.logical_and(mask1, mask12), flow_2d_mask)

        pc1 = depth2pc(depth1, f=f, cx=cx, cy=cy)[mask]
        pc2 = depth2pc(depth12, f=f, cx=cx, cy=cy, flow=flow_2d)[mask]
        flow_3d = pc2 - pc1

        mask1 = np.sqrt(flow_3d[:, 0]**2 + flow_3d[:, 1]**2 + flow_3d[:, 2]**2) < self.cfgs.max_3dflow
        pc1, flow_3d = pc1[mask1], flow_3d[mask1]

        flow_3d_mask = np.ones(flow_3d.shape[0], dtype=np.float32)

        # remove out-of-boundary regions of pc2 to create occlusion
        xy2 = project_pc2image(pc2, image_h, image_w, f, cx, cy, clip=False)
        boundary_mask = np.logical_and(
            np.logical_and(xy2[..., 0] >= 0, xy2[..., 0] < image_w),
            np.logical_and(xy2[..., 1] >= 0, xy2[..., 1] < image_h)
        )
        pc2 = pc2[boundary_mask]

        flow_2d = np.concatenate([flow_2d.astype(np.float32), flow_2d_mask[..., None].astype(np.float32)], axis=-1)
        flow_3d = np.concatenate([flow_3d.astype(np.float32), flow_3d_mask[..., None].astype(np.float32)], axis=-1)

        # data augmentation
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, event_voxel = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation, event_voxel
        )

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1).astype(np.float32) 
        images = np.concatenate([image1, image2], axis=-1).astype(np.float32)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['depths'] = np.stack([np.array(depth1), np.array(depth2), np.array(depth12)])
        data_dict['event_voxel'] = event_voxel
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])
        data_dict['occ_mask_2d'] = np.array(mask).astype(np.float32)

        return data_dict

    def get_image1_path(self, i):
        if self.is_preprocess:
            path = self.preprocess_list[i]
        else:
            path = self.left_image1_filenames[i]
        return path

    def get_raw_events(self, i):
        return self.get_item_events(i)


class DSECPreprocessTrain(DSECTrain):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.is_preprocess = True

    def fetch_valids(self):

        if self.data_seqs is None or self.data_seqs == 'full' or self.data_seqs == ['full']:
            base_seqs = sorted([f for f in os.listdir(self.preprocess_root) \
                if os.path.isdir(os.path.join(self.preprocess_root, f))])
            if self.split == 'full':
                logging.info('using DSEC train seqs')
            elif self.split == 'train':
                logging.info('using DSEC train seqs')
                base_seqs = [seq for seq in base_seqs if seq in TRAIN_SEQUENCE.keys()\
                    and TRAIN_SEQUENCE[seq] is True]
            elif self.split == 'val':
                logging.info('using DSEC val seqs')
                base_seqs = [seq for seq in base_seqs if seq in TRAIN_SEQUENCE.keys()\
                    and TRAIN_SEQUENCE[seq] is False]
        else:
            logging.info('using DSEC seqs' + "".join(self.data_seqs))
            base_seqs = [self.data_seqs] if isinstance(
                self.data_seqs, str) else self.data_seqs

        for seq in base_seqs:
            full_seq = os.path.join(self.preprocess_root, seq)
            assert os.path.isdir(full_seq)

            preprocess_dir = os.path.join(self.preprocess_root, seq)
            preprocess_names = sorted([f for f in os.listdir(preprocess_dir) \
                if os.path.isfile(os.path.join(preprocess_dir, f)) and f.endswith(".hdf5")])
            preprocess_names = [os.path.join(
                preprocess_dir, f) for f in preprocess_names]

            for preprocess_name in preprocess_names:
                self.preprocess_list.append(preprocess_name)

        self.data_length = len(self.preprocess_list)

    def get_raw_events(self, i):
        if len(self.event_filenames) == 0:
            super().fetch_valids()
        return self.get_item_events(i)
