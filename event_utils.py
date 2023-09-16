import sys
sys.path.append('utils')
import numpy as np
import h5py
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch


def load_events_h5(path):
    file = h5py.File(path, 'r')
    length = len(file['x'])
    events = np.zeros([length, 4], dtype=np.float32)
    events[:, 0] = file['x']
    events[:, 1] = file['y']
    events[:, 2] = file['t']
    events[:, 3] = file['p']
    file.close()
    return events


def eventsToXYTP(events, post_process=False):
    event_x = events[:, 0].astype(np.int32)
    event_y = events[:, 1].astype(np.int32)
    event_pols = events[:, 3].astype(np.int32)

    event_timestamps = events[:, 2]

    if post_process:
        last_stamp = event_timestamps[-1]
        first_stamp = event_timestamps[0]
        deltaT = last_stamp - first_stamp
        event_timestamps = (event_timestamps - first_stamp) / (deltaT + 1e-6)
        # event_timestamps.astype(np.float32)

    # event_pols[event_pols == 0] = -1  # polarity should be +1 / -1

    return event_x, event_y, event_timestamps, event_pols


def eventsToVoxelOld(events, num_bins=5, height=None, width=None, event_polarity=False):
    event_x, event_y, event_timestamps, event_pols = eventsToXYTP(events, post_process=False)

    if height is None or width is None:
        width = event_x.max() + 1
        height = event_y.max() + 1

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # print(event_x.max(), event_x.min(), event_x.mean(), event_y.max(), event_y.min(), event_y.mean())
    # print(event_timestamps.max(), event_timestamps.min(), event_timestamps.mean())
    last_stamp = event_timestamps[-1]
    first_stamp = event_timestamps[0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        print("In eventsToVoxelOld, deltaT == 0:", len(event_timestamps), last_stamp, first_stamp, deltaT)

    event_timestamps = (num_bins - 1) * (event_timestamps - first_stamp) / (deltaT + 1e-6)
    event_pols[event_pols == 0] = -1  # polarity should be +1 / -1

    tis = event_timestamps.astype(np.int)
    dts = event_timestamps - tis
    if event_polarity is False:
        vals_left = event_pols * (1.0 - dts)
        vals_right = event_pols * dts

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, event_x[valid_indices] + event_y[valid_indices] * width
                  + tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, event_x[valid_indices] + event_y[valid_indices] * width
                  + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        return voxel_grid
    else:
        voxel_grid_negative = voxel_grid.copy()

        event_pols_positive = event_pols.copy()
        event_pols_positive[event_pols_positive == -1] = 0
        event_pols_negative = event_pols.copy()
        event_pols_negative[event_pols_negative == 1] = 0

        vals_left_positive = event_pols_positive * (1.0 - dts)
        vals_right_positive = event_pols_positive * dts
        vals_left_negative = event_pols_negative * (1.0 - dts)
        vals_right_negative = event_pols_negative * dts

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, event_x[valid_indices] + event_y[valid_indices] * width
                  + tis[valid_indices] * width * height, vals_left_positive[valid_indices])
        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, event_x[valid_indices] + event_y[valid_indices] * width
                  + (tis[valid_indices] + 1) * width * height, vals_right_positive[valid_indices])
        np.add.at(voxel_grid_negative, event_x[valid_indices] + event_y[valid_indices] * width
                  + tis[valid_indices] * width * height, vals_left_negative[valid_indices])
        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid_negative, event_x[valid_indices] + event_y[valid_indices] * width
                  + (tis[valid_indices] + 1) * width * height, vals_right_negative[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        voxel_grid_negative = np.reshape(voxel_grid_negative, (num_bins, height, width))
        # print(voxel_grid.max(), voxel_grid.min(), voxel_grid_negative.max(), voxel_grid_negative.min())
        return np.concatenate((voxel_grid, voxel_grid_negative), axis=0)


def eventsToVoxel(events, num_bins=5, height=None, width=None, event_polarity=False, temporal_bilinear=True):
    return eventsToVoxelTorch(events, num_bins, height, width, event_polarity, temporal_bilinear).numpy()


def eventsToVoxelTorch(events, num_bins=5, height=None, width=None, event_polarity=False, temporal_bilinear=True):
    xs, ys, ts, ps = eventsToXYTP(events, post_process=True)

    if height is None or width is None:
        width = xs.max() + 1
        height = ys.max() + 1

    if not event_polarity:
        # generate voxel grid which has size num_bins x H x W
        voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(height, width), temporal_bilinear=temporal_bilinear)
    else:
        # generate voxel grid which has size 2*num_bins x H x W
        voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(height, width), temporal_bilinear=temporal_bilinear)
        voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

    return voxel_grid


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search sorted pytorch tensor
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param padding if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size, dtype=torch.float32).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps.float(), accumulate=True)
    return img


def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if isinstance(xs, np.ndarray):
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ts = torch.from_numpy(ts)
        ps = torch.from_numpy(ps)

    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))

    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend) 
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    B : number of bins in output voxel grids (int)
    device : the device that the events are on
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel_pos: voxel of the positive events
    voxel_neg: voxel of the negative events
    """

    if isinstance(xs, np.ndarray):
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ts = torch.from_numpy(ts)
        ps = torch.from_numpy(ps)

    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg


def eventsVoxelToImage(event_voxel, color=False):

    if not isinstance(event_voxel, np.ndarray):
        event_preview = event_voxel.detach().cpu().numpy()
    event_preview = np.sum(event_voxel, axis=0)

    # normalize event image to [0, 255] for display
    # m, M = -10.0, 10.0
    event_preview = np.clip(
        (255.0 * (event_preview - np.min(event_preview)) / (
            np.max(event_preview) - np.min(event_preview + 1e-5))).astype(np.uint8), 0, 255)

    if color:
        event_preview = np.dstack([event_preview] * 3)

    return event_preview


def eventsToGreyimage(events):
    """
    Place events into an image using numpy
    """
    ex, ey, _, ep = eventsToXYTP(events, post_process=True)

    width = ex.max() + 1
    height = ey.max() + 1

    mask = np.where(ex>=width-1, 0, 1)*np.where(ey>=height-1, 0, 1)*np.where(ex<0, 0, 1)*np.where(ey<0, 0, 1)
    coords = np.stack((ey*mask, ex*mask))
    abs_coords = np.ravel_multi_index(coords, [height, width])
    img = np.bincount(abs_coords, minlength=height*width).reshape(height, width).astype(np.float32)

    img = np.clip((10000 * img / (np.max(img) - np.min(img) + 1e-5)).astype(np.uint8), 0, 255)

    return img


def eventsToColorImage(events, background='black'):
    """
    Place events into an image using numpy
    """
    ex, ey, _, ep = eventsToXYTP(events, post_process=True)

    width = ex.max() + 1
    height = ey.max() + 1

    # Blue is positive, red is negative
    if background == 'black':
        dvs_img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        dvs_img = np.ones((height, width, 3), dtype=np.uint8)

    dvs_img[ey[ep==1], ex[ep==1]] = [1, 0, 0]
    dvs_img[ey[ep==-1], ex[ep==-1]] = [0, 0, 1]

    return dvs_img * 255


def eventsToColorImage_old(events, background='black'):
    """
    Place events into an image using numpy
    """
    ex, ey, _, ep = eventsToXYTP(events, post_process=True)

    width = ex.max() + 1
    height = ey.max() + 1

    ex_p = ex[ep == 1]
    ey_p = ey[ep == 1]
    ep_p = ep[ep == 1]
    
    ex_n = ex[ep == -1]
    ey_n = ey[ep == -1]
    ep_n = ep[ep == -1]

    mask_p = np.where(ex_p>width-1, 0, 1)*np.where(ey_p>height-1, 0, 1)*np.where(ex_p<0, 0, 1)*np.where(ey_p<0, 0, 1)
    coords_p = np.stack((ey_p*mask_p, ex_p*mask_p))
    abs_coords_p = np.ravel_multi_index(coords_p, [height, width])
    imgp = np.bincount(abs_coords_p, minlength=height*width).reshape(height, width)
    imgp = (imgp - imgp.min()) / (imgp.max() - imgp.min())

    mask_n = np.where(ex_n>=width-1, 0, 1)*np.where(ey_n>=height-1, 0, 1)*np.where(ex_n<0, 0, 1)*np.where(ey_n<0, 0, 1)
    coords_n = np.stack((ey_n*mask_n, ex_n*mask_n))
    abs_coords_n = np.ravel_multi_index(coords_n, [height, width])
    imgn = np.bincount(abs_coords_n, minlength=height*width).reshape(height, width)
    imgn = (imgn - imgn.min()) / (imgn.max() - imgn.min())

    imgp = np.clip((255 * imgp / (np.max(imgp) - np.min(imgp) + 1e-5)).astype(np.uint8), 0, 255)
    imgn = np.clip((255 * imgn / (np.max(imgn) - np.min(imgn) + 1e-5)).astype(np.uint8), 0, 255)

    mask = np.zeros([height, width], dtype=np.float32)
    if background != 'black':
        mask = np.where(imgn<=0, 1, 0)*np.where(imgp<=0, 1, 0)
        imgn[mask==1] = 1
        imgp[mask==1] = 1

    image_bgr = np.stack(
        [
            imgn,
            np.ones([height, width], dtype=np.float32) * mask,
            imgp
        ], -1
    ) * 255

    return image_bgr


def writeEventsVoxelToViz(filename, event_voxel):
    if not isinstance(event_voxel, np.ndarray):
        event_voxel = event_voxel.detach().cpu().numpy()

    img = eventsVoxelToImage(event_voxel)
    cv2.imwrite(filename, img)


def writeEventsToVoxelViz(filename, events):
    if not isinstance(events, np.ndarray):
        events = events.detach().cpu().numpy()

    img = eventsVoxelToImage(eventsToVoxel(events))
    cv2.imwrite(filename, img)


def writeEventsToGreyViz(filename, events):
    if not isinstance(events, np.ndarray):
        events = events.detach().cpu().numpy()

    img = eventsToGreyimage(events)
    cv2.imwrite(filename, img)


def writeEventsToColorViz(filename, events, center_crop=None):
    if not isinstance(events, np.ndarray):
        events = events.detach().cpu().numpy()

    img = eventsToColorImage(events, background='write')
    if center_crop is not None:
        height, width, _ = img.shape
        crop_height, crop_width = center_crop
        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2
        img = img[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
    cv2.imwrite(filename, img)
