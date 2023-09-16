import re
import cv2
import sys
import imageio
import logging
import numpy as np
import torch.utils.data
import torch.distributed as dist
from matplotlib.colors import hsv_to_rgb


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def dist_reduce_sum(value, n_gpus):
    if n_gpus <= 1:
        return value
    tensor = torch.Tensor([value]).cuda()
    dist.all_reduce(tensor)
    return tensor


def copy_to_device(inputs, device, non_blocking=True):
    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: copy_to_device(v, device, non_blocking) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))
    return inputs


def size_of_batch(inputs):
    if isinstance(inputs, list):
        return size_of_batch(inputs[0])
    elif isinstance(inputs, dict):
        return size_of_batch(list(inputs.values())[0])
    elif isinstance(inputs, torch.Tensor):
        return inputs.shape[0]
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))


def load_tiff(filename):
    img = imageio.imread(filename, format="tiff")
    assert img.ndim == 2
    return img


def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data


def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow


def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, -1)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask


def save_flow(filepath, flow):
    assert flow.shape[2] == 2
    magic = np.array(202021.25, dtype=np.float32)
    h = np.array(flow.shape[0], dtype=np.int32)
    w = np.array(flow.shape[1], dtype=np.int32)
    with open(filepath, 'wb') as f:
        f.write(magic.tobytes())
        f.write(w.tobytes())
        f.write(h.tobytes())
        f.write(flow.tobytes())


def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)


def load_disp_png(filepath):
    array = cv2.imread(filepath, -1)
    valid_mask = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid_mask)] = -1.0
    return disp, valid_mask


def save_disp_png(filepath, disp, mask=None):
    if mask is None:
        mask = disp > 0
    disp = np.uint16(disp * 256.0)
    disp[np.logical_not(mask)] = 0
    cv2.imwrite(filepath, disp)


def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat


def zero_padding(inputs, pad_h, pad_w):
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result


def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def depth2pc(depth, f, cx, cy, flow=None):
    h, w = depth.shape

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def project_pc2image(pc, image_h, image_w, f, cx=None, cy=None, clip=True):
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    cx = (image_w - 1) / 2 if cx is None else cx
    cy = (image_h - 1) / 2 if cy is None else cy

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if clip:
        return np.concatenate([
            np.clip(image_x[:, None], a_min=0, a_max=image_w - 1),
            np.clip(image_y[:, None], a_min=0, a_max=image_h - 1),
        ], axis=-1)
    else:
        return np.concatenate([
            image_x[:, None],
            image_y[:, None]
        ], axis=-1)


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    if flow_uv.shape[2] == 2:
        u = flow_uv[:, :, 0]
        v = flow_uv[:, :, 1]
    else:
        u = flow_uv[0]
        v = flow_uv[1]

    assert(u.shape == v.shape)

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def viz_optical_flow(flow, max_flow=512):
    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)

    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)

    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)

    return image_rgb


def viz_depth(depth):
    depth /= 100
    logdepth = np.ones(depth.shape) + \
        (np.log(depth) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    logdepth = np.stack((logdepth, )*3, axis=-1)

    return logdepth


def eventsToColorImage(x, y, pol, W, H, background='white', is_float=False, t=None):
    """
    Place events into an image using numpy
    """
    if not is_float:
        x = x.astype(np.int)
        y = y.astype(np.int)

        pol[pol > 0] = 1
        pol[pol <= 0] = 0

        # Blue is positive, red is negative
        if background == 'black':
            dvs_img = np.zeros((H, Warning, 3), dtype=np.uint8)
        else:
            dvs_img = np.ones((H, W, 3), dtype=np.uint8)

        dvs_img[y[pol == False], x[pol == False]] = [1, 0, 0]
        dvs_img[y[pol == True], x[pol == True]] = [0, 0, 1]

        return dvs_img * 255

    else:
        # https://github.com/uzh-rpg/DSEC/issues/16#issuecomment-855266070

        x = x.squeeze()
        y = y.squeeze()
        pol = pol.squeeze()
        t = t.squeeze()
        assert x.size == y.size == pol.size
        assert H > 0
        assert W > 0
        img_acc = np.zeros((H, W), dtype='float32').ravel()

        pol = pol.astype('int')
        x0 = x.astype('int')
        y0 = y.astype('int')
        t = t.astype('float64')
        value = 2*pol - 1

        t_norm = (t - t.min())/(t.max() - t.min())
        t_norm = t_norm**2
        t_norm = t_norm.astype('float32')
        assert t_norm.min() >= 0
        assert t_norm.max() <= 1

        for xlim in [x0, x0+1]:
            for ylim in [y0, y0+1]:
                mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
                interp_weights = value * \
                    (1 - np.abs(xlim-x)) * (1 - np.abs(ylim-y)) * t_norm

                index = W * ylim.astype('int') + \
                    xlim.astype('int')

                np.add.at(img_acc, index[mask], interp_weights[mask])

        img_acc = np.reshape(img_acc, (H, W))

        img_out = np.full((H, W, 3), fill_value=255, dtype='uint8')

        # Simple thresholding
        #img_out[img_acc > 0] = [0,0,255]
        #img_out[img_acc < 0] = [255,0,0]

        # With weighting (more complicated alternative)
        clip_percentile = 80
        min_percentile = - \
            np.percentile(np.abs(img_acc[img_acc < 0]), clip_percentile)
        max_percentile = np.percentile(
            np.abs(img_acc[img_acc > 0]), clip_percentile)
        img_acc = np.clip(img_acc, min_percentile, max_percentile)

        img_acc_max = img_acc.max()
        idx_pos = img_acc > 0
        img_acc[idx_pos] = img_acc[idx_pos]/img_acc_max
        val_pos = img_acc[idx_pos]
        # img_out[idx_pos] = np.stack((255-val_pos*255, 255-val_pos*255, np.ones_like(val_pos)*255), axis=1)
        img_out[idx_pos] = np.stack((np.zeros_like(val_pos), np.zeros_like(
            val_pos), np.ones_like(val_pos)*255), axis=1)

        img_acc_min = img_acc.min()
        idx_neg = img_acc < 0
        img_acc[idx_neg] = img_acc[idx_neg]/img_acc_min
        val_neg = img_acc[idx_neg]
        img_out[idx_neg] = np.stack((np.ones_like(
            val_neg)*255, np.zeros_like(val_neg), np.zeros_like(val_neg)), axis=1)
        return img_out


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = torch.nn.functional.grid_sample(
        x, v_grid, mode=mode, padding_mode=pad, align_corners=False)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    
    is_numpy = False
    if isinstance(flow12, np.ndarray) and isinstance(flow21, np.ndarray):
        assert flow12.shape[2] == 2
        flow12 = torch.from_numpy(flow12).permute(2, 0, 1).unsqueeze(0)
        flow21 = torch.from_numpy(flow21).permute(2, 0, 1).unsqueeze(0)
        is_numpy = True

    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    if is_numpy:
        return occ.float()[0][0].numpy()
    else:
        return occ.float()


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) *
                        (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) *
                        (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()


def flow_warp_numpy(img, flow, filling_value=0, interpolate_mode='nearest'):
    """Use flow to warp img.
    Args:
        img (ndarray, float or uint8): Image to be warped.
        flow (ndarray, float): Optical Flow.
        filling_value (int): The missing pixels will be set with filling_value.
        interpolate_mode (str): bilinear -> Bilinear Interpolation;
                                nearest -> Nearest Neighbor.
    Returns:
        ndarray: Warped image with the same shape of img
    """
    assert flow.ndim == 3, 'Flow must be in 3D arrays.'
    height = flow.shape[0]
    width = flow.shape[1]
    channels = img.shape[2]

    output = np.ones(
        (height, width, channels), dtype=img.dtype) * filling_value

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow[:, :, 1]
    dy = grid[:, :, 1] + flow[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    if interpolate_mode == 'nearest':
        output[valid, :] = img[dx[valid].round().astype(int),
                               dy[valid].round().astype(int), :]
    elif interpolate_mode == 'bilinear':
        # dirty walkround for integer positions
        eps_ = 1e-6
        dx, dy = dx + eps_, dy + eps_
        left_top_ = img[np.floor(dx[valid]).astype(int),
                        np.floor(dy[valid]).astype(int), :] * (
                            np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                np.ceil(dy[valid]) - dy[valid])[:, None]
        left_down_ = img[np.ceil(dx[valid]).astype(int),
                         np.floor(dy[valid]).astype(int), :] * (
                             dx[valid] - np.floor(dx[valid]))[:, None] * (
                                 np.ceil(dy[valid]) - dy[valid])[:, None]
        right_top_ = img[np.floor(dx[valid]).astype(int),
                         np.ceil(dy[valid]).astype(int), :] * (
                             np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                 dy[valid] - np.floor(dy[valid]))[:, None]
        right_down_ = img[np.ceil(dx[valid]).astype(int),
                          np.ceil(dy[valid]).astype(int), :] * (
                              dx[valid] - np.floor(dx[valid]))[:, None] * (
                                  dy[valid] - np.floor(dy[valid]))[:, None]
        output[valid, :] = left_top_ + left_down_ + right_top_ + right_down_
    else:
        raise NotImplementedError(
            'We only support interpolation modes of nearest and bilinear, '
            f'but got {interpolate_mode}.')
    return output.astype(img.dtype)
