import os
import math
import numpy as np
import cv2

fake_path = 'fake_B'
real_path = 'real_B'

n_channels = 3


# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def evulate():
    fake_paths = get_image_paths(fake_path)
    numbers = len(fake_paths)
    sum_psnr = 0
    max_psnr = 0
    min_psnr = 100
    sum_ssim = 0
    max_ssim = 0
    min_ssim = 1
    for fake in fake_paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_name = os.path.basename(fake)
        real = os.path.join(real_path, img_name)
        # print(img_name)
        # print(hr_path)
        # print(sr_path)
        img_fake = imread_uint(fake, n_channels=n_channels)  # HR image, int8
        img_real = imread_uint(real, n_channels=n_channels)  # HR image, int8
        psnr = calculate_psnr(img_real, img_fake, )
        # print(psnr)
        sum_psnr += psnr
        max_psnr = max(max_psnr, psnr)
        min_psnr = min(min_psnr, psnr)
        ssim = calculate_ssim(img_real, img_fake, )
        # print(ssim)
        sum_ssim += ssim
        max_ssim = max(max_ssim, ssim)
        min_ssim = min(min_ssim, ssim)
    print('Average psnr = ', sum_psnr / numbers)
    print('min_psnr = ', min_psnr)
    print('Max_psnr = ', max_psnr)
    print('Average ssim = ', sum_ssim / numbers)
    print('min_ssim = ', min_ssim)
    print('Max_ssim = ', max_ssim)


if __name__ == '__main__':
    print('-------------------------compute psnr and ssim for evulate sr model---------------------------------')
    evulate()
