# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Augmentation pipeline used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
Built around the same concepts that were originally proposed in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import torch.nn.functional as F
import numpy as np
import torch
from ayt.torch_utils import persistence
from ayt.torch_utils import misc

#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

def gaussian_kernel_1d(kernel_size, sigma):
    center = kernel_size // 2
    x = torch.arange(0, kernel_size).float() - center
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_kernel_2d(kernel_size, sigma):
    kernel_1d = gaussian_kernel_1d(kernel_size, sigma)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d

def apply_gaussian_blur(image, kernel):
    image = image[None]
    kernel_size = kernel.shape[0]
    num_channels = image.shape[1]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(num_channels, 1, 1, 1)
    padding = kernel_size // 2
    image = F.pad(image, pad=4*(padding,), mode='reflect')
    blurred_image = F.conv2d(image, kernel, groups=num_channels)
    return blurred_image[0]

#----------------------------------------------------------------------------
# Augmentation pipeline main class.
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.

@persistence.persistent_class
class AugmentPipe:
    def __init__(self, p=1,
        xflip=0, yflip=0, rotate_int=0, translate_int=0, translate_int_max=0.125,
        scale=0, rotate_frac=0, aniso=0, translate_frac=0, scale_std=0.2, rotate_frac_max=1, aniso_std=0.2, aniso_rotate_prob=0.5, translate_frac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        gaussian_blur=0, blur_kernel_size=11, blur_sigma_max=6,
        mixup=0, noise=0, mixup_beta = 1.0, noise_std=1.0,
    ):
        super().__init__()
        self.p                  = float(p)                  # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip              = float(xflip)              # Probability multiplier for x-flip.
        self.yflip              = float(yflip)              # Probability multiplier for y-flip.
        self.rotate_int         = float(rotate_int)         # Probability multiplier for integer rotation.
        self.translate_int      = float(translate_int)      # Probability multiplier for integer translation.
        self.translate_int_max  = float(translate_int_max)  # Range of integer translation, relative to image dimensions.

        # Geometric transformations.
        self.scale              = float(scale)              # Probability multiplier for isotropic scaling.
        self.rotate_frac        = float(rotate_frac)        # Probability multiplier for fractional rotation.
        self.aniso              = float(aniso)              # Probability multiplier for anisotropic scaling.
        self.translate_frac     = float(translate_frac)     # Probability multiplier for fractional translation.
        self.scale_std          = float(scale_std)          # Log2 standard deviation of isotropic scaling.
        self.rotate_frac_max    = float(rotate_frac_max)    # Range of fractional rotation, 1 = full circle.
        self.aniso_std          = float(aniso_std)          # Log2 standard deviation of anisotropic scaling.
        self.aniso_rotate_prob  = float(aniso_rotate_prob)  # Probability of doing anisotropic scaling w.r.t. rotated coordinate frame.
        self.translate_frac_std = float(translate_frac_std) # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness         = float(brightness)         # Probability multiplier for brightness.
        self.contrast           = float(contrast)           # Probability multiplier for contrast.
        self.lumaflip           = float(lumaflip)           # Probability multiplier for luma flip.
        self.hue                = float(hue)                # Probability multiplier for hue rotation.
        self.saturation         = float(saturation)         # Probability multiplier for saturation.
        self.brightness_std     = float(brightness_std)     # Standard deviation of brightness.
        self.contrast_std       = float(contrast_std)       # Log2 standard deviation of contrast.
        self.hue_max            = float(hue_max)            # Range of hue rotation, 1 = full circle.
        self.saturation_std     = float(saturation_std)     # Log2 standard deviation of saturation.

        # Photometric transformations.
        self.gaussian_blur      = float(gaussian_blur)
        self.blur_kernel_size   = float(blur_kernel_size)
        self.blur_sigma_max     = float(blur_sigma_max)

        # Noise perturbations.
        self.noise              = float(noise)              # Probability multiplier for Gaussian noise.
        self.mixup              = float(mixup)              # Probability multiplier for mixup.
        self.noise_std          = float(noise_std)          # Noise standard deviation.
        self.mixup_beta         = float(mixup_beta)         # Mixup beta distribution beta parameter.

    def __call__(self, images, mag=None):
        N, C, H, W = images.shape
        device = images.device
        labels = [torch.zeros([images.shape[0], 0], device=device)]
        mag = torch.ones([N],device=device) if (mag is None) else mag

        # ---------------
        # Pixel blitting.
        # ---------------

        if self.xflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.xflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(3), images)
            labels += [w]

        if self.yflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.yflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(2), images)
            labels += [w]

        if self.rotate_int > 0:
            w = torch.randint(4, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.rotate_int * self.p, w, torch.zeros_like(w))
            images = torch.where((w == 1) | (w == 2), images.flip(3), images)
            images = torch.where((w == 2) | (w == 3), images.flip(2), images)
            images = torch.where((w == 1) | (w == 3), images.transpose(2, 3), images)
            labels += [(w == 1) | (w == 2), (w == 2) | (w == 3)]

        if self.translate_int > 0:
            w = torch.rand([2, N, 1, 1, 1], device=device) * 2 - 1
            w = torch.where(torch.rand([1, N, 1, 1, 1], device=device) < self.translate_int * self.p, w, torch.zeros_like(w))
            tx = w[0].mul(W * self.translate_int_max).round().to(torch.int64)
            ty = w[1].mul(H * self.translate_int_max).round().to(torch.int64)
            b, c, y, x = torch.meshgrid(*(torch.arange(x, device=device) for x in images.shape), indexing='ij')
            x = W - 1 - (W - 1 - (x - tx) % (W * 2 - 2)).abs()
            y = H - 1 - (H - 1 - (y + ty) % (H * 2 - 2)).abs()
            images = images.flatten()[(((b * C) + c) * H + y) * W + x]
            labels += [tx.div(W * self.translate_int_max), ty.div(H * self.translate_int_max)]

        # ------------------------------------------------
        # Select parameters for geometric transformations.
        # ------------------------------------------------

        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        if self.scale > 0:
            w = mag * torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.scale * self.p, w, torch.zeros_like(w))
            s = w.mul(self.scale_std).exp2()
            G_inv = G_inv @ scale2d_inv(s, s)
            labels += [w]

        if self.rotate_frac > 0:
            w = mag * (torch.rand([N], device=device) * 2 - 1) * (np.pi * self.rotate_frac_max)
            w = torch.where(torch.rand([N], device=device) < self.rotate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ rotate2d_inv(-w)
            labels += [w.cos() - 1, w.sin()]

        if self.aniso > 0:
            w = mag * torch.randn([N], device=device)
            r = mag * (torch.rand([N], device=device) * 2 - 1) * np.pi
            w = torch.where(torch.rand([N], device=device) < self.aniso * self.p, w, torch.zeros_like(w))
            r = torch.where(torch.rand([N], device=device) < self.aniso_rotate_prob, r, torch.zeros_like(r))
            s = w.mul(self.aniso_std).exp2()
            G_inv = G_inv @ rotate2d_inv(r) @ scale2d_inv(s, 1 / s) @ rotate2d_inv(-r)
            labels += [w * r.cos(), w * r.sin()]

        if self.translate_frac > 0:
            w = mag.reshape(1,-1) * torch.randn([2, N], device=device)
            w = torch.where(torch.rand([1, N], device=device) < self.translate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ translate2d_inv(w[0].mul(W * self.translate_frac_std), w[1].mul(H * self.translate_frac_std))
            labels += [w[0], w[1]]

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        if G_inv is not I_3:
            cx = (W - 1) / 2
            cy = (H - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz = np.asarray(wavelets['sym6'], dtype=np.float32)
            Hz_pad = len(Hz) // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([W - 1, H - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0,mx1,my0,my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            conv_weight = misc.constant(Hz[None, None, ::-1], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) + 1) // 2
            images = torch.stack([images, torch.zeros_like(images)], dim=4).reshape(N, C, images.shape[2], -1)[:, :, :, :-1]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], padding=[0,conv_pad])
            images = torch.stack([images, torch.zeros_like(images)], dim=3).reshape(N, C, -1, images.shape[3])[:, :, :-1, :]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], padding=[conv_pad,0])
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [N, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
            images = torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            conv_weight = misc.constant(Hz[None, None, :], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) - 1) // 2
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], stride=[1,2], padding=[0,conv_pad])[:, :, :, Hz_pad : -Hz_pad]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], stride=[2,1], padding=[conv_pad,0])[:, :, Hz_pad : -Hz_pad, :]

        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        I_4 = torch.eye(4, device=device)
        M = I_4
        luma_axis = misc.constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)

        if self.brightness > 0:
            w = mag * torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.brightness * self.p, w, torch.zeros_like(w))
            b = w * self.brightness_std
            M = translate3d(b, b, b) @ M
            labels += [w]

        if self.contrast > 0:
            w = mag * torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.contrast * self.p, w, torch.zeros_like(w))
            c = w.mul(self.contrast_std).exp2()
            M = scale3d(c, c, c) @ M
            labels += [w]

        if self.lumaflip > 0:
            w = torch.randint(2, [N, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1], device=device) < self.lumaflip * self.p, w, torch.zeros_like(w))
            M = (I_4 - 2 * luma_axis.ger(luma_axis) * w) @ M
            labels += [w]

        if self.hue > 0:
            w = mag * (torch.rand([N], device=device) * 2 - 1) * (np.pi * self.hue_max)
            w = torch.where(torch.rand([N], device=device) < self.hue * self.p, w, torch.zeros_like(w))
            M = rotate3d(luma_axis, w) @ M
            labels += [w.cos() - 1, w.sin()]

        if self.saturation > 0:
            w = mag.reshape(-1,1,1) * torch.randn([N, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1], device=device) < self.saturation * self.p, w, torch.zeros_like(w))
            M = (luma_axis.ger(luma_axis) + (I_4 - luma_axis.ger(luma_axis)) * w.mul(self.saturation_std).exp2()) @ M
            labels += [w]

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        if M is not I_4:
            images = images.reshape([N, C, H * W])
            if C == 3:
                images = M[:, :3, :3] @ images + M[:, :3, 3:]
            elif C == 1:
                M = M[:, :3, :].mean(dim=1, keepdims=True)
                images = images * M[:, :, :3].sum(dim=2, keepdims=True) + M[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([N, C, H, W])
        
        # ------------------------------
        # Execute photometric transformations.
        # ------------------------------

        if self.gaussian_blur > 0:
            w = mag * torch.rand([N], device=device) * self.blur_sigma_max
            w = torch.where(torch.rand([N], device=device) < self.gaussian_blur * self.p, w, torch.zeros_like(w))
            kernels = torch.cat([gaussian_kernel_2d(self.blur_kernel_size,w[i].item())[None] for i in range(N)],dim=0).to(device)
            images_blur = torch.vmap(apply_gaussian_blur)(images,kernels)
            images[w>0.0] = images_blur[w>0.0]
            labels += [w]
        
        # ------------------------------------------------
        # Select parameters for additive perturbation.
        # ------------------------------------------------

        EPS = torch.zeros([N,C,H,W], device=device)

        if self.mixup > 0:
            w = mag * 0.5 * torch.tensor(np.random.beta(a=1.0,b=self.mixup_beta,size=N), dtype=float, device=device)
            w = torch.where(torch.rand([N], device=device) < self.mixup * self.p, w, torch.zeros_like(w))
            images_perm = images.flip(dims=[0])
            EPS += w.reshape(-1,1,1,1) * (images_perm - images)
            labels += [w]

        if self.noise > 0:
            w = mag * torch.rand([N], device=device) * self.noise_std
            w = torch.where(torch.rand([N], device=device) < self.noise * self.p, w, torch.zeros_like(w))
            shell = torch.randn([N,C,H,W], device=device)
            shell = np.sqrt(C*H*W) * shell / (shell.flatten(start_dim=1).norm(dim=1)).reshape(-1,1,1,1)
            EPS += w.reshape(-1,1,1,1) * shell
            labels += [w]
        
        # ------------------------------
        # Execute additive perturbation.
        # ------------------------------

        images = images + EPS

        # ------------------------------
        # Return augmented image and label.
        # ------------------------------

        labels = torch.cat([x.to(torch.float32).reshape(N, -1) for x in labels], dim=1)
        return images, labels

#----------------------------------------------------------------------------

def get_augmentation(cfg):
    aug = AugmentPipe(
        p = cfg['p'],
        xflip = cfg['xflip'],
        yflip = cfg['yflip'],
        rotate_int = cfg['rotate_int'],
        translate_int = cfg['translate_int'],
        translate_int_max = cfg['translate_int_max'],
        scale = cfg['scale'],
        rotate_frac = cfg['rotate_frac'],
        aniso = cfg['aniso'],
        translate_frac = cfg['translate_frac'],
        scale_std = cfg['scale_std'],
        rotate_frac_max = cfg['rotate_frac_max'],
        aniso_std = cfg['aniso_std'],
        aniso_rotate_prob = cfg['aniso_rotate_prob'],
        translate_frac_std = cfg['translate_frac_std'],
        brightness = cfg['brightness'],
        contrast = cfg['contrast'],
        lumaflip = cfg['lumaflip'],
        hue = cfg['hue'],
        saturation = cfg['saturation'],
        brightness_std = cfg['brightness_std'],
        contrast_std = cfg['contrast_std'],
        hue_max = cfg['hue_max'],
        saturation_std = cfg['saturation_std'],
        gaussian_blur=cfg['gaussian_blur'],
        blur_kernel_size=cfg['blur_kernel_size'],
        blur_sigma_max=cfg['blur_sigma_max'],
        mixup = cfg['mixup'],
        mixup_beta = cfg['mixup_beta'],
        noise = cfg['noise'],
        noise_std = cfg['noise_std'],
    )
    return aug