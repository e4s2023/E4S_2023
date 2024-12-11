import os

import torch
from torch import conv2d
from torch.autograd import Function
from torch.utils.cpp_extension import load

ACCELERATOR = os.environ.get('ACCELERATOR', 'cuda')

if ACCELERATOR == 'cuda':
    module_path = os.path.dirname(__file__)
    upfirdn2d_op = load(
        'upfirdn2d',
        sources=[
            os.path.join(module_path, 'upfirdn2d.cpp'),
            os.path.join(module_path, 'upfirdn2d_kernel.cu'),
        ],
    )
else:
    from typing import Union, List

    # The following four functions are Apache-2.0 licensed from github.com/open-mmlab/mmcv

    def upfirdn2d(input: torch.Tensor,
                  filter: torch.Tensor,
                  up: int = 1,
                  down: int = 1,
                  padding: Union[int, List[int]] = 0,
                  flip_filter: bool = False,
                  gain: Union[float, int] = 1,
                  use_custom_op: bool = True):
        """Pad, upsample, filter, and downsample a batch of 2D images.

        Performs the following sequence of operations for each channel:

        1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

        2. Pad the image with the specified number of zeros on each side
        (`padding`). Negative padding corresponds to cropping the image.

        3. Convolve the image with the specified 2D FIR filter (`f`),
        shrinking it so that the footprint of all output pixels lies within
        the input image.

        4. Downsample the image by keeping every Nth pixel (`down`).

        This sequence of operations bears close resemblance to
            scipy.signal.upfirdn().

        The fused op is considerably more efficient than performing the same
        calculation using standard PyTorch ops. It supports gradients of arbitrary
        order.

        Args:
            input (torch.Tensor): Float32/float64/float16 input tensor of the shape
                `[batch_size, num_channels, in_height, in_width]`.
            filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
                filter_width]` (non-separable), `[filter_taps]` (separable), or
                `None` (identity).
            up (int): Integer upsampling factor. Can be a single int or a
                list/tuple `[x, y]`. Defaults to 1.
            down (int): Integer downsampling factor. Can be a single int
                or a list/tuple `[x, y]`. Defaults to 1.
            padding (int | tuple[int]): Padding with respect to the upsampled
                image. Can be a single number or a list/tuple `[x, y]` or
                `[x_before, x_after, y_before, y_after]`. Defaults to 0.
            flip_filter (bool): False = convolution, True = correlation.
                Defaults to False.
            gain (int): Overall scaling factor for signal magnitude.
                Defaults to 1.
            use_custom_op (bool): Whether to use customized op.
                Defaults to True.

        Returns:
            Tensor of the shape `[batch_size, num_channels, out_height, out_width]`
        """
        assert isinstance(input, torch.Tensor)
        return _upfirdn2d_ref(
            input,
            filter,
            up=up,
            down=down,
            padding=padding,
            flip_filter=flip_filter,
            gain=gain)


    def _parse_scaling(scaling):
        """Parse scaling into list [x, y]"""
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        assert isinstance(scaling, (list, tuple))
        assert all(isinstance(x, int) for x in scaling)
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    def _parse_padding(padding):
        """Parse padding into list [padx0, padx1, pady0, pady1]"""
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, int) for x in padding)
        if len(padding) == 2:
            padx, pady = padding
            padding = [padx, padx, pady, pady]
        padx0, padx1, pady0, pady1 = padding
        return padx0, padx1, pady0, pady1

    def _upfirdn2d_ref(input: torch.Tensor,
                       filter: torch.Tensor,
                       up: int = 1,
                       down: int = 1,
                       padding: Union[int, List[int]] = 0,
                       flip_filter: bool = False,
                       gain: Union[float, int] = 1):
        """Slow reference implementation of `upfirdn2d()` using standard PyTorch
        ops.

        Args:
            input (torch.Tensor): Float32/float64/float16 input tensor of the shape
                `[batch_size, num_channels, in_height, in_width]`.
            filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
                filter_width]` (non-separable), `[filter_taps]` (separable), or
                `None` (identity).
            up (int): Integer upsampling factor. Can be a single int or a
                list/tuple `[x, y]`. Defaults to 1.
            down (int): Integer downsampling factor. Can be a single int
                or a list/tuple `[x, y]`. Defaults to 1.
            padding (int | tuple[int]): Padding with respect to the upsampled
                image. Can be a single number or a list/tuple `[x, y]` or
                `[x_before, x_after, y_before, y_after]`. Defaults to 0.
            flip_filter (bool): False = convolution, True = correlation.
                Defaults to False.
            gain (int): Overall scaling factor for signal magnitude.
                Defaults to 1.

        Returns:
            torch.Tensor: Tensor of the shape `[batch_size, num_channels,
                out_height, out_width]`.
        """
        # Validate arguments.
        assert isinstance(input, torch.Tensor) and input.ndim == 4
        if filter is None:
            filter = torch.ones([1, 1], dtype=torch.float32, device=input.device)
        assert isinstance(filter, torch.Tensor) and filter.ndim in [1, 2]
        assert filter.dtype == torch.float32 and not filter.requires_grad
        batch_size, num_channels, in_height, in_width = input.shape
        upx, upy = _parse_scaling(up)
        downx, downy = _parse_scaling(down)
        padx0, padx1, pady0, pady1 = _parse_padding(padding)

        # Check that upsampled buffer is not smaller than the filter.
        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= filter.shape[-1] and upH >= filter.shape[0]

        # Upsample by inserting zeros.
        x = input.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
        x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = torch.nn.functional.pad(
            x, [max(padx0, 0),
                max(padx1, 0),
                max(pady0, 0),
                max(pady1, 0)])
        x = x[:, :,
            max(-pady0, 0):x.shape[2] - max(-pady1, 0),
            max(-padx0, 0):x.shape[3] - max(-padx1, 0)]

        # Setup filter.
        filter = filter * (gain ** (filter.ndim / 2))
        filter = filter.to(x.dtype)
        if not flip_filter:
            filter = filter.flip(list(range(filter.ndim)))

        # Convolve with the filter.
        filter = filter[None, None].repeat([num_channels, 1] + [1] * filter.ndim)
        if filter.ndim == 4:
            x = conv2d(input=x, weight=filter, groups=num_channels)
        else:
            x = conv2d(input=x, weight=filter.unsqueeze(2), groups=num_channels)
            x = conv2d(input=x, weight=filter.unsqueeze(3), groups=num_channels)

        # Downsample by throwing away pixels.
        x = x[:, :, ::downy, ::downx]
        return x

class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
            ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out


def upfirdn2d_native(
        input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
          :,
          max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
          :,
          ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, ::down_y, ::down_x, :]
