import typing as tp
import torch
import torch.nn as nn
from torchtyping import TensorType


def _ceil_int_div(n: int, m: int) -> int:
    """Ceil integer division"""
    return (n + m - 1) // m


def _make_mask_a(out_channels: int, in_channels: int, kernel_size: int) -> torch.Tensor:
    kernel_center = _ceil_int_div(kernel_size, 2) - 1

    mask = torch.zeros(
        out_channels, in_channels, kernel_size, kernel_size, dtype=torch.int64
    )

    # for all in and out channels, everything thats above the middle row can be
    # seen
    mask[:, :, :kernel_center] = 1
    # for all in and out channels, for the middle row, only the columns before
    # the middle column can be seen
    mask[:, :, kernel_center, :kernel_center] = 1
    io_ratio = out_channels // in_channels  # number of out channels per in channel

    # for the middle pixels in each channel... magic
    mask[:, :, kernel_center, kernel_center] = torch.repeat_interleave(
        torch.tril(
            torch.ones(in_channels, in_channels, dtype=torch.int64), diagonal=-1
        ),
        repeats=io_ratio,
        dim=0,
    )
    return mask


# Ideas taken from
# https://github.com/suflaj/masked-convolution/blob/a33a9acde4aabc664da765a960dc729378f1d842/masked_convolution/masked_convolution.py#L24
class MaskedConv2d(nn.Conv2d):
    mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        mask_type: tp.Literal["a", "b"] = "a",
    ) -> None:
        # NOTE: for now these argument are enough
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        assert out_channels % in_channels == 0, (
            "output channels must be multiple of the number of input channels, "
            "see the paper, section 3.4, for details"
        )
        if mask_type == "b":
            raise NotImplementedError("ay yo, chill")

        self.register_buffer(
            "mask", _make_mask_a(out_channels, in_channels, kernel_size)
        )
        # NOTE: when creating a nn.Parameter, input data is detached in the
        # nn.Parameter constructor
        self.weight = torch.nn.Parameter(self.mask * self.weight)

        def apply_mask(grad):
            return self.mask * grad

        self.weight.register_hook(apply_mask)


# TODO: fix this wtf
batch = height = width = hidden = channels = None


class ConvLSTMCell(nn.Module):
    in_channels: int
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # TODO: these should be MaskedConv1d
        self.K_ss = nn.Conv1d(
            out_channels, 4 * out_channels, kernel_size, padding="same"
        )
        self.K_is = nn.Conv1d(
            in_channels, 4 * out_channels, kernel_size, padding="same"
        )

    def forward(
        self,
        input: TensorType["batch", "in_channels", "width"],
        hidden_state: tuple[
            TensorType["batch", "out_channels", "width"],
            TensorType["batch", "out_channels", "width"],
        ],
    ) -> tuple[
        TensorType["batch", "out_channels", "width"],
        TensorType["batch", "out_channels", "width"],
    ]:
        # NOTE: variable names are terrible, see paper equations
        prev_h, prev_c = hidden_state
        o, f, i, g = (
            activation(chunk)
            for activation, chunk in zip(
                (*(torch.sigmoid,) * 3, torch.tanh),
                torch.chunk(self.K_ss(prev_h) + self.K_is(input), chunks=4, dim=1),
            )
        )
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c


class ConvLSTM(nn.Module):
    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(
        self,
        input: torch.Tensor,
        hidden_state: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        pass


class RowLSTM(nn.Module):
    # TODO: complete class
    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.conv_mask_a = MaskedConv2d(in_channels, hidden_size, kernel_size=7)
