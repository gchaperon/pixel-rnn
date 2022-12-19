import typing as tp
import torch
import torch.nn as nn


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
            raise NotImplementedError

        self.register_buffer(
            "mask", _make_mask_a(out_channels, in_channels, kernel_size)
        )
        # NOTE: when creating a nn.Parameter, input data is detached in the
        # nn.Parameter constructor
        self.weight = torch.nn.Parameter(self.mask * self.weight)

        def apply_mask(grad):
            return self.mask * grad

        self.weight.register_hook(apply_mask)


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
