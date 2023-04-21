import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Similar to ResNet https://arxiv.org/abs/1512.03385
    two conv2d+batchnorm2d layers with residual connection.
    replace pooling with stride
    """

    def __init__(self, i_channels, o_channels, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            i_channels,
            o_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=o_channels)
        self.conv2 = nn.Conv2d(
            o_channels, o_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=o_channels)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(i_channels, o_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(o_channels),
            )

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))

        # downsample residual to match conv output
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = F.relu(x)
        return x


class Detector_FPN(nn.Module):
    """ResNet(18) inspired architecture with Feature Pyramid Network
    References:
        FPN for Object Detection - https://arxiv.org/pdf/1612.03144.pdf
    Code References:
        https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
        https://keras.io/examples/vision/retinanet/
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self):
        super(Detector_FPN, self).__init__()

        # output filters at each 'stage'
        filters = [16, 32, 64, 128, 256, 512]

        # Bottom-Up pathway
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.MaxPool2d(2),
        )

        # Stages used to build pyramid
        self.conv_c2 = ConvBlock(filters[0], filters[1], stride=2, padding=1)
        self.conv_c3 = ConvBlock(filters[1], filters[2], stride=2, padding=1)
        self.conv_c4 = ConvBlock(filters[2], filters[3], stride=2, padding=1)
        self.conv_c5 = ConvBlock(filters[3], filters[4], stride=2, padding=1)
        # self.conv_c6 = ConvBlock(filters[4], filters[5], stride=2, padding=1)

        # pyramid channels
        py_chs = 512

        # Top-Down pathway
        self.conv_pyramid_top = nn.Conv2d(filters[4], py_chs, 1)

        # Lateral Connections
        self.conv_c2_lat = nn.Conv2d(filters[1], py_chs, kernel_size=1)
        self.conv_c3_lat = nn.Conv2d(filters[2], py_chs, kernel_size=1)
        self.conv_c4_lat = nn.Conv2d(filters[3], py_chs, kernel_size=1)

        # smooth pyramid levels
        self.conv_p2_smooth = nn.Conv2d(py_chs, py_chs, kernel_size=3, padding=1)
        self.conv_p3_smooth = nn.Conv2d(py_chs, py_chs, kernel_size=3, padding=1)
        self.conv_p4_smooth = nn.Conv2d(py_chs, py_chs, kernel_size=3, padding=1)

        # average pooling to flatten features
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Does image have a star
        self.has_star = nn.Sequential(
            nn.Flatten(),
            nn.Linear(py_chs, filters[4]),
            nn.ReLU(),
            nn.Linear(filters[4], 1),
            nn.Sigmoid()
        )
        # If image has star find bounding box
        self.find_box = nn.Sequential(
            nn.Flatten(),
            nn.Linear(py_chs, py_chs),
            nn.ReLU(),
            nn.Linear(py_chs, 5),
        )

    def forward(self, x):

        # Bottom-Up pathway
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)
        c4 = self.conv_c4(c3)
        c5 = self.conv_c5(c4)
        # c6 = self.conv_c6(c5)

        # Top-Down pathway
        p5 = self.conv_pyramid_top(c5)
        p4 = self._upsample_add(p5, self.conv_c4_lat(c4))
        p3 = self._upsample_add(p4, self.conv_c3_lat(c3))
        p2 = self._upsample_add(p3, self.conv_c2_lat(c2))

        # smoothing the pyramid
        p2 = self.conv_p2_smooth(p2)

        # top of the pyramid has the best semantic features
        # bottom layer has the best global features
        cls_feat = self.avg_pooling(p5)
        reg_feat = self.avg_pooling(p2)

        classify = self.has_star(cls_feat)
        regression = self.find_box(reg_feat)

        p_star = classify.view(x.shape[0], 1)
        bbox = regression.view(x.shape[0], 5)

        for i in range(x.shape[0]):
            if p_star[i].item() < 0.8:
                box = bbox.clone()
                box[i] = torch.from_numpy(np.full(5, np.nan)).to(box)
                bbox = box
        return p_star, bbox

    def _upsample_add(self, p_prev, lc):
        """takes a pyramid layer, upsamples by factor of 2 and adds lateral connections
        Arguments:
            p_prev {tensor} -- coarser feature map
            lc {tensor} -- lateral connection
        Returns:
            finer feature map, lower pyramid layer
        """
        p = F.interpolate(p_prev, size=(lc.shape[-2:]), mode="nearest")
        return p + lc


# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.rand((2, 1, 200, 200))
    net = Detector_FPN()
    out = net(inp)

    # print(out.shape)
    summary(net, input_size=inp.shape[1:])
    # print(net)