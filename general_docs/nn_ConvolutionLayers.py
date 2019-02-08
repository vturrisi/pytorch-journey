
# * Conv1d/Conv2d/Conv3d
# Conv1d/Conv2d/Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# applies the convolution operation to a minibatch of 1-d tensors with number of channels in_channels
# the number of filters is equal to the number of channels out_channels

# kernel_size: size of the filter
# stride: how much to slide a filter at each step
# padding: zero padding
# dilatation: spacing between kernel points
# groups: makes filters see only part of the inputs
# for example, if groups == number of filters (out_channels), each filter only sees one in_channel
# bias: use bias


# # Example:
# in_channels = 2
# out_channels = 2
# kernel_size = 2

# conv_filter1 = [[1, 1],
#                 [1, 1]]
# conv_filter2 = [[2, 2],
#                 [2, 2]]

# minibatch of 1 instance
# input layer 1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# input layer 2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#
# (1 * 0 + 1 * 1) + (1 * 10 + 1 * 11) = 22
# (1 * 1 + 1 * 2) + (1 * 11 + 1 * 12) = 26
# ...
# (1 * 8 + 1 * 9) + (1 * 18 + 1 * 19) = 54

# (2 * 0 + 2 * 1) + (2 * 10 + 2 * 11) = 44
# (2 * 1 + 2 * 2) + (2 * 11 + 2 * 12) = 52
# ...
# (2 * 8 + 2 * 9) + (2 * 18 + 2 * 19) = 108

# output layer 1 = [ 22., 26., 30., 34., 38., 42., 46., 50., 54.]
# output layer 2 = [ 44., 52., 60., 68., 76., 84., 92.,100., 108.]


# Transposed convolution
# https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0
