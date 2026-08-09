import math

def calc_conv_output(in_size, kernel_size, stride=1, padding=0, dilation=1):
    out_size = math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    return out_size

# مثال:
h_out = calc_conv_output(in_size=28, kernel_size=3, stride=1, padding=1)
print(f"ابعاد خروجی کانولوشن: {h_out} x {h_out}")  # خروجی: 28 x 28