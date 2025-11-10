import os
import numpy as np
import cv2
import qrcode

def make_border_qr(filepath, border_scale=0.009, qr_scale=0.08):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    border_size = int(image.shape[1]*border_scale)
    border_image = make_black_border(image, border_size)

    fmt = os.path.basename(filepath)

    # qr = pylibdmtx.encode(fmt, size='16x48',)
    # qr = np.array(Image.frombytes('RGB', (qr.width, qr.height), qr.pixels))

    # barc = barcode.get_barcode_class('code128')
    # qr = barc(fmt, writer=barcode.writer.ImageWriter())
    # qr = np.array(qr.render())

    qr = np.array(qrcode.make(
        data=fmt,
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=6,
        border=0)
    )[..., None].repeat(3, -1)*255
    
    scale = (image.shape[0] * qr_scale)/qr.shape[0]
    qr = cv2.resize(qr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    print(filepath, image.shape, qr.shape, qr.shape[0]/image.shape[0], qr.shape[1]/image.shape[1], border_size/image.shape[1])
    print(qr.shape)

    border_image[2*border_size:2*border_size+qr.shape[0], 2*border_size:2*border_size+qr.shape[1]] = qr

    return border_image


if __name__ == '__main__':
    imagedir = './images' # 原始图片目录
    labeldir = './labels' # 原始图标签目录
    gendir = './generate' # 生成二维码图片目录
    os.makedirs(gendir, exist_ok=True)

    # 生成二维码
    for filename in os.listdir(imagedir):
        filepath = os.path.join(imagedir, filename)
        border_image = make_border_qr(filepath, border_scale=0.009, qr_scale=0.08)
        cv2.imwrite(os.path.join(gendir, filename), border_image)
    