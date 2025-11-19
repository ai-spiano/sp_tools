import os
import numpy as np
import cv2
import qrcode


def make_border_qr(filepath, border_scale=0.009, qr_scale=0.08):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    pad = int(image.shape[0] * qr_scale)
    border_size = int(image.shape[1]*border_scale)

    pad_image = cv2.copyMakeBorder(image, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    border_image = cv2.copyMakeBorder(pad_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    fmt = f'{os.path.basename(filepath)}\n{pad}\n{border_size}'
    qr = np.array(qrcode.make(
        data=fmt,
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=6,
        border=0)
    )[..., None].repeat(3, -1)*255
    
    scale = float(pad) / qr.shape[0]
    qr = cv2.resize(qr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    border_image[2*border_size:2*border_size+qr.shape[0], 2*border_size:2*border_size+qr.shape[1]] = qr

    return border_image


if __name__ == '__main__':
    imagedir = '/home/zjx/work/gitwork/sp_tools/data/xml_data/images' # 原始图片目录
    gendir = './generate' # 生成二维码图片目录
    os.makedirs(gendir, exist_ok=True)

    # 生成二维码
    for filename in os.listdir(imagedir):
        filepath = os.path.join(imagedir, filename)
        border_image = make_border_qr(filepath, border_scale=0.009, qr_scale=0.08)
        cv2.imwrite(os.path.join(gendir, filename), border_image)
    