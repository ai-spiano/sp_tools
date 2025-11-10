import os
import numpy as np
import cv2
from pyzbar import pyzbar


def decode(filepath, labeldir='./labels/',scale = 2, arc_scale=0.01, qr_scale=0.12,debugdir=''):
    if len(debugdir):
        os.makedirs(debugdir, exist_ok=True)

    src_image = cv2.imread(filepath)
    resize_image = cv2.resize(src_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=121, C=1)
    gray_canny = cv2.Canny(gray, 100, 300, 5, L2gradient=True)

    # 寻找大四边形的轮廓
    contours, hierarchys = cv2.findContours(gray_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangle_contour_ids = []
    for i in range(len(contours)):
        contour = contours[i]
        if cv2.contourArea(contour) / (resize_image.shape[0] * resize_image.shape[1]) < 0.05:
            continue

        epsilon = arc_scale * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and abs(cv2.contourArea(approx) - cv2.contourArea(contour)) / cv2.contourArea(contour) < 0.2:
            # 判断为四边形
            quadrangle_contour_ids.append(i)
            # cv2.drawContours(resize_image, [contour], 0, (0, 255, 0), 2)
            
    # 寻找大四边环形
    parent_contour = None
    child_contour = None
    for i in quadrangle_contour_ids:
        child_id = hierarchys[0][i][2]          # 第一个子轮廓索引
        while child_id != -1:                   # 遍历所有子轮廓
            if child_id!=i and child_id in quadrangle_contour_ids and 0.8 < cv2.contourArea(contours[child_id]) / cv2.contourArea(contours[i]) < 0.99:
                # 该父子轮廓为曲谱轮廓
                parent_contour = contours[i]
                child_contour = contours[child_id]
                # cv2.drawContours(resize_image, [parent_contour], 0, (0, 255, 0), 2)
                # cv2.drawContours(resize_image, [child_contour], 0, (0, 0, 255), 2)
                break
            child_id = hierarchys[0][child_id][0]  # 同级下一个子轮廓

    if parent_contour is None:
        print('未检测到曲谱轮廓')
        return None

    # 获取父轮廓和子轮廓的角点
    parent_pts = cv2.approxPolyDP(parent_contour, arc_scale * cv2.arcLength(parent_contour, True), True).reshape(4,2)
    child_pts = cv2.approxPolyDP(child_contour, arc_scale * cv2.arcLength(child_contour, True), True).reshape(4,2)

    # 将 parent_pts 按“第一个点与第二个点距离最短”的顺序重排
    parent_pts = parent_pts.reshape((4, 2))
    # 计算所有相邻点对距离
    dists = np.linalg.norm(np.roll(parent_pts, -1, axis=0) - parent_pts, axis=1)
    # 找到距离最短的那条边的起点索引
    start_idx = int(np.argmin(dists))
    # 按新顺序排列
    parent_ordered = np.roll(parent_pts, -start_idx, axis=0)

    # child_pts 按照与 parent_ordered 最近对应关系重排
    child_flat = child_pts.reshape(4, 2)
    child_ordered = np.zeros_like(child_flat)
    for i, p_pt in enumerate(parent_ordered):
        dists = np.linalg.norm(child_flat - p_pt, axis=1)
        nearest_idx = np.argmin(dists)
        child_ordered[i] = child_flat[nearest_idx]

    # # 可用可不用
    # # 构造优化目标：让 parent_ordered 到 child_ordered 的4段距离尽量相等
    # def loss(pts_flat):
    #     pts = pts_flat.reshape(4, 2)
    #     dists = np.linalg.norm(pts - child_ordered, axis=1)
    #     return np.std(dists)          # 标准差越小，4段距离越接近

    # res = minimize(loss, parent_ordered.ravel(), method='BFGS')
    # parent_ordered = (parent_ordered + res.x.reshape(4, 2)) / 2

    parent_w = int(np.linalg.norm(parent_ordered[0] - parent_ordered[1]))
    parent_h = int(np.linalg.norm(parent_ordered[1] - parent_ordered[2]))
    

    # # 可视化对应点（调试用）
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    # for i in range(4):
    #     cv2.circle(resize_image, tuple(parent_ordered[i].astype(int)), 1, colors[i], -1)
    #     cv2.circle(resize_image, tuple(child_ordered[i].astype(int)), 1, colors[i], -1)
    # if debug:
    #     cv2.imwrite('temp_results/ordered.png', resize_image)

    # 通过变换后，获取二维码区域，并识别
    rect_pts = np.array([[0, 0], [parent_w, 0], [parent_w, parent_h], [0, parent_h]], dtype=np.float32)
    for i in [0,2]:
        parent_ordered = np.roll(parent_ordered, -i, axis=0)

        M_parent2rect = cv2.getPerspectiveTransform(parent_ordered.astype(np.float32), rect_pts)
        rect_img = cv2.warpPerspective(resize_image, M_parent2rect, (parent_w, parent_h))

        qr_roi = rect_img[:int(parent_h * qr_scale), :int(parent_h * qr_scale)]
        qr_code = pyzbar.decode(qr_roi)
        # datamatrix_code = pylibdmtx.decode(datamatrix)
        
        if len(qr_code)>0:
            break
    
    if len(qr_code)==0:
        return None
    

    # 从二维码中获得标签名
    labelname = qr_code[0].data.decode('utf-8')

    # 读取 label 图并获取其宽高
    label = cv2.imread(os.path.join(labeldir, labelname))
    if label is None:
        print(f'未找到标签图 {labelname}')
        return None

    label[np.all(label != (255, 255, 255), axis=2)] = (0, 0, 0)
    # 将 label 图取反：白变黑，黑变白
    label = cv2.bitwise_not(label)
    label_h, label_w = label.shape[:2]

    # 将 label 的四个角点映射到原图对应 parent_ordered 的位置
    label_pts = np.array([[0, 0], [label_w, 0], [label_w, label_h], [0, label_h]], dtype=np.float32)
    # 缩放回原尺度坐标
    parent_pts = parent_ordered.astype(np.float32) / scale

    # 计算 原图 到标签的透视变换矩阵
    M_parent2label = cv2.getPerspectiveTransform(parent_pts , label_pts)
    warped_image = cv2.warpPerspective(src_image, M_parent2label, (label_w, label_h))
    if len(debugdir):
        cv2.imwrite(os.path.join(debugdir, 'warped_image.png'), warped_image)

    # 计算label到原图的透视变换矩阵
    M_label2parent = cv2.getPerspectiveTransform(label_pts, parent_pts)
    # 将 label 透视变换到原图
    warped_label = cv2.warpPerspective(label, M_label2parent, (src_image.shape[1], src_image.shape[0]))
    if len(debugdir):
        cv2.imwrite(os.path.join(debugdir, 'warped_label.png'), warped_label)

    # 将变换后的 label 叠加到原图（可选：用 alpha 混合或覆盖）
    if len(debugdir):
        alpha = 0.4
        overlay = cv2.addWeighted(src_image, 1 - alpha, warped_label, alpha, 0)
        cv2.imwrite(os.path.join(debugdir, 'overlay.png'), overlay)
    
    return M_parent2label, warped_image, M_label2parent, warped_label


if __name__ == '__main__':
    testdir = './test' # 拍照图片目录
    labeldir = './labels' # 分割标签目录
    outputdir = './outputs' # 输出目录
    debugdir = '' # 调试目录

    os.makedirs(outputdir, exist_ok=True)

    # 解码二维码
    for filename in os.listdir(testdir):
        filepath = os.path.join(testdir, filename)
        result = decode(filepath, labeldir=labeldir, scale = 2, arc_scale=0.01, qr_scale=0.12,debug=debugdir)
        if result is None:
            print(f'转换失败: {filename}')
        else:
            M_parent2label, warped_image, M_label2parent, warped_label = result
            # 保存M_parent2label, warped_image, M_label2parent, warped_label
            np.save(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_img2label.npy'), M_parent2label)
            cv2.imwrite(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_warped_image.png'), warped_image)
            np.save(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_label2img.npy'), M_label2parent)
            cv2.imwrite(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_warped_label.png'), warped_label)

