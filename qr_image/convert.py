import os
import numpy as np
import cv2
from pyzbar import pyzbar


def decode(filepath, labeldir='./labels/',scale = 2, arc_scale=0.01, qr_scale=0.12,debugdir=''):
    if len(debugdir):
        os.makedirs(debugdir, exist_ok=True)

    obj = cv2.wechat_qrcode_WeChatQRCode()
    src_image = cv2.imread(filepath)
    resize_image = cv2.resize(src_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 将图像转到 HSV 空间，通过阈值提取黑色区域
    hsv = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HSV)
    # 黑色在 HSV 中 H 任意，S 低，V 低；这里给出一个宽松的黑色区间
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 可选：对黑色掩膜做形态学处理，去小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=1) 

    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    gray = (gray * (1- black_mask / 255)).astype(np.uint8)
    # gray = gray[..., None].repeat(3, -1)
    cv2.imwrite(os.path.join(debugdir, 'gray.png'), gray)


    # gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=121, C=1)
    cv2.imwrite(os.path.join(debugdir, 'gray.png'), gray)

    # 对二值图做开运算，先腐蚀再膨胀，去除小噪点并保持整体形状
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    
    cv2.imwrite(os.path.join(debugdir, 'gray.png'), gray)


    # if len(debugdir):
    #     cv2.imwrite(os.path.join(debugdir, 'gray.png'), gray)
    gray_canny = cv2.Canny(gray, 100, 300, 5, L2gradient=True)
    # 对二值图做膨胀运算，填补小孔洞，连接邻近区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gray_canny = cv2.dilate(gray_canny, kernel, iterations=1)
    gray_canny = cv2.morphologyEx(gray_canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    

    if len(debugdir):
        cv2.imwrite(os.path.join(debugdir, 'gray_canny.png'), gray_canny)

    # 寻找大四边形的轮廓
    contours, hierarchys = cv2.findContours(gray_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quadrangle_contour_ids = []
    for i in range(len(contours)):
        contour = contours[i]

        epsilon = arc_scale * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if cv2.contourArea(approx) / (resize_image.shape[0] * resize_image.shape[1]) < 0.5:
            continue

        
        if len(approx) == 4 and cv2.isContourConvex(approx) and abs(cv2.contourArea(approx) - cv2.contourArea(contour)) / cv2.contourArea(contour) < 0.2:
            # 判断为四边形
            quadrangle_contour_ids.append(i)
            if len(debugdir):
                cv2.drawContours(resize_image, [contour], 0, (0, 255, 0), 2)
            
    # if len(debugdir):
    #     cv2.imwrite(os.path.join(debugdir, 'quadrangle.png'), resize_image)
    
    # 寻找大四边环形
    parent_contour = None
    child_contour = None
    for i in quadrangle_contour_ids:
        child_id = hierarchys[0][i][2]          # 第一个子轮廓索引
        while child_id != -1:                   # 遍历所有子轮廓
            if child_id!=i and child_id in quadrangle_contour_ids and 0.95 < cv2.contourArea(contours[child_id]) / cv2.contourArea(contours[i]) < 0.99:
                # 该父子轮廓为曲谱轮廓
                parent_contour = contours[i]
                child_contour = contours[child_id]
                # cv2.drawContours(resize_image, [parent_contour], 0, (0, 255, 0), 2)
                # cv2.drawContours(resize_image, [child_contour], 0, (0, 0, 255), 2)
                break
            child_id = hierarchys[0][child_id][0]  # 同级下一个子轮廓

    if parent_contour is None:
        print(f'未检测到曲谱轮廓: {filepath}')
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
    # if len(debugdir):
    #     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    #     for i in range(4):
    #         cv2.circle(resize_image, tuple(parent_ordered[i].astype(int)), 1, colors[i], -1)
    #         cv2.circle(resize_image, tuple(child_ordered[i].astype(int)), 1, colors[i], -1)
    
    #     cv2.imwrite(os.path.join(debugdir, 'ordered.png'), resize_image)

    # 通过变换后，获取二维码区域，并识别
    qr_rois = []
    rect_pts = np.array([[0, 0], [parent_w, 0], [parent_w, parent_h], [0, parent_h]], dtype=np.float32)
    
    for i in [0,2]:
        parent_ordered = np.roll(parent_ordered, -i, axis=0)

        M_parent2rect = cv2.getPerspectiveTransform(parent_ordered.astype(np.float32), rect_pts)
        rect_img = cv2.warpPerspective(resize_image, M_parent2rect, (parent_w, parent_h))

        qr_roi = rect_img[:int(parent_h * qr_scale), :int(parent_h * qr_scale)]
        # ROI 转灰度
        # qr_roi = cv2.cvtColor(qr_roi, cv2.COLOR_BGR2GRAY)
        # # 加大锐化：使用拉普拉斯算子
        # kernel = np.array([[0, -1, 0],
        #                    [-1, 5, -1],
        #                    [0, -1, 0]], dtype=np.float32)
        # qr_roi = cv2.filter2D(qr_roi, -1, kernel)

        qr_rois.append(qr_roi)
        # cv2.imwrite(os.path.join(debugdir, 'qr_roi.png'), qr_roi)

        # qr_code = pyzbar.decode(qr_roi)
        # datamatrix_code = pylibdmtx.decode(datamatrix)
        qr_code, points = obj.detectAndDecode(qr_roi)
        
        if len(qr_code)>0:
            break
    
    if len(qr_code)==0:
        parent_ordered = parent_ordered[::-1]
        child_ordered = child_ordered[::-1]

        for i in [0,2]:
            parent_ordered = np.roll(parent_ordered, -i, axis=0)

            M_parent2rect = cv2.getPerspectiveTransform(parent_ordered.astype(np.float32), rect_pts)
            rect_img = cv2.warpPerspective(resize_image, M_parent2rect, (parent_w, parent_h))

            qr_roi = rect_img[:int(parent_h * qr_scale), :int(parent_h * qr_scale)]
            
            # ROI 转灰度
            # qr_roi = cv2.cvtColor(qr_roi, cv2.COLOR_BGR2GRAY)
            # # 加大锐化：使用拉普拉斯算子
            # kernel = np.array([[0, -1, 0],
            #                 [-1, 5, -1],
            #                 [0, -1, 0]], dtype=np.float32)
            # qr_roi = cv2.filter2D(qr_roi, -1, kernel)

            qr_rois.append(qr_roi)
            # cv2.imwrite(os.path.join(debugdir, 'qr_roi.png'), qr_roi)
            # qr_code = pyzbar.decode(qr_roi)
            # datamatrix_code = pylibdmtx.decode(datamatrix)
            qr_code, points = obj.detectAndDecode(qr_roi)
            
            if len(qr_code)>0:
                break
    if len(qr_code)==0:
        print(f'未检测到二维码: {filepath}')
        if debugdir:
            for i, r in enumerate(qr_rois):
                cv2.imwrite(os.path.join(debugdir, f'qr_roi_{i}.png'), r)
        return None
    

    # 从二维码中获得标签名
    # info = qr_code[0].data.decode('utf-8').split('\n')
    info = qr_code[0].split('\n')
    labelname = info[0]
    pad = int(info[1])
    border_size = int(info[2])
    
    # 读取 label 图并获取其宽高
    label = cv2.imread(os.path.join(labeldir, labelname))
    if label is None:
        print(f'未找到标签图 {labelname}')
        return None

    # 对标签进行pad
    padlabel = cv2.copyMakeBorder(label, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # 对标签框进行偏移(如果是框数据的话)
    # TODO
    #############################################

    padlabel[np.all(padlabel != (255, 255, 255), axis=2)] = (0, 0, 0)
    # 将 label 图取反：白变黑，黑变白
    padlabel = cv2.bitwise_not(padlabel)
    label_h, label_w = padlabel.shape[:2]

    # 将 label 的四个角点映射到原图对应 parent_ordered 的位置
    label_pts = np.array([[0, 0], [label_w, 0], [label_w, label_h], [0, label_h]], dtype=np.float32)
    # 缩放回原尺度坐标
    parent_pts = parent_ordered.astype(np.float32) / scale

    # 计算 原图 到标签的透视变换矩阵
    M_parent2label = cv2.getPerspectiveTransform(parent_pts , label_pts)
    warped_image = cv2.warpPerspective(src_image, M_parent2label, (label_w, label_h))
    warped_image = warped_image[pad:]
    # if len(debugdir):
    #     cv2.imwrite(os.path.join(debugdir, 'warped_image.png'), warped_image)

    # 计算label到原图的透视变换矩阵
    M_label2parent = cv2.getPerspectiveTransform(label_pts, parent_pts)

    warped_label = cv2.warpPerspective(padlabel, M_label2parent, (src_image.shape[1], src_image.shape[0]))
    # if len(debugdir):
    #     cv2.imwrite(os.path.join(debugdir, 'warped_label.png'), warped_label)

    # 将变换后的 label 叠加到原图（可选：用 alpha 混合或覆盖）
    if len(debugdir):
        alpha = 0.4
        overlay = cv2.addWeighted(src_image, 1 - alpha, warped_label, alpha, 0)
        cv2.imwrite(os.path.join(debugdir, 'overlay.png'), overlay)

        
        overlay = cv2.addWeighted(warped_image, 1 - alpha, label, alpha, 0)
        cv2.imwrite(os.path.join(debugdir, 'overlay2.png'), overlay)
    
    return pad, border_size, M_parent2label, warped_image, M_label2parent, warped_label


if __name__ == '__main__':
    testdir = '/home/zjx/work/data/11-12-1' # 拍照图片目录
    labeldir = '/home/zjx/work/gitwork/sp_tools/data/xml_data/images' # 分割标签目录
    outputdir = './outputs' # 输出目录
    debugdir = './debugs' # 调试目录
    import shutil
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    # 解码二维码
    for filename in os.listdir(testdir):
        filepath = os.path.join(testdir, filename)
        result = decode(filepath, labeldir=labeldir, scale = 1, arc_scale=0.01, qr_scale=0.1,debugdir=debugdir)
        if result is None:
            print(f'转换失败: {filename}')
        else:
            print(f'转换成功: {filename}')
            pad, border_size, M_parent2label, warped_image, M_label2parent, warped_label = result
            # 保存M_parent2label, warped_image, M_label2parent, warped_label
            # np.save(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_img2label.npy'), M_parent2label)

            info = {
                'pad': pad,
                'border_size': border_size,
                'M_label2parent': M_label2parent,
                'M_parent2label': M_parent2label,
            }
            np.save(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_info.npy'), info)

            cv2.imwrite(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_warped_image.png'), warped_image)
            cv2.imwrite(os.path.join(outputdir, f'{os.path.splitext(filename)[0]}_warped_label.png'), warped_label)

