import os
from uuid import uuid4
import numpy as np
import cv2
import shutil
from multiprocessing import Pool, cpu_count
import functools
import json


def save_step(debugdir: str, name: str, img):
    if debugdir != "":
        os.makedirs(debugdir, exist_ok=True)
        cv2.imwrite(os.path.join(debugdir, name), img)


def detect_corners_region_edge(
    img,
    original_img,
    pad,
    child_ordered=None,
    debugdir="",
):

    # ---------- Step 0: 读取拍照图片 ----------
    save_step(debugdir, "step0_original.jpg", img)

    H, W = img.shape[:2]

    # ---------- Step 1: 灰度 ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step(debugdir, "step1_gray.jpg", gray)

    # ---------- Step 2: 自适应二值化，得到黑框区域 ----------
    # 黑色 → 白（255），背景 → 黑（0）
    bin_img = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=51,  # 根据图像分辨率可适当调大/调小
        C=5,
    )
    save_step(debugdir, "step2_binary.jpg", bin_img)
    # ---------- Step 2.2: 连通域分析（只保留最大区域 = 黑框） ----------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    # 排除背景 label=0，选择面积最大的
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = np.argmax(areas) + 1  # +1 因为排除了背景

    # 得到黑框 mask
    mask_black_frame = (labels == max_label).astype(np.uint8) * 255

    def fill_small_holes(bin_mask: np.ndarray, max_hole_area: int = 5000) -> np.ndarray:

        h, w = bin_mask.shape[:2]

        # 2. 所有黑区域：0 -> 255，255 -> 0
        #    这一步只是为了让 connectedComponents 以“黑”为前景算连通域
        background = (bin_mask == 0).astype(np.uint8) * 255

        # 3. 对黑区域做连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            background, connectivity=8
        )

        filled = bin_mask.copy()

        # 4. 只填「不碰边界 + 面积小于阈值」的黑连通域
        for label in range(1, num_labels):  # 0 是 background 的背景
            x, y, cw, ch, area = stats[label]

            touches_border = x == 0 or y == 0 or x + cw == w or y + ch == h

            # 条件：不碰边界 且 面积不大 ⇒ 认为是洞
            if (not touches_border) and (area <= max_hole_area):
                filled[labels == label] = 255

        return filled

    mask_black_frame = fill_small_holes(mask_black_frame)

    save_step(debugdir, "step2_binary_max_r_egion.jpg", mask_black_frame)

    clean_bin_img = bin_img

    # ---------- Step 3: 粗线（白框）的中心线 ----------

    mask = mask_black_frame
    H, W = mask.shape

    # ---- Step: 估计白线线宽 thickness ----

    def split_runs(indices):
        """把连续像素分成若干段"""
        if len(indices) == 0:
            return []
        runs = []
        start = indices[0]
        prev = indices[0]
        for v in indices[1:]:
            if v == prev + 1:
                prev = v
            else:
                runs.append((start, prev))
                start = v
                prev = v
        runs.append((start, prev))
        return runs

    widths = []

    # 扫描所有行
    for y in range(H):
        xs = np.where(mask_black_frame[y] == 255)[0]
        runs = split_runs(xs)
        for s, e in runs:
            width = e - s + 1
            # 过滤小噪点，典型厚度不可能小于 5
            if width > 5:
                widths.append(width)

    # 如果高度>宽度，可继续沿列方向扫描，提高鲁棒性
    for x in range(W):
        ys = np.where(mask_black_frame[:, x] == 255)[0]
        runs = split_runs(ys)
        for s, e in runs:
            height = e - s + 1
            if height > 5:
                widths.append(height)

    # ---- 最终厚度 = 众数（最常见的 run length） ----
    if len(widths) == 0:
        thickness = 20  # fallback
    else:
        # 取出现次数最多的宽度
        vals, counts = np.unique(widths, return_counts=True)
        thickness = int(vals[np.argmax(counts)])

    # print("Estimated line thickness =", thickness)

    line_thickness = thickness
    # 允许一定浮动
    max_thickness_for_side = line_thickness * 2
    min_thickness_for_side = line_thickness * 0.5

    other_min_side = int(min_thickness_for_side * 0.7)

    # 1) 竖直边：对每一行找“左边框”和“右边框”的中心点
    left_centerline = []
    right_centerline = []

    for y in range(H):
        xs = np.where(mask[y] == 255)[0]
        if len(xs) == 0:
            continue
        runs = split_runs(xs)
        for s, e in runs:
            w_run = e - s + 1
            # 太宽的段（比如顶边整条横线）跳过
            if w_run > max_thickness_for_side or w_run < min_thickness_for_side:
                continue
            mid = int((s + e) / 2.0 + 0.5)

            if (
                np.sum(mask[y - other_min_side : y + other_min_side, mid] == 255)
                < other_min_side * 2
            ):
                continue

            if mid < W / 2.0:
                left_centerline.append((y, mid))
            else:
                right_centerline.append((y, mid))

    left_centerline = np.array(left_centerline, dtype=float)
    right_centerline = np.array(right_centerline, dtype=float)

    # 2) 水平边：对每一列找“上边框”和“下边框”的中心点
    top_centerline = []
    bottom_centerline = []

    for x in range(W):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) == 0:
            continue
        runs = split_runs(ys)
        for s, e in runs:
            h_run = e - s + 1

            if h_run > max_thickness_for_side or h_run < min_thickness_for_side:
                continue

            mid = int((s + e) / 2.0 + 0.5)
            if (
                np.sum(mask[mid, x - other_min_side : x + other_min_side] == 255)
                < other_min_side * 2
            ):
                continue

            mid = (s + e) / 2.0
            if mid < H / 2.0:
                top_centerline.append((mid, x))
            else:
                bottom_centerline.append((mid, x))

    top_centerline = np.array(top_centerline, dtype=float)
    bottom_centerline = np.array(bottom_centerline, dtype=float)

    left_centerline = left_centerline[np.argsort(left_centerline[:, 0])]
    right_centerline = right_centerline[np.argsort(right_centerline[:, 0])]
    top_centerline = top_centerline[np.argsort(top_centerline[:, 1])]
    bottom_centerline = bottom_centerline[np.argsort(bottom_centerline[:, 1])]

    # nms

    # 汇总
    centerlines = {
        "left": left_centerline,
        "right": right_centerline,
        "top": top_centerline,
        "bottom": bottom_centerline,
    }

    # ----------可视化：粗线的中心线 ----------
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    colors = {
        "left": (0, 0, 255),  # 红
        "right": (255, 0, 0),
        "top": (0, 255, 0),
        "bottom": (255, 0, 255),
    }

    # 画中心线（粗线中线）
    for name, pts in centerlines.items():
        col = colors[name]
        for h, w in pts:
            cv2.circle(vis, (int(w), int(h)), 1, col, -1)

    save_step(debugdir, "step3_thick_border_centerlines.jpg", vis)

    # ---------- Step 4: 用局部拟合延长线求角点 ----------

    def draw_line(vis, line, color, thickness=2):
        """
        line: (a,b,c) for ax + by + c = 0
        在整张图上绘制直线
        """
        a, b, c = line
        H, W = vis.shape[:2]

        pts = []

        # 与 x=0 的交点
        # ax + b y + c = 0 → y = -(c + a*0)/b
        if abs(b) > 1e-6:
            y0 = -c / b
            if 0 <= y0 < H:
                pts.append((0, int(y0)))

        # 与 x=W-1 的交点
        if abs(b) > 1e-6:
            yW = -(c + a * (W - 1)) / b
            if 0 <= yW < H:
                pts.append((W - 1, int(yW)))

        # 与 y=0 的交点
        if abs(a) > 1e-6:
            x0 = -c / a
            if 0 <= x0 < W:
                pts.append((int(x0), 0))

        # 与 y=H-1 的交点
        if abs(a) > 1e-6:
            xH = -(c + b * (H - 1)) / a
            if 0 <= xH < W:
                pts.append((int(xH), H - 1))

        # 至少两个点才能画线
        if len(pts) >= 2:
            cv2.line(vis, pts[0], pts[1], color, thickness)

    def fit_line(points):
        """
        最小二乘拟合直线 ax + by + c = 0
        输入: points (N,2)
        输出: (a,b,c)
        """
        x = points[:, 1]
        y = points[:, 0]
        # 拟合 y = kx + b
        A = np.vstack([x, np.ones_like(x)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # 转换成 ax + by + c = 0 → kx - y + b = 0
        a = k
        b2 = -1
        c = b
        return a, b2, c

    def intersect(line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2

        d = a1 * b2 - a2 * b1
        if abs(d) < 1e-9:
            return None  # 平行

        w = (b1 * c2 - b2 * c1) / d
        h = (c1 * a2 - c2 * a1) / d
        return np.array([h, w], dtype=float)

    cut_w_line_len = int(W * 0.1)
    cut_h_line_len = int(H * 0.1)

    # 4 条边：top / right / bottom / left
    top = centerlines["top"]
    bottom = centerlines["bottom"]
    left = centerlines["left"]
    right = centerlines["right"]

    # 取每条边的前后 15% 段
    top_head = top[:cut_w_line_len, :]
    top_tail = top[-cut_w_line_len:, :]

    bottom_head = bottom[:cut_w_line_len, :]
    bottom_tail = bottom[-cut_w_line_len:, :]

    left_head = left[:cut_h_line_len, :]
    left_tail = left[-cut_h_line_len:, :]

    right_head = right[:cut_h_line_len, :]
    right_tail = right[-cut_h_line_len:, :]

    # 拟合 8 条直线
    L_top_head = fit_line(top_head)
    L_top_tail = fit_line(top_tail)
    L_bottom_head = fit_line(bottom_head)
    L_bottom_tail = fit_line(bottom_tail)
    L_left_head = fit_line(left_head)
    L_left_tail = fit_line(left_tail)
    L_right_head = fit_line(right_head)
    L_right_tail = fit_line(right_tail)

    # ---------- 求四个角 ----------
    TL = intersect(L_top_head, L_left_head)
    TR = intersect(L_top_tail, L_right_head)
    BR = intersect(L_bottom_tail, L_right_tail)
    BL = intersect(L_bottom_head, L_left_tail)

    # child_ordered=child_ordered
    corners = np.array([TL, TR, BR, BL], dtype=float)

    # ---------- 可视化 ----------

    # ---------- Step4：可视化 8 条拟合线 ----------

    vis_lines = cv2.cvtColor(mask_black_frame, cv2.COLOR_GRAY2BGR)

    colors1 = {
        "top_head": (0, 0, 255),  # 红
        "top_tail": (0, 255, 255),  # 黄
        "bottom_head": (0, 255, 0),  # 绿
        "bottom_tail": (255, 255, 0),  # 青
        "left_head": (255, 0, 0),  # 蓝
        "left_tail": (255, 0, 255),  # 紫
        "right_head": (255, 128, 0),  # 橙
        "right_tail": (128, 255, 128),  # 淡绿
    }

    draw_line(vis_lines, L_top_head, colors1["top_head"], 2)
    draw_line(vis_lines, L_top_tail, colors1["top_tail"], 2)
    draw_line(vis_lines, L_bottom_head, colors1["bottom_head"], 2)
    draw_line(vis_lines, L_bottom_tail, colors1["bottom_tail"], 2)
    draw_line(vis_lines, L_left_head, colors1["left_head"], 2)
    draw_line(vis_lines, L_left_tail, colors1["left_tail"], 2)
    draw_line(vis_lines, L_right_head, colors1["right_head"], 2)
    draw_line(vis_lines, L_right_tail, colors1["right_tail"], 2)

    save_step(debugdir, "step4_fitted_8lines.jpg", vis_lines)

    vis4 = cv2.cvtColor(mask_black_frame, cv2.COLOR_GRAY2BGR)
    for h, w in corners:
        cv2.circle(vis4, (int(w), int(h)), 3, (0, 0, 255), -1)

    # 画中心线（粗线中线）
    for name, pts in centerlines.items():
        col = colors[name]
        for h, w in pts:
            cv2.circle(vis4, (int(w), int(h)), 1, col, -1)

    save_step(debugdir, "step4_corner_fit_8lines.jpg", vis4)

    # ---------- Step 5: 读取原始图片 投影 ----------
    pad_half = pad // 2

    ori_H, ori_W = original_img.shape[:2]
    cut_original_img = original_img[
        pad_half : ori_H - pad_half,
        pad_half : ori_W - pad_half,
    ]
    cut_H, cut_W = cut_original_img.shape[:2]

    ori_out = 255 - cut_original_img
    save_step(debugdir, "step5_1_ori_grid_points.jpg", ori_out)

    def overlay_binary_mask_on_image(base, warp, alpha=0.5):
        """
        base_img:       拍照图 BGR
        warped_binary:  warp 后的 cut_original_img（0/255）
        alpha:          红色透明度
        """

        warp[..., 0] = 0  # B
        warp[..., 1] = 0

        out = base * 0.5 + warp * 0.5
        return out.astype(np.uint8)

    def warp_single_quad_hw(src, src_quad_hw, dst_quad_hw, out_h, out_w, dst_img):
        """
        在已有 dst_img 上贴入单块 warp 结果（无缝）。
        src_quad_hw / dst_quad_hw 均为 (h,w)
        """

        # === 1. 转成 OpenCV (x,y) ===
        src_quad_xy = np.float32([[w, h] for h, w in src_quad_hw])
        dst_quad_xy = np.float32([[w, h] for h, w in dst_quad_hw])

        # === 2. 计算 H ===
        H = cv2.getPerspectiveTransform(src_quad_xy, dst_quad_xy)

        # === 3. warp 整张（简单可靠） ===
        warped = cv2.warpPerspective(src, H, (out_w, out_h))

        # === 4. 对应目标区域 mask ===
        mask = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_quad_xy.astype(np.int32), 255)

        # === 5. 贴到大图（无缝） ===
        dst_img[mask == 255] = warped[mask == 255]

    src_corners_xy = np.float32(
        [
            [0, 0],
            [0, cut_W - 1],
            [cut_H - 1, cut_W - 1],
            [cut_H - 1, 0],
        ]
    )

    H, W = img.shape[:2]
    dst_img = np.zeros(img.shape, dtype=img.dtype)
    warp_single_quad_hw(ori_out, src_corners_xy, corners, H, W, dst_img)

    overlay = overlay_binary_mask_on_image(img, dst_img, alpha=0.5)
    save_step(debugdir, "step5_img_mix_result.png", overlay)

    clean_bin_rgb = cv2.cvtColor(clean_bin_img, cv2.COLOR_GRAY2BGR)
    overlay = overlay_binary_mask_on_image(clean_bin_rgb, dst_img, alpha=0.5)
    save_step(debugdir, "step5_clean_bin_mix_result.png", overlay)

    # ---------- Step6：调整四个点 ----------
    ori_out_gray = cv2.cvtColor(ori_out, cv2.COLOR_BGR2GRAY)
    ksize = 7
    ori_out
    ori_out_gray_blur = cv2.GaussianBlur(ori_out_gray, (ksize, ksize), sigmaX=0)
    clean_bin_blur = cv2.GaussianBlur(clean_bin_img, (ksize, ksize), sigmaX=0)

    save_step(debugdir, "step6_1_ori_out_gray_blur.png", ori_out_gray_blur)
    save_step(debugdir, "step6_2_clean_bin_rgb_blurr.png", clean_bin_blur)

    def opt_porin(delta):

        def cal_iou(ori, base, src_quad, dst_quad):
            H, W = base.shape[:2]
            dst_img = np.zeros(base.shape, dtype=img.dtype)
            warp_single_quad_hw(ori, src_quad, dst_quad, H, W, dst_img)

            img_iou = (dst_img / 255.0) * (base / 255.0)

            ov = np.sum(img_iou) / (H * W)
            return ov

        for i in range(4):
            corners_l = corners.copy()
            corners_l[i, 1] -= delta
            corners_r = corners.copy()
            corners_r[i, 1] += delta

            iou = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners)
            ioul = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_l)
            iour = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_r)

            if ioul > iou and ioul > iour:
                dw = -delta
            elif iour > iou and iour > ioul:
                dw = delta
            else:
                dw = 0
            corners[i, 1] += dw

        for i in range(4):
            corners_t = corners.copy()
            corners_t[i, 0] -= delta
            corners_b = corners.copy()
            corners_b[i, 0] += delta

            iou = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners)
            iout = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_t)
            ioub = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_b)

            if iout > iou and iout > ioub:
                dh = -delta
            elif ioub > iou and ioub > iout:
                dh = delta
            else:
                dh = 0

            corners[i, 0] += dh

    iter = 5
    delta = 1
    for n in range(iter):
        opt_porin(delta)

    H, W = img.shape[:2]
    dst_img = np.zeros(img.shape, dtype=img.dtype)
    warp_single_quad_hw(ori_out, src_corners_xy, corners, H, W, dst_img)

    overlay = overlay_binary_mask_on_image(img, dst_img, alpha=0.5)
    save_step(debugdir, "step6_3_img_mix_result.png", overlay)

    clean_bin_rgb = cv2.cvtColor(clean_bin_img, cv2.COLOR_GRAY2BGR)
    overlay = overlay_binary_mask_on_image(clean_bin_rgb, dst_img, alpha=0.5)
    save_step(debugdir, "step6_4_clean_bin_mix_result.png", overlay)

    src_quad_xy = np.float32([[w, h] for h, w in src_corners_xy])
    dst_quad_xy = np.float32([[w, h] for h, w in corners])

    clean_mask = clean_bin_img / 255.0
    iou = (dst_img[..., 2] / 225.0 * clean_mask).sum() / clean_mask.sum()

    return src_quad_xy, dst_quad_xy, iou


def decode(
    filepath,
    labeldir="./labels/",
    scale=2,
    arc_scale=0.01,
    qr_scale=0.15,
    min_area_scale=0.5,
    debugdir="",
):
    if len(debugdir):
        os.makedirs(debugdir, exist_ok=True)

    obj = cv2.wechat_qrcode_WeChatQRCode()
    src_image = cv2.imread(filepath)
    resize_image = cv2.resize(
        src_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

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
    gray = (gray * (1 - black_mask / 255)).astype(np.uint8)

    # gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=121, C=1
    )

    # 对二值图做开运算，先腐蚀再膨胀，去除小噪点并保持整体形状
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(
        os.path.join(debugdir, "photocopy.png"), gray
    )  # 这里的中间结果图可能可以作为影印版的数据增强用

    gray_canny = cv2.Canny(gray, 100, 300, 5, L2gradient=True)
    # 对二值图做膨胀运算，填补小孔洞，连接邻近区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gray_canny = cv2.dilate(gray_canny, kernel, iterations=1)
    gray_canny = cv2.morphologyEx(gray_canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(os.path.join(debugdir, "gray_canny.png"), gray_canny)

    # 寻找大四边形的轮廓
    contours, hierarchys = cv2.findContours(
        gray_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    quadrangle_contour_ids = []
    for i in range(len(contours)):
        contour = contours[i]

        epsilon = arc_scale * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if (
            cv2.contourArea(approx) / (resize_image.shape[0] * resize_image.shape[1])
            < min_area_scale
        ):  # 0.5:
            continue

        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
            and abs(cv2.contourArea(approx) - cv2.contourArea(contour))
            / cv2.contourArea(contour)
            < 0.2
        ):
            # 判断为四边形
            quadrangle_contour_ids.append(i)
            # cv2.drawContours(resize_image, [contour], 0, (0, 255, 0), 2)

    # cv2.imwrite(os.path.join(debugdir, 'quadrangle.png'), resize_image)

    # 寻找大四边环形
    parent_contour = None
    child_contour = None
    for i in quadrangle_contour_ids:
        child_id = hierarchys[0][i][2]  # 第一个子轮廓索引
        while child_id != -1:  # 遍历所有子轮廓
            if (
                child_id != i
                and child_id in quadrangle_contour_ids
                and 0.95
                < cv2.contourArea(contours[child_id]) / cv2.contourArea(contours[i])
                < 0.99
            ):
                # 该父子轮廓为曲谱轮廓
                parent_contour = contours[i]
                child_contour = contours[child_id]
                # cv2.drawContours(resize_image, [parent_contour], 0, (0, 255, 0), 2)
                # cv2.drawContours(resize_image, [child_contour], 0, (0, 0, 255), 2)
                break
            child_id = hierarchys[0][child_id][0]  # 同级下一个子轮廓

    if parent_contour is None:
        print(f"未检测到曲谱轮廓: {filepath}")
        return None

    # 获取父轮廓和子轮廓的角点
    # parent_pts = cv2.approxPolyDP(parent_contour, arc_scale * cv2.arcLength(parent_contour, True), True).reshape(4,2)
    child_pts = cv2.approxPolyDP(
        child_contour, arc_scale * cv2.arcLength(child_contour, True), True
    ).reshape(4, 2)

    # 后续只用到子轮廓
    child_pts = child_pts.reshape((4, 2))
    dists = np.linalg.norm(np.roll(child_pts, -1, axis=0) - child_pts, axis=1)
    start_idx = int(np.argmin(dists))
    child_ordered = np.roll(child_pts, -start_idx, axis=0)

    child_w = int(np.linalg.norm(child_ordered[0] - child_ordered[1]))
    child_h = int(np.linalg.norm(child_ordered[1] - child_ordered[2]))

    # 可视化对应点（调试用）
    if len(debugdir):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i in range(4):
            # cv2.circle(resize_image, tuple(parent_ordered[i].astype(int)), 1, colors[i], -1)
            cv2.circle(
                resize_image, tuple(child_ordered[i].astype(int)), 1, colors[i], -1
            )

        cv2.imwrite(os.path.join(debugdir, "ordered.png"), resize_image)

    # 通过变换后，获取二维码区域，并识别
    qr_rois = []
    rect_pts = np.array(
        [[0, 0], [child_w, 0], [child_w, child_h], [0, child_h]], dtype=np.float32
    )

    for i in [0, 2]:
        child_ordered = np.roll(child_ordered, -i, axis=0)

        M_child2rect = cv2.getPerspectiveTransform(
            child_ordered.astype(np.float32), rect_pts
        )
        rect_img = cv2.warpPerspective(resize_image, M_child2rect, (child_w, child_h))

        qr_roi = rect_img[: int(child_h * qr_scale), : int(child_w * qr_scale)]
        qr_rois.append(qr_roi)
        qr_code, points = obj.detectAndDecode(qr_roi)

        if len(qr_code) > 0:
            break

    if len(qr_code) == 0:
        # parent_ordered = parent_ordered[::-1]
        child_ordered = child_ordered[::-1]

        for i in [0, 2]:
            child_ordered = np.roll(child_ordered, -i, axis=0)

            M_child2rect = cv2.getPerspectiveTransform(
                child_ordered.astype(np.float32), rect_pts
            )
            rect_img = cv2.warpPerspective(
                resize_image, M_child2rect, (child_w, child_h)
            )
            qr_roi = rect_img[: int(child_h * qr_scale), : int(child_w * qr_scale)]
            qr_rois.append(qr_roi)
            qr_code, points = obj.detectAndDecode(qr_roi)

            if len(qr_code) > 0:
                break

    if len(qr_code) == 0:
        print(f"未检测到二维码: {filepath}")
        if debugdir:
            for i, r in enumerate(qr_rois):
                cv2.imwrite(os.path.join(debugdir, f"qr_roi_{i}.png"), r)
        return None

    # 从二维码中获得标签名
    info = qr_code[0].split("\n")
    labelname = info[0]
    pad = int(info[1])
    border_size = int(info[2])

    # 读取 label 图并获取其宽高
    label = cv2.imread(os.path.join(labeldir, labelname))
    label_size = label.shape
    if label is None:
        print(f"未找到标签图 {labelname}")
        return None

    # 对标签进行pad
    padlabel = cv2.copyMakeBorder(
        label, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    padlabel = cv2.copyMakeBorder(
        padlabel,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    # 对标签框进行偏移(如果是框数据的话)
    # TODO
    #############################################

    padlabel[np.all(padlabel != (255, 255, 255), axis=2)] = (0, 0, 0)
    # 将 label 图取反：白变黑，黑变白
    padlabel = cv2.bitwise_not(padlabel)
    label_h, label_w = padlabel.shape[:2]
    label_w -= 2 * border_size
    label_h -= 2 * border_size

    child_pts = child_ordered.astype(np.float32) / scale

    # 获取变换矩阵
    # src_corners_xy : label 的坐标
    # dst_quad_xy ： 拍照图的坐标
    src_quad_xy, dst_quad_xy, iou = detect_corners_region_edge(
        src_image,
        label,
        border_size,
        child_ordered=child_ordered,
        debugdir=debugdir,
    )

    # 计算 原图 到标签的透视变换矩阵
    M_child2label = cv2.getPerspectiveTransform(dst_quad_xy, src_quad_xy)
    warped_image = cv2.warpPerspective(
        src_image, M_child2label, (label_w + border_size, label_h + border_size)
    )
    warped_image = warped_image[border_size + pad :, border_size:]

    # 计算label到原图的透视变换矩阵
    M_label2child = cv2.getPerspectiveTransform(src_quad_xy, dst_quad_xy)
    warped_label = cv2.warpPerspective(
        padlabel, M_label2child, (src_image.shape[1], src_image.shape[0])
    )

    return (
        label_size,
        labelname,
        pad,
        border_size,
        M_child2label,
        warped_image,
        M_label2child,
        warped_label,
        iou,
    )


def process_single_image(filepath, outputdir, labeldir):
    """单张图片处理函数，供多进程调用"""
    new_dir_name = str(uuid4())
    subdir = os.path.join(outputdir, new_dir_name)
    os.makedirs(subdir, exist_ok=True)
    shutil.copy(filepath, os.path.join(subdir, "src_image.png"))
    result = decode(
        filepath,
        labeldir=labeldir,
        scale=1,
        arc_scale=0.01,
        qr_scale=0.15,
        min_area_scale=0.5,
        debugdir=subdir,
    )
    if result is None:
        print(f"转换失败: {filepath}")
        return None
    else:
        (
            label_size,
            labelname,
            pad,
            border_size,
            M_src2label,
            warped_image,
            M_label2src,
            warped_label,
            iou,
        ) = result
        print(f"转换成功: iou:{iou}, {filepath}, {subdir}")
        info = {
            "pad": pad,
            "iou": iou,
            "labelname": labelname,
            "label_size": label_size,
            "border_size": border_size,
            "M_label2src": M_label2src,
            "M_src2label": M_src2label,
        }
        np.save(os.path.join(subdir, f"{new_dir_name}_info.npy"), info)
        cv2.imwrite(
            os.path.join(subdir, f"{new_dir_name}_warped_image.png"), warped_image
        )
        cv2.imwrite(
            os.path.join(subdir, f"{new_dir_name}_warped_label.png"), warped_label
        )
        debug_info = {
            "src_img": filepath,
            "label_name": labelname,
            "pad": pad,
            "iou": iou,
            "label_size": label_size,
            "border_size": border_size,
            "M_label2src": M_label2src.tolist(),
            "M_src2label": M_src2label.tolist(),
        }
        with open(os.path.join(subdir, f"debug_info.json"), "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        return filepath


def test_decode_img():

    # 单图
    # /home/glq/sp_tools/output/fc2c13e4-8316-4055-a7af-29141dd5ad1b
    src_path = (
        "/data/xml_data/photos/2025111901/5547dd9c-6a09-414f-a967-63ffd1314371-26.png"
    )
    label_dir = "/data/xml_data/generate/"
    decode(src_path, label_dir, debugdir="/home/glq/sp_tools/debug/")


def test_single_image():

    src_path = (
        "/data/xml_data/photos/2025111901/5547dd9c-6a09-414f-a967-63ffd1314371-26.png"
    )
    label_dir = "/data/xml_data/generate/"
    process_single_image(src_path, "/home/glq/sp_tools/debug", label_dir)


def all_convter():
    testdir = "/data/xml_data/photos/2025111901"  # 拍照图片目录
    labeldir = "/data/xml_data/generate"  # 分割标签目录
    outputdir = "/home/glq/sp_tools/output/"  # 输出目录

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir, exist_ok=True)

    # 解码二维码（多进程）
    filepaths = [
        os.path.join(testdir, f)
        for f in os.listdir(testdir)
        if os.path.isfile(os.path.join(testdir, f))
    ]

    # 根据CPU核心数设置进程池大小，留1核心给系统
    pool_size = max(1, cpu_count() - 1)
    with Pool(processes=pool_size) as pool:
        # 使用functools.partial固定其他参数
        worker = functools.partial(
            process_single_image, outputdir=outputdir, labeldir=labeldir
        )
        # 并行处理所有图片
        pool.map(worker, filepaths)


if __name__ == "__main__":
    # test_decode_img()
    # test_single_image()
    all_convter()
