import cv2
import numpy as np
import os
import shutil
import math
import torch
import torch.nn.functional as F
from scipy.signal import correlate2d

def save_step(out_dir: str, name: str, img):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name), img)



def detect_corners_region_edge(
    img_path: str,
    original_path :str,
    out_dir: str = "debug_region_edge",
):
    """
    使用 “黑色框区域 + 边缘” 的方法检测黑色矩形框的外角点和内角点，
    并把每个中间步骤输出成图片文件。

    返回:
        outer_corners: (4, 2) float32，按 [左上, 右上, 右下, 左下] 顺序
        inner_corners: (4, 2) float32，同上
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Step 0: 读取拍照图片 ----------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    save_step(out_dir, "step0_original.jpg", img)

    H, W = img.shape[:2]

    # ---------- Step 1: 灰度 ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step(out_dir, "step1_gray.jpg", gray)

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
    save_step(out_dir, "step2_binary.jpg", bin_img)
    # ---------- Step 2.2: 连通域分析（只保留最大区域 = 黑框） ----------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8
    )

    # 排除背景 label=0，选择面积最大的
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = np.argmax(areas) + 1  # +1 因为排除了背景

    # 得到黑框 mask
    mask_black_frame = (labels == max_label).astype(np.uint8) * 255

    save_step(out_dir, "step2_binary_max_region.jpg", mask_black_frame)

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
        thickness = 20   # fallback
    else:
        # 取出现次数最多的宽度
        vals, counts = np.unique(widths, return_counts=True)
        thickness = int(vals[np.argmax(counts)])

    print("Estimated line thickness =", thickness)


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

            if np.sum(mask[y - other_min_side: y + other_min_side, mid] == 255) < other_min_side*2:
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
            if np.sum(mask[mid, x - other_min_side: x + other_min_side] == 255) < other_min_side*2:
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
        "left":   (0, 0, 255),    # 红
        "right":  (255, 0, 0),
        "top":    (0, 255, 0),
        "bottom": (255, 0, 255),
    }

    # 画中心线（粗线中线）
    for name, pts in centerlines.items():
        col = colors[name]
        for h, w in pts:
            cv2.circle(vis, (int(w), int(h)), 1, col, -1)

    save_step(out_dir, "step3_thick_border_centerlines.jpg", vis)


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
            yW = -(c + a*(W-1)) / b
            if 0 <= yW < H:
                pts.append((W-1, int(yW)))

        # 与 y=0 的交点
        if abs(a) > 1e-6:
            x0 = -c / a
            if 0 <= x0 < W:
                pts.append((int(x0), 0))

        # 与 y=H-1 的交点
        if abs(a) > 1e-6:
            xH = -(c + b*(H-1)) / a
            if 0 <= xH < W:
                pts.append((int(xH), H-1))

        # 至少两个点才能画线
        if len(pts) >= 2:
            cv2.line(vis, pts[0], pts[1], color, thickness)


    def fit_line(points):
        """
        最小二乘拟合直线 ax + by + c = 0
        输入: points (N,2)
        输出: (a,b,c)
        """
        x = points[:,1]
        y = points[:,0]
        # 拟合 y = kx + b
        A = np.vstack([x, np.ones_like(x)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # 转换成 ax + by + c = 0 → kx - y + b = 0
        a = k
        b2 = -1
        c = b
        return a, b2, c


    def intersect(line1, line2):
        a1,b1,c1 = line1
        a2,b2,c2 = line2

        d = a1*b2 - a2*b1
        if abs(d) < 1e-9:
            return None  # 平行

        w = (b1*c2 - b2*c1) / d
        h = (c1*a2 - c2*a1) / d
        return np.array([h,w], dtype=float)


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

    bottom_head = bottom[ :cut_w_line_len, :]
    bottom_tail = bottom[-cut_w_line_len:, :]

    left_head = left[:cut_h_line_len, :]
    left_tail = left[-cut_h_line_len:, :] 

    right_head = right[:cut_h_line_len, :]
    right_tail = right[-cut_h_line_len:, :] 


    # 拟合 8 条直线
    L_top_head    = fit_line(top_head)
    L_top_tail    = fit_line(top_tail)
    L_bottom_head = fit_line(bottom_head)
    L_bottom_tail = fit_line(bottom_tail)
    L_left_head   = fit_line(left_head)
    L_left_tail   = fit_line(left_tail)
    L_right_head  = fit_line(right_head)
    L_right_tail  = fit_line(right_tail)

    # ---------- 求四个角 ----------
    TL = intersect(L_top_head,    L_left_head)
    TR = intersect(L_top_tail,    L_right_head)
    BR = intersect(L_bottom_tail, L_right_tail)
    BL = intersect(L_bottom_head, L_left_tail)

    corners = np.array([TL, TR, BR, BL], dtype=float)

    # ---------- 可视化 ----------

    # ---------- Step4：可视化 8 条拟合线 ----------

    vis_lines = cv2.cvtColor(mask_black_frame, cv2.COLOR_GRAY2BGR)

    colors1 = {
        "top_head":    (0,   0, 255),   # 红
        "top_tail":    (0, 255, 255),   # 黄
        "bottom_head": (0, 255, 0),     # 绿
        "bottom_tail": (255,255, 0),    # 青
        "left_head":   (255, 0, 0),     # 蓝
        "left_tail":   (255, 0,255),    # 紫
        "right_head":  (255,128, 0),    # 橙
        "right_tail":  (128,255,128),   # 淡绿
    }

    draw_line(vis_lines, L_top_head,    colors1["top_head"],    2)
    draw_line(vis_lines, L_top_tail,    colors1["top_tail"],    2)
    draw_line(vis_lines, L_bottom_head, colors1["bottom_head"], 2)
    draw_line(vis_lines, L_bottom_tail, colors1["bottom_tail"], 2)
    draw_line(vis_lines, L_left_head,   colors1["left_head"],   2)
    draw_line(vis_lines, L_left_tail,   colors1["left_tail"],   2)
    draw_line(vis_lines, L_right_head,  colors1["right_head"],  2)
    draw_line(vis_lines, L_right_tail,  colors1["right_tail"],  2)

    save_step(out_dir, "step4_fitted_8lines.jpg", vis_lines)

    vis4 = cv2.cvtColor(mask_black_frame, cv2.COLOR_GRAY2BGR)
    for h, w in corners:
        cv2.circle(vis4, (int(w), int(h)), 3, (0,0,255), -1)


    # 画中心线（粗线中线）
    for name, pts in centerlines.items():
        col = colors[name]
        for h, w in pts:
            cv2.circle(vis4, (int(w), int(h)), 1, col, -1)

    save_step(out_dir, "step4_corner_fit_8lines.jpg", vis4)



    # ---------- Step 5: 读取原始图片 投影 ----------
    original_img = cv2.imread(original_path)
    if original_img is None:
        raise FileNotFoundError(f"Cannot read image: {original_path}")

    pad = 26 //2

    ori_H, ori_W = original_img.shape[:2]
    cut_original_img = original_img[pad:ori_H-pad, pad:ori_W-pad, ]
    cut_H, cut_W = cut_original_img.shape[:2]

    ori_out = 255 - cut_original_img
    save_step(out_dir, "step5_1_ori_grid_points.jpg", ori_out)


    def overlay_binary_mask_on_image(base, warp, alpha=0.5):
        """
        base_img:       拍照图 BGR
        warped_binary:  warp 后的 cut_original_img（0/255）
        alpha:          红色透明度
        """


        warp[..., 0] = 0  # B
        warp[..., 1] = 0

        out = base*0.5  + warp*0.5
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
    

    src_corners_xy = np.float32([
        [0, 0],
        [0, cut_W-1],
        [cut_H-1, cut_W-1],
        [cut_H-1, 0],
    ])


    H, W = img.shape[:2]
    dst_img = np.zeros(img.shape, dtype=img.dtype)
    warp_single_quad_hw(ori_out, src_corners_xy, corners, H, W, dst_img)

    overlay = overlay_binary_mask_on_image(img, dst_img, alpha=0.5)
    save_step(out_dir, "step5_img_mix_result.png", overlay)

    clean_bin_rgb = cv2.cvtColor(clean_bin_img, cv2.COLOR_GRAY2BGR)
    overlay = overlay_binary_mask_on_image(clean_bin_rgb, dst_img, alpha=0.5)
    save_step(out_dir, "step5_clean_bin_mix_result.png", overlay)


    # ---------- Step6：调整四个点 ----------
    ori_out_gray = cv2.cvtColor(ori_out, cv2.COLOR_BGR2GRAY)
    ksize = 7
    ori_out
    ori_out_gray_blur = cv2.GaussianBlur(ori_out_gray, (ksize, ksize), sigmaX=0)
    clean_bin_blur = cv2.GaussianBlur(clean_bin_img, (ksize, ksize), sigmaX=0)

    save_step(out_dir, "step6_1_ori_out_gray_blur.png", ori_out_gray_blur)
    save_step(out_dir, "step6_2_clean_bin_rgb_blurr.png", clean_bin_blur)



    def opt_porin(delta):

        def cal_iou(ori, base, src_quad, dst_quad):
            H, W = base.shape[:2]
            dst_img = np.zeros(base.shape, dtype=img.dtype)
            warp_single_quad_hw(ori, src_quad, dst_quad, H, W, dst_img)


            img_iou = (dst_img /255.0) * (base/255.0)

            ov = np.sum(img_iou)/(H * W)
            return ov


        for i in range(4):
            corners_l = corners.copy()
            corners_l[i,1] -= delta
            corners_r = corners.copy()
            corners_r[i,1] += delta

            iou = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners)
            ioul = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_l)
            iour = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_r)

            if ioul > iou and ioul> iour:
                dw = -delta
            elif iour>iou and iour> ioul:
                dw = delta
            else:
                dw = 0
            corners[i,1] += dw


        for i in range(4):
            corners_t = corners.copy()
            corners_t[i,0] -= delta
            corners_b = corners.copy()
            corners_b[i,0] += delta

            iou = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners)
            iout = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_t)
            ioub = cal_iou(ori_out_gray_blur, clean_bin_blur, src_corners_xy, corners_b)

            if iout > iou and iout> ioub:
                dh = -delta
            elif ioub > iou and ioub> iout:
                dh = delta
            else:
                dh = 0

            corners[i,0] += dh

    iter = 5
    delta = 1
    for n in range(iter):
        opt_porin(delta)


    H, W = img.shape[:2]
    dst_img = np.zeros(img.shape, dtype=img.dtype)
    warp_single_quad_hw(ori_out, src_corners_xy, corners, H, W, dst_img)

    overlay = overlay_binary_mask_on_image(img, dst_img, alpha=0.5)
    save_step(out_dir, "step6_3_img_mix_result.png", overlay)


    clean_bin_rgb = cv2.cvtColor(clean_bin_img, cv2.COLOR_GRAY2BGR)
    overlay = overlay_binary_mask_on_image(clean_bin_rgb, dst_img, alpha=0.5)
    save_step(out_dir, "step6_4_clean_bin_mix_result.png", overlay)

if __name__ == "__main__":
    out_dir = "/home/glq/sp_tools/debug_region_edge"
    # img_path = "/home/glq/sp_tools/test_data/IMG20251129154449.jpg"
    # original_path = "/home/glq/sp_tools/test_data/0a1a7e38-afee-433b-a16d-8f4f46ecdf07-2.png"

    # img_path = "/home/glq/sp_tools/test_data/IMG20251129155059.jpg"
    # original_path = "/home/glq/sp_tools/test_data/0a1a7e38-afee-433b-a16d-8f4f46ecdf07-13.png"
    

    img_path = "/home/glq/sp_tools/test_data/IMG20251129155127.jpg"
    original_path = "/home/glq/sp_tools/test_data/0a2b75e5-a6f1-4615-bdd5-0811364bbde4-1.png"

    img_path = "/data/xml_data/photos_cvt/d433e6d5-d057-45d4-bd44-1e910ccf56c4/ordered.png"
    original_path

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)   # 相当于 rm -rf out_dir
    os.makedirs(out_dir)     


    detect_corners_region_edge(
        img_path,
        original_path,
        out_dir=out_dir,
    )
