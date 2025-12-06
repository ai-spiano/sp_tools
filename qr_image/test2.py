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
    inner_offset: float = 30.0,
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



    # Step 1 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2 光照均衡
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, blur, scale=255)

    # Step 3 OTSU 阈值
    _, clean_bin_img = cv2.threshold(
        norm, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 4 噪点去除
    kernel = np.ones((3,3), np.uint8)
    clean_bin_img = cv2.morphologyEx(clean_bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_bin_img = cv2.morphologyEx(clean_bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    save_step(out_dir, "step2_binary_clean.jpg", clean_bin_img)



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


    cut_line_len = int(W * 0.05)

    # 4 条边：top / right / bottom / left
    top = centerlines["top"]
    bottom = centerlines["bottom"]
    left = centerlines["left"]
    right = centerlines["right"]

    # 取每条边的前后 15% 段
    top_head = top[:cut_line_len, :]
    top_tail = top[-cut_line_len:, :]

    bottom_head = bottom[ :cut_line_len, :]
    bottom_tail = bottom[-cut_line_len:, :]

    left_head = left[:cut_line_len, :]
    left_tail = left[-cut_line_len:, :] 

    right_head = right[:cut_line_len, :]
    right_tail = right[-cut_line_len:, :] 


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

    # 切分成n个小格子 64x64

    def build_src_grid(src_w: int, src_h: int, cell: int = 64):
        """
        把原图划分成固定大小 (cell x cell) 的网格，
        返回网格点矩阵 grid[j,i] = (x,y)
        
        grid shape = (num_rows+1, num_cols+1, 2)
        """
        num_cols = src_w // cell + 1
        num_rows = src_h // cell + 1

        # 最后多出来的边（不足 64 像素）先不处理，可后续单独补
        grid = np.zeros((num_rows + 1, num_cols + 1, 2), dtype=np.float32)

        for j in range(num_rows + 1):
            h = min(j * cell, src_h - 1)
            for i in range(num_cols + 1):
                w = min(i * cell, src_w - 1)
                grid[j, i] = (h, w)

        return grid, num_cols, num_rows
    
    grid_size = 512

    cut_H, cut_W = cut_original_img.shape[:2]
    cut_grid, _, _= build_src_grid(cut_W, cut_H, grid_size)

    vis_pts = cut_original_img.copy()
    
    ny, nx, _ = cut_grid.shape # 对应 build_src_grid 里的 ny, nx
    for j in range(ny):
        for i in range(nx):
            h, w = cut_grid[j, i]
            cv2.circle(vis_pts, (int(w), int(h)), 2, (0, 0, 255), -1)  # 红点

    save_step(out_dir, "step5_ori_grid_points.jpg", vis_pts)





    # step 6 初步投影
    def build_dst_grid_from_4corners(ori_grid, src_h, src_w, corners):
        """
        支持 (h, w) (即 (y, x)) 方式存储的网格。
        
        ori_grid[j,i] = (src_y, src_x)
        corners 顺序保持: [TL, TR, BR, BL] = 左上, 右上, 右下, 左下

        返回 dst_grid[j,i] = (dst_y, dst_x)
        """
        # 拍摄图的四个角点 (dst_y, dst_x)
        TL, TR, BR, BL = corners.astype(np.float32)   # 每个也是(Y, X)

        ny_plus, nx_plus, _ = ori_grid.shape

        dst_grid = np.zeros_like(ori_grid, dtype=np.float32)

        for j in range(ny_plus):
            for i in range(nx_plus):

                src_y, src_x = ori_grid[j, i]  # 注意顺序 h,w

                # 归一化坐标
                v = src_y / (src_h-1)   # 高度方向
                u = src_x / (src_w-1)   # 宽度方向

                # 防止浮动出界
                u = float(np.clip(u, 0.0, 1.0))
                v = float(np.clip(v, 0.0, 1.0))

                # 双线性插值: 输出 (dst_y, dst_x)
                P = (
                    (1 - u) * (1 - v) * TL +
                    u       * (1 - v) * TR +
                    u       * v       * BR +
                    (1 - u) * v       * BL
                )

                dst_grid[j, i] = P   # P 也是 (dst_y, dst_x)

        return dst_grid

    src_corners_xy = np.float32([
        [0, 0],
        [cut_W-1, 0],
        [cut_W-1, cut_H-1],
        [0, cut_H-1],
    ])
    corners_xy = np.float32([[w, h] for h, w in corners])
    H = cv2.getPerspectiveTransform(src_corners_xy, corners_xy)
    
    def build_dst_grid_from_homography(ori_grid_hw, H):
        """
        ori_grid_hw[j,i] = (h, w)
        H: 3x3 单应矩阵
        return dst_grid_hw[j,i] = (dst_h, dst_w)
        """
        ny_plus, nx_plus, _ = ori_grid_hw.shape
        dst_grid = np.zeros_like(ori_grid_hw, dtype=np.float32)

        for j in range(ny_plus):
            for i in range(nx_plus):
                h, w = ori_grid_hw[j, i]          # (h,y) = (row), (w,x)= (col)
                pt = np.array([w, h, 1.0], dtype=np.float32)

                hp = H @ pt
                hp /= hp[2]

                dst_w, dst_h = hp[0], hp[1]
                dst_grid[j, i] = (dst_h, dst_w)

        return dst_grid

    dst_img = img.copy()
    dst_grid = build_dst_grid_from_homography(cut_grid, H)
    nh, nw = dst_grid.shape[:2]
    for j in range(nh):
        for i in range(nw):
            h, w = dst_grid[j, i]
            cv2.circle(dst_img, (int(w), int(h)), 2, (0, 0, 255), -1)  # 红点

    save_step(out_dir, "step6_dst_gray_points.jpg", dst_img)

    # step 7 通过之前检测框修复dst grid


    def build_dst_grid_by_edge_curves(
        ori_grid, 
        top_pts, bottom_pts, left_pts, right_pts
    ):
        """
        用四条边界曲线构建整个 dst_grid （Coons Patch 曲面）
        ori_grid: (ny+1, nx+1, 2)
        边界曲线需要是等长重采样过的点：
            top_pts:    (nx+1, 2)
            bottom_pts: (nx+1, 2)
            left_pts:   (ny+1, 2)
            right_pts:  (ny+1, 2)
        返回：
            dst_grid (ny+1, nx+1, 2)
        """

        ny_plus, nx_plus, _ = ori_grid.shape
        ny = ny_plus - 1
        nx = nx_plus - 1

        dst_grid = np.zeros_like(ori_grid, dtype=np.float32)

        # 四个角点（Y,X）
        TL = top_pts[0]
        TR = top_pts[-1]
        BR = bottom_pts[-1]
        BL = bottom_pts[0]

        for j in range(ny_plus):
            v = j / ny
            # C_left(v), C_right(v)
            L = left_pts[j]
            R = right_pts[j]

            for i in range(nx_plus):
                u = i / nx

                # C_top(u), C_bottom(u)
                T = top_pts[i]
                B = bottom_pts[i]

                # Bilinear corner blending term
                corners_uv = (
                    (1-u)*(1-v)*TL +
                    u*(1-v)*TR +
                    u*v*BR +
                    (1-u)*v*BL
                )

                # Coons Patch Surface
                S = (
                    (1-v)*T + v*B +
                    (1-u)*L + u*R -
                    corners_uv
                )

                dst_grid[j, i] = S

        return dst_grid

    def interpolate_x_from_polyline(poly, y_query):
        """
        poly: (N,2)  (h,w)
        输入 y → 输出 w
        """
        pts = np.asarray(poly, dtype=np.float32)
        ys = pts[:,0]   # h
        xs = pts[:,1]   # w

        if y_query <= ys[0]:
            return xs[0]
        if y_query >= ys[-1]:
            return xs[-1]

        idx = np.searchsorted(ys, y_query) - 1
        idx = max(0, min(idx, len(ys) - 2))

        y0, x0 = ys[idx],     xs[idx]
        y1, x1 = ys[idx + 1], xs[idx + 1]

        t = (y_query - y0) / (y1 - y0 + 1e-8)
        return (1 - t) * x0 + t * x1

    def interpolate_y_from_polyline(poly, x_query):
        pts = np.asarray(poly, dtype=np.float32)
        ys = pts[:,0]  # h
        xs = pts[:,1]  # w

        if x_query <= xs[0]:
            return ys[0]
        if x_query >= xs[-1]:
            return ys[-1]

        idx = np.searchsorted(xs, x_query) - 1
        idx = max(0, min(idx, len(xs)-2))

        x0, y0 = xs[idx],     ys[idx]
        x1, y1 = xs[idx + 1], ys[idx + 1]

        t = (x_query - x0) / (x1 - x0 + 1e-8)
        return (1 - t) * y0 + t * y1



    # "left": left_centerline,
    # "right": right_centerline,
    # "top": top_centerline,
    # "bottom": bottom_centerline,

    for i in range(nw):
        th, tw = dst_grid[0, i,]
        bh, bw = dst_grid[-1, i,]

        new_h_top    = interpolate_y_from_polyline(top_centerline, tw)
        new_h_bottom = interpolate_y_from_polyline(bottom_centerline, bw)

        dst_grid[0, i, 0] = new_h_top
        dst_grid[-1, i, 0] = new_h_bottom

    for i in range(nh):
        th, tw = dst_grid[i, 0,]
        bh, bw = dst_grid[i, -1,]

        new_w_l    = interpolate_x_from_polyline(left_centerline, th)
        new_w_r = interpolate_x_from_polyline(right_centerline, bh)

        dst_grid[i, 0, 1] = new_w_l
        dst_grid[i, -1, 1] = new_w_r


    def build_coons_patch_from_boundary(dst_grid):
        """
        基于四周边界构建内部网格（Coons Patch）, 最稳、最自然
        """
        H, W, _ = dst_grid.shape

        top    = dst_grid[0, :]
        bottom = dst_grid[-1, :]
        left   = dst_grid[:, 0]
        right  = dst_grid[:, -1]

        TL = top[0]
        TR = top[-1]
        BL = bottom[0]
        BR = bottom[-1]

        new_grid = np.zeros_like(dst_grid, dtype=np.float32)

        for j in range(H):
            v = j / (H - 1)
            L = left[j]
            R = right[j]

            for i in range(W):
                u = i / (W - 1)
                T = top[i]
                B = bottom[i]

                C_uv = (
                    (1-u)*(1-v)*TL +
                    u*(1-v)*TR +
                    u*v*BR +
                    (1-u)*v*BL
                )

                S = (
                    (1-v)*T + v*B +
                    (1-u)*L + u*R -
                    C_uv
                )

                new_grid[j, i] = S

        return new_grid

    opt_grid = build_coons_patch_from_boundary(dst_grid.copy())


    dst_img = img.copy()
    nh, nw = dst_grid.shape[:2]
    for j in range(nh):
        for i in range(nw):
            h, w = dst_grid[j, i]
            cv2.circle(dst_img, (int(w), int(h)), 5, (0, 0, 255), -1)  # 红点

    for j in range(nh):
        for i in range(nw):
            h, w = opt_grid[j, i]
            cv2.circle(dst_img, (int(w), int(h)), 3, (255, 0, 0), -1)  # 蓝

    save_step(out_dir, "step7_opt_dst_grid.jpg", dst_img)


    # ---------- Step 8: 投影 ----------


    # def warp_single_quad_hw(src, src_quad_hw, dst_quad_hw, out_h, out_w):
    #     """
    #     src: 原图 (H,W,3)
    #     src_quad_hw: 源四点 [(h,w),...]
    #     dst_quad_hw: 目标四点 [(h,w),...]
    #     out_h, out_w: 输出图片大小
    #     """

    #     # === 创建空白输出 ===
    #     dst_img = np.full((out_h, out_w, 3), 255, dtype=src.dtype)

    #     # ====== 将 (h,w) → (x,y) ======
    #     # OpenCV 坐标是 (x,y)
    #     def hw2xy(quad_hw):
    #         return np.float32([[w, h] for h, w in quad_hw])

    #     src_quad_xy = hw2xy(src_quad_hw)
    #     dst_quad_xy = hw2xy(dst_quad_hw)

    #     # === 单应矩阵 ===
    #     H = cv2.getPerspectiveTransform(src_quad_xy, dst_quad_xy)

    #     # === warp 整张图（简单可靠，不裁 ROI） ===
    #     warped = cv2.warpPerspective(src, H, (out_w, out_h))

    #     # === 生成 mask，防止把没覆盖区域冲掉 ===
    #     mask = np.zeros((out_h, out_w), dtype=np.uint8)
    #     cv2.fillConvexPoly(mask, dst_quad_xy.astype(np.int32), 255)

    #     # === 按 mask 贴到 dst_img ===
    #     dst_img[mask == 255] = warped[mask == 255]

    #     return dst_img


    # src_quad = [
    #     cut_grid[0,0],
    #     cut_grid[0,2],
    #     cut_grid[2,2],
    #     cut_grid[2,0],
    # ]

    # dst_quad = [
    #     dst_grid[0,0],
    #     dst_grid[0,2],
    #     dst_grid[2,2],
    #     dst_grid[2,0],
    # ]

    # H, W = img.shape[:2]
    # res = warp_single_quad_hw(cut_original_img, src_quad, dst_quad, H, W)
    # cv2.imwrite("warp_test.png", res)


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



    def warp_single_quad_hw_local(src, src_quad_hw, dst_quad_hw, dst_img, pad=2, bg_value=255):
        """
        src:         原图 (H,W,3) 或 (H,W)
        src_quad_hw: 源四点 [(h,w), ...]
        dst_quad_hw: 目标四点 [(h,w), ...]
        dst_img:     大的输出图 (会原地修改)
        pad:         在 dst / src ROI 上各扩一圈像素，避免 1px 的缝
        bg_value:    warp 时外部填充颜色（白底一般 255）
        """

        src_H, src_W = src.shape[:2]
        dst_H, dst_W = dst_img.shape[:2]

        # ---- 1. (h,w) -> (x,y) ----
        src_xy = np.float32([[w, h] for h, w in src_quad_hw])
        dst_xy = np.float32([[w, h] for h, w in dst_quad_hw])

        # ---- 2. 源 ROI + pad ----
        src_hs = [p[0] for p in src_quad_hw]
        src_ws = [p[1] for p in src_quad_hw]
        sy0 = max(0, int(np.floor(min(src_hs))) - pad)
        sy1 = min(src_H - 1, int(np.ceil(max(src_hs))) + pad)
        sx0 = max(0, int(np.floor(min(src_ws))) - pad)
        sx1 = min(src_W - 1, int(np.ceil(max(src_ws))) + pad)

        src_patch = src[sy0:sy1+1, sx0:sx1+1]
        if src_patch.size == 0:
            return

        src_xy_local = src_xy.copy()
        src_xy_local[:, 0] -= sx0
        src_xy_local[:, 1] -= sy0

        # ---- 3. 目标 ROI + pad ----
        dst_hs = [p[0] for p in dst_quad_hw]
        dst_ws = [p[1] for p in dst_quad_hw]
        dy0 = max(0, int(np.floor(min(dst_hs))) - pad)
        dy1 = min(dst_H - 1, int(np.ceil(max(dst_hs))) + pad)
        dx0 = max(0, int(np.floor(min(dst_ws))) - pad)
        dx1 = min(dst_W - 1, int(np.ceil(max(dst_ws))) + pad)

        roi_h = dy1 - dy0 + 1
        roi_w = dx1 - dx0 + 1
        if roi_h <= 0 or roi_w <= 0:
            return

        dst_xy_local = dst_xy.copy()
        dst_xy_local[:, 0] -= dx0
        dst_xy_local[:, 1] -= dy0

        # ---- 4. 单应矩阵（在 local 坐标系里）----
        H = cv2.getPerspectiveTransform(src_xy_local, dst_xy_local)

        # ---- 5. warp 源 patch -> 目标 ROI ----
        warped = cv2.warpPerspective(
            src_patch,
            H,
            (roi_w, roi_h),
            flags=cv2.INTER_NEAREST,                  # 关键：最近邻，避免灰边
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_value
        )

        # ---- 6. 生成 mask，限定四边形区域 ----
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_xy_local.astype(np.int32), 255)

        return warped, mask, (dy0, dx0), (roi_h, roi_w)



    def warp_mesh_piecewise_hw(src, ori_grid, dst_grid, out_h, out_w):
        """
        src:      原图
        ori_grid: (ny+1, nx+1, 2)  每点 (h,w)
        dst_grid: (ny+1, nx+1, 2)  每点 (h,w)
        out_h, out_w 输出分辨率

        返回：最终 warp 完成的大图（无缝）
        """

        ny_plus, nx_plus, _ = ori_grid.shape
        ny = ny_plus - 1
        nx = nx_plus - 1

        dst_img = np.full((out_h, out_w, 3), 255, dtype=src.dtype)

        # 遍历每个格子
        for j in range(ny):
            for i in range(nx):
                # 源四点
                src_quad = [
                    ori_grid[j,   i],
                    ori_grid[j,   i+1],
                    ori_grid[j+1, i+1],
                    ori_grid[j+1, i],
                ]

                # 目标四点
                dst_quad = [
                    dst_grid[j,   i],
                    dst_grid[j,   i+1],
                    dst_grid[j+1, i+1],
                    dst_grid[j+1, i],
                ]

                # warp_single_quad_hw(src, src_quad, dst_quad,out_h, out_w, dst_img)
                warped, mask,  (dy0, dx0), (roi_h, roi_w) = warp_single_quad_hw_local(src, src_quad, dst_quad, dst_img)

                # 贴图
                region = dst_img[dy0:dy0+roi_h, dx0:dx0+roi_w]
                region[mask == 255] = warped[mask == 255]

        return dst_img




    def overlay_binary_mask_on_image(base, warp, alpha=0.5):
        """
        base_img:       拍照图 BGR
        warped_binary:  warp 后的 cut_original_img（0/255）
        alpha:          红色透明度
        """

        # 三通道判断黑色
        mask_black = np.all(warp == 0, axis=-1)     # shape (H,W)
        red_warp = warp.copy()
        red_warp[mask_black] = (0, 0, 255)   # 红色 BGR

        out = base*0.5  + red_warp*0.5
        return out.astype(np.uint8)

    H, W = img.shape[:2]
    final = warp_mesh_piecewise_hw(
        src=cut_original_img,
        ori_grid=cut_grid,    # (ny+1, nx+1, 2) (h,w)
        dst_grid=dst_grid,    # (ny+1, nx+1, 2) (h,w)
        out_h=H,
        out_w=W
    )

    save_step(out_dir, "step8_girdsample_result.jpg", final)
    
    overlay = overlay_binary_mask_on_image(img, final, alpha=0.5)
    save_step(out_dir, "step8_mix_result.jpg", overlay)


    # step 9 微调grid


    def visualize_binary(mask, save_path):
        """
        mask: shape (H,W), 值为 0/1
        """
        vis = (mask * 255).astype(np.uint8)     # 0 → 0(黑)，1 → 255(白)
        cv2.imwrite(save_path, vis)


    # camera_bin_img 二值化之后图
    # cut_original_img 原图

    mask_black = np.all(cut_original_img == 0, axis=-1)  # shape (H,W)

    ori_binary = np.zeros(mask_black.shape, dtype=np.uint8)
    ori_binary[mask_black] = 1

    camera_bin_img = clean_bin_img // 255


    visualize_binary(ori_binary, os.path.join(out_dir, "step9_ori_binary.jpg"))
    visualize_binary(camera_bin_img, os.path.join(out_dir, "step9_camera_binary.jpg"))


    def warp_mesh_binary_hw(ori_binary, ori_grid, dst_grid, out_h, out_w):
        H, W = out_h, out_w
        dst_bin = np.zeros((H, W), dtype=np.uint8)

        ny = ori_grid.shape[0] - 1
        nx = ori_grid.shape[1] - 1

        for j in range(ny):
            for i in range(nx):
                src_quad = [
                    ori_grid[j,   i],
                    ori_grid[j,   i+1],
                    ori_grid[j+1, i+1],
                    ori_grid[j+1, i],
                ]

                dst_quad = [
                    dst_grid[j,   i],
                    dst_grid[j,   i+1],
                    dst_grid[j+1, i+1],
                    dst_grid[j+1, i],
                ]

                # warp 单块
                src_quad_xy = np.float32([[w, h] for h, w in src_quad])
                dst_quad_xy = np.float32([[w, h] for h, w in dst_quad])
                H_mat = cv2.getPerspectiveTransform(src_quad_xy, dst_quad_xy)

                warped = cv2.warpPerspective(ori_binary.astype(np.uint8), H_mat, (W, H))

                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_quad_xy.astype(np.int32), 1)

                dst_bin[mask == 1] = warped[mask == 1]

        return dst_bin



    def opt_h_grid_point(j, i,
                            ori_binary, 
                            ori_grid, dst_grid,
                            camera_bin_img,
                            delta=1.0):

        h0, w0 = dst_grid[j, i]

        #  *   *   *
        #  *   ji  *
        #  *   *   *

        # 左上
        src_quad_lt = [ori_grid[j-1, i-1],ori_grid[j-1, i],ori_grid[j, i],ori_grid[j, i-1]]
        src_quad_lb = [ori_grid[j,   i-1],ori_grid[j,   i],ori_grid[j+1, i+1],ori_grid[j+1, i]]
        src_quad_rt = [ori_grid[j-1, i],ori_grid[j-1, i+1],ori_grid[j, i+1],ori_grid[j, i]]
        # src_quad_rb = [ori_grid[j,   i],ori_grid[j,   i+1],ori_grid[j+1, i+1],ori_grid[j+1, i]]


        dst_quad_lt = [dst_grid[j-1, i-1],dst_grid[j-1, i],dst_grid[j, i],dst_grid[j, i-1]]
        # dst_quad_lb = [dst_grid[j,   i-1],dst_grid[j,   i],dst_grid[j+1, i+1],dst_grid[j+1, i]]
        dst_quad_rt = [dst_grid[j-1, i],dst_grid[j-1, i+1],dst_grid[j, i+1],dst_grid[j, i]]
        # dst_quad_rb = [dst_grid[j,   i],dst_grid[j,   i+1],dst_grid[j+1, i+1],dst_grid[j+1, i]]


        def cal_iou(ori_binary, src_quad_lt, dst_quad_lt, dst_img):
            warped, mask,  (dy0, dx0), (roi_h, roi_w) = warp_single_quad_hw_local(ori_binary*255, src_quad_lt, dst_quad_lt, dst_img)
            camera_slice = camera_bin_img[dy0:dy0+roi_h, dx0:dx0+roi_w]
            img_iou = camera_slice * (warped // 255)
            img_iou[mask == 255] = img_iou[mask == 255]

            ov = np.sum(img_iou)/(roi_h * roi_w)
            return ov

        # 移动前iou
        ori_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad_lt, dst_img)
        ori_iou_rt = cal_iou(ori_binary, src_quad_rt, dst_quad_rt, dst_img)
        ori_iou = (ori_iou_lt + ori_iou_rt) / 2

        if ori_iou < 0.04:
            print(j,i, ori_iou)
            return False


        # 上移动
        dst_quad = dst_quad_lt.copy()
        dst_quad[2][0] -= delta
        t_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        dst_quad = dst_quad_rt.copy()
        dst_quad[3][0] -= delta
        t_iou_rt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        t_iou = (t_iou_lt + t_iou_rt) / 2

        # 下移动
        dst_quad = dst_quad_lt.copy()
        dst_quad[2][0] += delta
        b_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        dst_quad = dst_quad_rt.copy()
        dst_quad[3][0] += delta
        b_iou_rt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        b_iou = (b_iou_lt + b_iou_rt) / 2


        if t_iou >= ori_iou and t_iou >= b_iou:
            # 向上移动
            dh = -delta
        elif b_iou >= ori_iou and b_iou >= t_iou:
            # 向下移动 
            dh = delta
        else :
            dh = 0

        dst_grid[j, i] = (h0 + dh, w0)

        return dh != 0

    def opt_w_grid_point(j, i,
                            ori_binary, 
                            ori_grid, dst_grid,
                            camera_bin_img,
                            delta=1.0):

        h0, w0 = dst_grid[j, i]

        #  *   *   *
        #  *   ji  *
        #  *   *   *

        # 左上
        src_quad_lt = [ori_grid[j-1, i-1],ori_grid[j-1, i],ori_grid[j, i],ori_grid[j, i-1]]
        src_quad_lb = [ori_grid[j,   i-1],ori_grid[j,   i],ori_grid[j+1, i+1],ori_grid[j+1, i]]
        src_quad_rt = [ori_grid[j-1, i],ori_grid[j-1, i+1],ori_grid[j, i+1],ori_grid[j, i]]
        # src_quad_rb = [ori_grid[j,   i],ori_grid[j,   i+1],ori_grid[j+1, i+1],ori_grid[j+1, i]]


        dst_quad_lt = [dst_grid[j-1, i-1],dst_grid[j-1, i],dst_grid[j, i],dst_grid[j, i-1]]
        # dst_quad_lb = [dst_grid[j,   i-1],dst_grid[j,   i],dst_grid[j+1, i+1],dst_grid[j+1, i]]
        dst_quad_rt = [dst_grid[j-1, i],dst_grid[j-1, i+1],dst_grid[j, i+1],dst_grid[j, i]]
        # dst_quad_rb = [dst_grid[j,   i],dst_grid[j,   i+1],dst_grid[j+1, i+1],dst_grid[j+1, i]]


        def cal_iou(ori_binary, src_quad_lt, dst_quad_lt, dst_img):
            warped, mask,  (dy0, dx0), (roi_h, roi_w) = warp_single_quad_hw_local(ori_binary*255, src_quad_lt, dst_quad_lt, dst_img)
            camera_slice = camera_bin_img[dy0:dy0+roi_h, dx0:dx0+roi_w]
            img_iou = camera_slice * (warped // 255)
            img_iou[mask == 255] = img_iou[mask == 255]

            ov = np.sum(img_iou)/(roi_h * roi_w)
            return ov

        # 移动前iou
        ori_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad_lt, dst_img)
        ori_iou_rt = cal_iou(ori_binary, src_quad_rt, dst_quad_rt, dst_img)
        ori_iou = (ori_iou_lt + ori_iou_rt) / 2

        if ori_iou < 0.04:
            print(j,i, ori_iou)
            return False


        # 左移动
        dst_quad = dst_quad_lt.copy()
        dst_quad[2][1] -= delta
        l_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        dst_quad = dst_quad_rt.copy()
        dst_quad[3][1] -= delta
        l_iou_rt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        l_iou = (l_iou_lt + l_iou_rt) / 2

        # 右移动
        dst_quad = dst_quad_lt.copy()
        dst_quad[2][1] += delta
        r_iou_lt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        dst_quad = dst_quad_rt.copy()
        dst_quad[3][1] += delta
        r_iou_rt = cal_iou(ori_binary, src_quad_lt, dst_quad, dst_img)
        r_iou = (r_iou_lt + r_iou_rt) / 2


        if l_iou >= ori_iou and r_iou >= r_iou:
            # 向上移动
            dw = -delta
        elif r_iou >= ori_iou and r_iou >= l_iou:
            # 向下移动 
            dw = delta
        else :
            dw = 0

        dst_grid[j, i] = (h0, w0 + dw)

        return dw != 0

    def local_ncc_shift(camera, warped, max_shift=3):
        """
        camera, warped: 2D numpy array (float32 or uint8)
        max_shift: 搜索范围，例如 3 表示 [-3,3]
        
        返回: (dh, dw)
        """
        camera = camera.astype(np.float32)
        warped = warped.astype(np.float32)

        H, W = camera.shape
        best_ncc = -1.0
        best_shift = (0, 0)

        # 搜索 小范围位移（整数）
        for dh in range(-max_shift, max_shift+1):
            for dw in range(-max_shift, max_shift+1):

                # 对齐两个 patch 的重叠区域
                y0a = max(0,  dh)
                y1a = min(H, H + dh)
                x0a = max(0,  dw)
                x1a = min(W, W + dw)

                y0b = max(0, -dh)
                y1b = min(H, H - dh)
                x0b = max(0, -dw)
                x1b = min(W, W - dw)

                A = camera[y0a:y1a, x0a:x1a]
                B = warped[y0b:y1b, x0b:x1b]

                if A.size < 50:  # 太小无意义
                    continue

                # 归一化相关
                A_mean = A.mean()
                B_mean = B.mean()

                num = np.sum((A - A_mean) * (B - B_mean))
                den = np.sqrt(np.sum((A - A_mean)**2) * np.sum((B - B_mean)**2) + 1e-6)

                ncc = num / den

                if ncc > best_ncc:
                    best_ncc = ncc
                    best_shift = (dh, dw)

        return best_shift, best_ncc


    def diamond_search_shift(ref, tgt, max_search=2):
        """
        ref: warped patch (binary 0/1 or 0/255)
        tgt: camera patch (binary 0/1 or 0/255)
        max_search: 搜索范围（像素）

        返回: (dw, dh)
        """

        # convert to 0/1
        if ref.max() > 1: ref = ref / 255.0
        if tgt.max() > 1: tgt = tgt / 255.0

        H, W = ref.shape

        # cost function: SAD
        def sad(dh, dw):
            # shift tgt by (dh,dw)
            y0 = max(0, dh)
            y1 = min(H, H + dh)
            x0 = max(0, dw)
            x1 = min(W, W + dw)

            ys0 = max(0, -dh)
            ys1 = ys0 + (y1 - y0)
            xs0 = max(0, -dw)
            xs1 = xs0 + (x1 - x0)

            if y1 <= y0 or x1 <= x0:
                return 1e9

            A = ref[y0:y1, x0:x1]
            B = tgt[ys0:ys1, xs0:xs1]

            return np.sum(np.abs(A - B))

        # Large Diamond Search Pattern (LDSP)
        LDSP = [
            ( 0,  0),
            (-1,  0), (1,  0),
            ( 0, -1), (0, 1),
        ]

        # Small DSP
        SDSP = [
            ( 0, 0),
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
        ]

        best_dh = 0
        best_dw = 0
        best_cost = sad(0, 0)

        step = max_search

        # --- LDSP coarse search ---
        while step >= 1:
            improved = False

            for dh0, dw0 in LDSP:
                dh = best_dh + dh0 * step
                dw = best_dw + dw0 * step

                # limit search
                if abs(dh) > max_search or abs(dw) > max_search:
                    continue

                cost = sad(dh, dw)
                if cost < best_cost:
                    best_cost = cost
                    best_dh = dh
                    best_dw = dw
                    improved = True

            if not improved:
                step //= 2  # 缩小步长

        # --- SDSP fine search ---
        for dh0, dw0 in SDSP:
            dh = best_dh + dh0
            dw = best_dw + dw0

            if abs(dh) > max_search or abs(dw) > max_search:
                continue

            cost = sad(dh, dw)
            if cost < best_cost:
                best_cost = cost
                best_dh = dh
                best_dw = dw

        return best_dw, best_dh  # (dx, dy) 与 phaseCorrelate 一致方向


    def cv_grid_point(j, i, ori_binary, 
                            ori_grid, dst_grid,
                            camera_bin_img,
                            delta=1.0):

        # 左上
        src_quad_lt = [ori_grid[j-1, i-1],ori_grid[j-1, i],ori_grid[j, i],ori_grid[j, i-1]]
        dst_quad_lt = [dst_grid[j-1, i-1],dst_grid[j-1, i],dst_grid[j, i],dst_grid[j, i-1]]


        warped, mask,  (dy0, dx0), (roi_h, roi_w) = warp_single_quad_hw_local(ori_binary*255, src_quad_lt, dst_quad_lt, dst_img)
        camera_slice = camera_bin_img[dy0:dy0+roi_h, dx0:dx0+roi_w]
        mask_iou = np.sum(camera_slice /255.) / (roi_h * roi_w)

        if mask_iou < 0.0001:
            return (0, 0), None
        camera_slice = camera_slice.astype(np.float32)
        warped = warped.astype(np.float32)

        shift, response = cv2.phaseCorrelate(camera_slice, warped)
        # dx: 往右正方向
        # dy: 往下正方向
        dw, dh = shift

        dw = np.clip(dw, -delta, delta)
        dh = np.clip(dh, -delta, delta)

        dst_grid[j, i] -= (dh, dw)

        print(f" j, i :{j, i}, mask_iou {mask_iou} shift:{shift}")


        return shift, response


    def opt_1_grid_iterative(ori_binary, camera_bin_img, ori_grid, dst_grid,
                          delta=1.0,
                          iters=3):
        ny_plus, nx_plus, _ = dst_grid.shape
        changed = False
        cmp_a = dst_grid.copy()

        for n in range(iters):
            for j in range(1, ny_plus - 1):
                for i in range(1, nx_plus - 1):

                    cv_grid_point(
                        j, i,
                        ori_binary, ori_grid, dst_grid,
                        camera_bin_img,
                        delta
                    )
            
            dst_img = overlay.copy()
            nh, nw = cmp_a.shape[:2]
            for j in range(nh):
                for i in range(nw):
                    h, w = cmp_a[j, i]
                    cv2.circle(dst_img, (int(w), int(h)), 5, (0, 0, 255), -1)  # 红点

            for j in range(nh):
                for i in range(nw):
                    h, w = dst_grid[j, i]
                    cv2.circle(dst_img, (int(w), int(h)), 3, (255, 0, 0), -1)  # 蓝

            save_step(out_dir, f"step9_opt1_iter_{n}_opt_dst_grid.jpg", dst_img)



    def opt_2_grid_iterative(ori_binary, camera_bin_img, ori_grid, dst_grid,
                          delta=1.0,
                          iters=3):
        ny_plus, nx_plus, _ = dst_grid.shape
        changed = False
        cmp_a = dst_grid.copy()

        for n in range(iters):
            for j in range(1, ny_plus - 1):
                for i in range(1, nx_plus - 1):

                    opt_h_grid_point(
                        j, i,
                        ori_binary, ori_grid, dst_grid,
                        camera_bin_img,
                        delta
                    )
                    opt_w_grid_point(
                        j, i,
                        ori_binary, ori_grid, dst_grid,
                        camera_bin_img,
                        delta
                    )

            
            dst_img = overlay.copy()
            nh, nw = cmp_a.shape[:2]
            for j in range(nh):
                for i in range(nw):
                    h, w = cmp_a[j, i]
                    cv2.circle(dst_img, (int(w), int(h)), 5, (0, 0, 255), -1)  # 红点

            for j in range(nh):
                for i in range(nw):
                    h, w = dst_grid[j, i]
                    cv2.circle(dst_img, (int(w), int(h)), 3, (255, 0, 0), -1)  # 蓝

            save_step(out_dir, f"step9_opt2_iter_{n}_opt_dst_grid.jpg", dst_img)


        return changed

    opt_1_grid_iterative(
        ori_binary=ori_binary,
        camera_bin_img=camera_bin_img,
        ori_grid=cut_grid,
        dst_grid=opt_grid, 
        delta=2,
        iters=5
    )

    # opt_2_grid_iterative(
    #     ori_binary=ori_binary,
    #     camera_bin_img=camera_bin_img,
    #     ori_grid=cut_grid,
    #     dst_grid=opt_grid, 
    #     delta=1,
    #     iters=10
    # )


    H, W = img.shape[:2]
    opt_final = warp_mesh_piecewise_hw(
        src=cut_original_img,
        ori_grid=cut_grid,    # (ny+1, nx+1, 2) (h,w)
        dst_grid=opt_grid,    # (ny+1, nx+1, 2) (h,w)
        out_h=H,
        out_w=W
    )
    
    overlay_9 = overlay_binary_mask_on_image(img, opt_final, alpha=0.5)
    save_step(out_dir, "step9_opt_grid.jpg", overlay_9)


if __name__ == "__main__":
    # img_path = "/home/glq/sp_tools/test_data/IMG20251129154449.jpg"
    # original_path = "/home/glq/sp_tools/test_data/0a1a7e38-afee-433b-a16d-8f4f46ecdf07-2.png"

    img_path = "/home/glq/sp_tools/test_data/IMG20251129155059.jpg"
    original_path = "/home/glq/sp_tools/test_data/0a1a7e38-afee-433b-a16d-8f4f46ecdf07-13.png"
    out_dir = "/home/glq/sp_tools/debug_region_edge"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)   # 相当于 rm -rf out_dir
    os.makedirs(out_dir)     


    detect_corners_region_edge(
        img_path,
        original_path,
        out_dir=out_dir,
        inner_offset=30.0,  # 可以根据黑框厚度微调：20~40 像素
    )
