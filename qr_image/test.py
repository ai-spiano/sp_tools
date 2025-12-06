
import cv2
import numpy as np
import os


def save(name, img, out_dir):
    cv2.imwrite(os.path.join(out_dir, name), img)


def detect_corners_super_stable(path, out_dir="debug_output", inner_offset=30):
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 0: load ---
    img = cv2.imread(path)
    save("step0_original.jpg", img, out_dir)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: 自适应二值化（最稳定） ---
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,  # 大窗口适应不均匀光照
        C=5
    )
    save("step1_binary.jpg", bin_img, out_dir)

    # --- Step 2: 形态学闭操作，让黑框闭合 ---
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    save("step2_closed.jpg", closed, out_dir)

    # --- Step 3: 获取最大轮廓（黑框） ---
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    contour_vis = img.copy()
    cv2.drawContours(contour_vis, [cnt], -1, (0, 0, 255), 4)
    save("step3_contour.jpg", contour_vis, out_dir)

    # --- Step 4: 将轮廓点分类为四个方向 ---
    pts = cnt.reshape(-1, 2)
    H, W = gray.shape

    # 上下左右分类（按与图像边界的距离分类）
    top_pts    = pts[pts[:,1] < H * 0.3]
    bottom_pts = pts[pts[:,1] > H * 0.7]
    left_pts   = pts[pts[:,0] < W * 0.3]
    right_pts  = pts[pts[:,0] > W * 0.7]

    clusters_vis = img.copy()
    for p in top_pts:    cv2.circle(clusters_vis, (p[0], p[1]), 2, (0, 0, 255), -1)   # red
    for p in bottom_pts: cv2.circle(clusters_vis, (p[0], p[1]), 2, (255, 0, 0), -1)   # blue
    for p in left_pts:   cv2.circle(clusters_vis, (p[0], p[1]), 2, (0, 255, 0), -1)   # green
    for p in right_pts:  cv2.circle(clusters_vis, (p[0], p[1]), 2, (0, 255, 255), -1) # yellow
    save("step4_clusters.jpg", clusters_vis, out_dir)

    # --- Step 5: 最小二乘拟合四条直线 ---
    def fit_line(pts):
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        return vx, vy, x0, y0

    L_top    = fit_line(top_pts)
    L_bottom = fit_line(bottom_pts)
    L_left   = fit_line(left_pts)
    L_right  = fit_line(right_pts)

    def draw_line(vis, L, color):
        vx, vy, x0, y0 = L
        h, w = vis.shape[:2]
        x1 = int(x0 - vx * 2000)
        y1 = int(y0 - vy * 2000)
        x2 = int(x0 + vx * 2000)
        y2 = int(y0 + vy * 2000)
        cv2.line(vis, (x1, y1), (x2, y2), color, 4)

    line_vis = img.copy()
    draw_line(line_vis, L_top,    (0, 0, 255))
    draw_line(line_vis, L_bottom, (255, 0, 0))
    draw_line(line_vis, L_left,   (0, 255, 0))
    draw_line(line_vis, L_right,  (0, 255, 255))
    save("step5_fitted_lines.jpg", line_vis, out_dir)

    # --- Step 6: 求交点 ---
    def intersect(L1, L2):
        vx1, vy1, x1, y1 = L1
        vx2, vy2, x2, y2 = L2
        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        b = np.array([x2 - x1, y2 - y1])
        t = np.linalg.solve(A, b)
        return np.array([x1 + vx1 * t[0], y1 + vy1 * t[0]])

    P1 = intersect(L_top, L_left)
    P2 = intersect(L_top, L_right)
    P3 = intersect(L_bottom, L_right)
    P4 = intersect(L_bottom, L_left)

    outer = np.array([P1, P2, P3, P4], dtype=np.float32)

    corner_vis = img.copy()
    for p in outer:
        cv2.circle(corner_vis, (int(p[0]), int(p[1])), 12, (0, 0, 255), -1)
    save("step6_outer_corners.jpg", corner_vis, out_dir)

    # --- Step 7: 内缩得到内角 ---
    center = np.mean(outer, axis=0)
    inner = []
    for p in outer:
        v = p - center
        v = v / np.linalg.norm(v)
        inner.append(p - inner_offset * v)

    inner = np.array(inner, dtype=np.float32)

    inner_vis = img.copy()
    for p in inner:
        cv2.circle(inner_vis, (int(p[0]), int(p[1])), 12, (255, 0, 0), -1)
    cv2.polylines(inner_vis, [inner.astype(int)], True, (255, 0, 0), 4)
    save("step7_inner_corners.jpg", inner_vis, out_dir)

    return outer, inner

# --------------------
# 使用示例：
outer, inner = detect_corners_super_stable(
    "/home/glq/sp_tools/test_data/IMG20251129154449.jpg",
    out_dir="/home/glq/sp_tools/debug_border",
    inner_offset=30
)