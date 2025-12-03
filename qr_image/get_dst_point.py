import os
import cv2
import numpy as np

class GetDstPoint:
    def __init__(self, info_path):
        info = np.load(info_path, allow_pickle=True)
        self.pad = info.item()['pad']
        self.labelname = info.item()['labelname'] # 原图名
        self.border_size = info.item()['border_size']
        self.M_label2src = info.item()['M_label2src']
        self.M_src2label = info.item()['M_src2label']

    def get_dst_point(self, x, y):
        # 源坐标点 (x, y)
        src_point = np.array([x+self.border_size, y+self.pad+self.border_size], dtype=np.float32)

        # 将源坐标转换为齐次坐标 [x, y, 1]
        src_point_homogeneous = np.array([src_point[0], src_point[1], 1], dtype=np.float32)

        # 执行矩阵乘法得到齐次目标坐标 [x'', y'', w]
        dst_point_homogeneous = self.M_label2src @ src_point_homogeneous

        # 归一化得到最终的二维目标坐标 [x', y']
        w = dst_point_homogeneous[2]
        dst_point = np.array([dst_point_homogeneous[0] / w, dst_point_homogeneous[1] / w], dtype=np.float32)

        # print(f"源坐标: {src_point}, 目标坐标: {dst_point}")
        return dst_point
    
def readTxtToYolo(txtpath):
    """
    从给定的文本路径读取标签文件内容，并转换为 YOLO 格式的边界框列表。
    如果标签文件不存在，则打印提示信息并返回空列表。
    """
    retdata = []
    try:
        with open(txtpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                yolo_box = map(float, line)
                retdata.append(list(yolo_box))
            f.close()
    except FileNotFoundError:
        print(f"Label file {txtpath} not found.")
    return retdata  

if __name__ == "__main__":
    label_dir = "/data/fl/code/datasets/beam_data/beam_labels"
    photo_dir = "/data/fl/sp_tools/2025112001"
    scr_img_dir = "/data/xml_data/images"
    new_label_dir = "/data/fl/code/datasets/beam_data/beam_labels_new"
    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)

    for dname in os.listdir(photo_dir):
        info_path = os.path.join(photo_dir, dname, dname + "_info.npy")
        gdp = GetDstPoint(info_path)
        label_path = os.path.join(label_dir, gdp.labelname[:-4] + ".txt")
        img_path = os.path.join(photo_dir, dname, "ordered.png")
        img = cv2.imread(img_path)
        scr_img_path = os.path.join(scr_img_dir, gdp.labelname)
        scr_img = cv2.imread(scr_img_path)
        h, w, _ = scr_img.shape
        bboxes = readTxtToYolo(label_path)
        for bbox in bboxes:
            x1 = bbox[5] * w
            y1 = bbox[6] * h
            x2 = bbox[7] * w
            y2 = bbox[8] * h
            dst_point1 = gdp.get_dst_point(x1, y1)
            dst_point2 = gdp.get_dst_point(x2, y2)
            cv2.circle(img, (int(dst_point1[0]), int(dst_point1[1])), 3, (0, 255, 0), -1)
            cv2.circle(img, (int(dst_point2[0]), int(dst_point2[1])), 3, (255, 0, 0), -1)
        
        cv2.imwrite(os.path.join(new_label_dir, gdp.labelname[:-4] + ".png"), img)
        pass