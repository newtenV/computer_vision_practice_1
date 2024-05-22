import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_sgbm(image_left_path, image_right_path):
    # 读取左图和右图
    imgL = cv2.imread(image_left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(image_right_path, cv2.IMREAD_GRAYSCALE)

    # 创建立体匹配对象 (SGBM)
    window_size = 5
    min_disp = -1
    num_disp = 16 * 5
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    # 计算视差图
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # 归一化视差图，便于显示
    disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # 显示结果
    plt.subplot(131), plt.imshow(imgL, cmap='gray'), plt.title('Left Image')
    plt.subplot(132), plt.imshow(imgR, cmap='gray'), plt.title('Right Image')
    plt.subplot(133), plt.imshow(disparity_normalized, cmap='jet'), plt.title('Disparity Map')
    plt.show()

    return disparity



image_left_path = 'p1.JPG'
image_right_path = 'p2.JPG'
disparity_map_sgbm = compute_disparity_sgbm(image_left_path, image_right_path)


