import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_homography(image1_path, image2_path):
    # 读取图像
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 检测SIFT特征点和描述符
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 按照距离排序匹配
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配点
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 应用单应性变换
    height, width = img2.shape
    img1_transformed = cv2.warpPerspective(img1, H, (width, height))

    # 显示结果
    plt.subplot(131), plt.imshow(img1, cmap='gray'), plt.title('Image 1')
    plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title('Image 2')
    plt.subplot(133), plt.imshow(img1_transformed, cmap='gray'), plt.title('Transformed Image 1')

    plt.show()

    return H


image1_path = 'p2.JPG'
image2_path = 'p1.JPG'
H = compute_homography(image1_path, image2_path)
print("Computed Homography Matrix:")
print(H)
