#### 单应性变换

##### 方法：

1. 检测 SIFT 特征点和描述符：利用 `cv2.SIFT_create()` 创建 SIFT 对象，然后对图像进行特征点检测和描述符计算。
2. 匹配特征点：使用暴力匹配器 `cv2.BFMatcher()` 对两幅图像的特征描述符进行匹配。
3. 计算单应性矩阵：利用匹配点对，使用 `cv2.findHomography()` 函数计算单应性矩阵，这里采用了 RANSAC 算法进行鲁棒性估计。
4. 应用单应性变换：利用 `cv2.warpPerspective()` 函数将第一幅图像进行单应性变换，得到在第二幅图像坐标系下的图像。

##### 结果：

![result](https://github.com/newtenV/computer_vision_practice_1/blob/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A04/result.png)
