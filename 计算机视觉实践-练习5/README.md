**图像视差匹配**

**实现方法：**

1. 创建立体匹配对象 (SGBM)：调用 `cv2.StereoSGBM_create()` 函数创建立体匹配对象，并设置相关参数，包括最小视差、视差范围、窗口大小等。
2. 计算视差图：利用创建的立体匹配对象的 `compute()` 函数计算左右图像之间的视差图。

**结果：**

![RESULT](https://github.com/newtenV/computer_vision_practice_1/blob/main/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5-%E7%BB%83%E4%B9%A05/result.png)
