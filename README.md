# rgdb_slam_learning
learning rgdb slam. based on rgbd-slam-tutorial-gx(gaoxiang) https://github.com/gaoxiang12/rgbd-slam-tutorial-gx.git

#Changes
1.图像特征提取采用ORBSLAM2中的算法
2.添加TUM数据集读取接口
3.添加各模块用时评估
4.完善整个SLAM系统

#Note
1.更改了TUM的**associate.py**脚本，请**复制到TUM数据集文件夹**运行
```bash
 ./associate.py rgb.txt depth.txt --forslam > images.txt
```
生成本项目所用脚本

#TODO
1.模块封装
2.多线程
