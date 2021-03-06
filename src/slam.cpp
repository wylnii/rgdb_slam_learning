/*************************************************************************
	> File Name: rgbd-slam-tutorial-gx/part V/src/visualOdometry.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
	> Created Time: 2015年08月15日 星期六 15时35分42秒
    * add g2o slam end to visual odometry
    * add keyframe and simple loop closure
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <boost/timer.hpp>

// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

// 估计一个运动的大小
double normofTransform(cv::Mat rvec, cv::Mat tvec);

// 检测两个帧，结果定义
enum CHECK_RESULT {
    NOT_MATCHED = 0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME
};

// 函数声明
CHECK_RESULT checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops = false);

// 检测近距离的回环
bool checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);

// 随机检测回环
bool checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);

int main(int argc, char **argv)
{
    // 前面部分和vo是一样的
    ParameterReader pd;

    long startIndex = strtol(pd.getData("start_index").c_str(), NULL, 10);
    int endIndex = atoi(pd.getData("end_index").c_str());

    vector<string> rgbList, depthList;
    bool rlt = getImageListFromConfigFile(rgbList, depthList, pd);
    if (!rlt)
        return -1;

    endIndex = std::min(endIndex, (int) rgbList.size());
    // 所有的关键帧都放在了这里
    vector<FRAME> keyframes;
    // initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex; // 当前索引为currIndex
    FRAME currFrame = readFrame(currIndex, rgbList, depthList, pd); // 上一帧数据

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    boost::timer t;
    computeKeyPointsAndDesp(currFrame, detector, descriptor);
    currFrame.timeUsage.orb = t.elapsed();

    PointCloud::Ptr cloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);

    /*******************************
    // 新增:有关g2o的初始化
    *******************************/
    // 初始化求解器
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
    globalOptimizer.setAlgorithm(solver);
    // 不要输出调试信息
    globalOptimizer.setVerbose(false);

    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); //估计为单位矩阵
    v->setFixed(true); //第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);

    double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure = pd.getData("check_loop_closure") == string("yes");


    PointCloud::Ptr map(new PointCloud()); //全局地图
    PointCloud::Ptr tmp(new PointCloud());

    pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
    pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0); //4m以上就不要了

    double gridsize = atof(pd.getData("voxel_grid").c_str()); //分辨图可以在parameters.txt里调
    voxel.setLeafSize(gridsize, gridsize, gridsize);

    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++)
    {
        cout << RESET"Reading files : " << currIndex << " (" << rgbList[currIndex] << ")" << endl;
        FRAME currFrame = readFrame(currIndex, rgbList, depthList, pd); // 读取currFrame

        t.restart();
        computeKeyPointsAndDesp(currFrame, detector, descriptor); //提取特征
        currFrame.timeUsage.orb = t.elapsed();

        t.restart();
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer); //匹配该帧与keyframes里最后一帧
        currFrame.timeUsage.motion = t.elapsed();

        switch (result) // 根据匹配结果不同采取不同策略
        {
            case NOT_MATCHED:
                //没匹配上，直接跳过
                cout << RED"Not enough inliers." << endl;
                break;
            case TOO_FAR_AWAY:
                // 太近了，也直接跳
                cout << RED"Too far away, may be an error." << endl;
                break;
            case TOO_CLOSE:
                // 太远了，可能出错了
                cout << RESET"Too close, not a keyframe" << endl;
                break;
            case KEYFRAME:
            {
                cout << GREEN"This is a new keyframe" << endl;
                // 不远不近，刚好
                /**
                 * This is important!!
                 * This is important!!
                 * This is important!!
                 * (very important so I've said three times!)
                 */
                // 检测回环
                bool nearLoop = false, farLoop = false;
                if (check_loop_closure)
                {
                    t.restart();
                    nearLoop = checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                    if(currIndex - keyframes.back().frameID > 10) //避免过于频繁的检测远距离回环
                    {
                        farLoop = checkRandomLoops(keyframes, currFrame, globalOptimizer);
                    }
                    currFrame.timeUsage.loop = t.elapsed();

                    if (nearLoop || farLoop)
                    {
                        t.restart();
                        globalOptimizer.initializeOptimization();
                        globalOptimizer.optimize(10); //可以指定优化步数
                        currFrame.timeUsage.optimize = t.elapsed();
                    }
                }

                t.restart();
                if (farLoop) //重新生成地图
                {
                    map->clear();
                    for (auto &f : keyframes)
                    {
                        // 从g2o里取出一帧
                        g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(globalOptimizer.vertex(f.frameID));
                        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
                        PointCloud::Ptr newCloud = image2PointCloud(f.rgb, f.depth, camera); //转成点云
                        // 以下是滤波
                        voxel.setInputCloud(newCloud);
                        voxel.filter(*tmp);
                        pass.setInputCloud(tmp);
                        pass.filter(*newCloud);
                        // 把点云变换后加入全局地图中
                        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
                        *map += *tmp;
                        tmp->clear();
                        newCloud->clear();
                    }
                }

                // 从g2o里取出一帧
                g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(globalOptimizer.vertex(currFrame.frameID));
                Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
                PointCloud::Ptr newCloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera); //转成点云
                // 以下是滤波
                voxel.setInputCloud(newCloud);
                voxel.filter(*tmp);
                pass.setInputCloud(tmp);
                pass.filter(*newCloud);
                // 把点云变换后加入全局地图中
                pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
                *map += *tmp;
                tmp->clear();
                newCloud->clear();

                currFrame.timeUsage.mapping = t.elapsed();

                keyframes.push_back(currFrame);
            }
                break;
            default:
                break;
        }
    }

    cout << RESET;

    ofstream of;
    of.open("timeusage.txt");
    of << fixed;
    for (auto &f : keyframes)
    {
        of << f.frameID << " " << f.timestamp << " ";
        of << f.timeUsage.orb << " " << f.timeUsage.motion << " " << f.timeUsage.loop << " "
           << f.timeUsage.optimize << " " << f.timeUsage.mapping << endl;
    }
    of.close();
    // 优化
    cout << RESET"optimizing pose graph, vertices: " << globalOptimizer.vertices().size() << endl;
    globalOptimizer.save("./result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(10); //可以指定优化步数
    globalOptimizer.save("./result_after.g2o");
    cout << "Optimization done." << endl;

    // 拼接点云地图
    cout << "saving the point cloud map..." << endl;

    voxel.setInputCloud(map);
    voxel.filter(*tmp);
    //存储
    pcl::io::savePCDFile("./result.pcd", *tmp);

    cout << "Final map is saved." << endl;
    return 0;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

CHECK_RESULT checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi(pd.getData("min_inliers").c_str());
    static double max_norm = atof(pd.getData("max_norm").c_str());
    static double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);

    if (is_loops == false)
        cout << RESET "goodMatchs: " << result.goodMatchs << " inliers: " << result.inliers;
    if (result.inliers < min_inliers) //inliers不够，放弃该帧
    {
        return NOT_MATCHED;
    }
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if (is_loops == false)
    {
        if (norm >= max_norm)
            return TOO_FAR_AWAY;   // too far away, may be error
    }
    else
    {
        if (norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if (norm <= keyframe_threshold)
        return TOO_CLOSE;   // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }
    // 边部分
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex(0, opti.vertex(f1.frameID));
    edge->setVertex(1, opti.vertex(f2.frameID));
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation(information);
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
    // edge->setMeasurement( T );
    edge->setMeasurement(T.inverse());
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}

/*
 * @return: detected loop or not;
 */
bool checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());

    CHECK_RESULT result;
    int result_cnt = 0;
    // 就是把currFrame和 frames里末尾几个测一遍
    if (frames.size() <= nearby_loops)
    {
        // no enough keyframes, check everyone
        for (size_t i = 0; i < frames.size(); i++)
        {
            result = checkKeyframes(frames[i], currFrame, opti, true);
            result_cnt += (result == KEYFRAME);
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++)
        {
            result = checkKeyframes(frames[i], currFrame, opti, true);
            result_cnt += (result == KEYFRAME);
        }
    }
    return (result_cnt > 0);
}

/*
 * @return: detected loop or not;
 */
bool checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());
    srandom(static_cast<unsigned int>(time(NULL)));
    // 随机取一些帧进行检测

    int size = frames.size();
    CHECK_RESULT result;
    int result_cnt = 0;
    if (size - nearby_loops >= random_loops)
    {
        // randomly check loops
        for (int i = 0; i < random_loops; i++)
        {
            int index = random() % (size - nearby_loops); //不包括已经近距离会换的关键帧
            result = checkKeyframes(frames[index], currFrame, opti, true);
            result_cnt += (result == KEYFRAME);
        }
    }
    return (result_cnt > 0);
}
