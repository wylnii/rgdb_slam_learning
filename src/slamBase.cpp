/*************************************************************************
    > File Name: src/slamBase.cpp
    > Author: xiang gao
    > Mail: gaoxiang12@mails.tsinghua.edu.cn
    > Implementation of slamBase.h
    > Created Time: 2015年07月18日 星期六 15时31分49秒
 ************************************************************************/

#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m+=2)
        for (int n=0; n < depth.cols; n+=2)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor )
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    _detector = cv::FeatureDetector::create( detector.c_str() );
    _descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if (!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
        return;
    }

    _detector->detect( frame.rgb, frame.kp );
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );

    return;
}

// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera , bool is_loops)
{
    static ParameterReader pd = ParameterReader::getInstance();
    static double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    static double min_good_match = atoi( pd.getData( "min_good_match" ).c_str() );
    static bool show_matches = pd.getData( "show_matches" )=="yes";
    vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );

    RESULT_OF_PNP result;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    //    if ( minDis < 10 )
    //        minDis = 10;
    
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }

    if(!is_loops && show_matches) //显示匹配过程
    {
        cv::Mat imgMatches;
        cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, matches, imgMatches );
        cv::imshow( "matchs", imgMatches );
        cv::waitKey( 0 );

        cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, goodMatches, imgMatches );
        cv::imshow( "good matchs", imgMatches );
        cv::waitKey( 0 );
    }


    result.goodMatchs = goodMatches.size();

    if (goodMatches.size() <= min_good_match)
    {
        result.inliers = -1;
        return result;
    }
    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    if (pts_obj.size() ==0 || pts_img.size()==0)
    {
        result.inliers = -1;
        return result;
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    if(!is_loops && show_matches)
    {
        vector< cv::DMatch > matchesShow;
        for (size_t i=0; i<inliers.rows; i++)
        {
            matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );
        }
        cv::Mat imgMatches;
        cv::drawMatches( frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, matchesShow, imgMatches );
        cv::imshow( "inlier matches", imgMatches );
//        cv::waitKey( 0 );
    }

    return result;
}


// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ )
            r(i,j) = R.at<double>(i,j);

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);
    return T;
}

// joinPointCloud 
// 输入：原始点云，新来的帧以及它的位姿
// 输出：将新来帧加到原始帧后的图像
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera ) 
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );

    // 合并点云
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd = ParameterReader::getInstance();
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}


FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}

FRAME readFrame(int index, vector<string> &rgbList, vector<string> &depthList, ParameterReader& pd)
{
    if(rgbList.size() <= index)
    {
        cout << RED"rgbList.size() <= index" << endl;
    }
    FRAME f;
    f.imageName = rgbList[index];
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    stringstream ss;
    ss<<rgbDir<<rgbList[index];
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<depthList[index];
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}

bool getImageListFromConfigFile(vector<string> &rgbList, vector<string> &depthList, ParameterReader& pd)
{
    string imagelist_file   =   pd.getData("imagelist_file");

    cout << "loading images from file : " << imagelist_file << endl;

    ifstream fin( imagelist_file );
    if(! fin.good())
    {
        cout << RED "can not open file!!!" << endl;
        return false;
    }
    rgbList.clear();
    depthList.clear();

    while(!fin.eof())
    {
        string str;
        getline(fin,str);
        if(str[0] == '#')
            continue;
        int pos = str.find(' ');
        if(pos < 0)
            continue;
        string file1 = str.substr(0, pos);
        string file2 = str.substr(pos+1, str.length());
        rgbList.push_back(file1);
        depthList.push_back(file2);
        if(fin.bad())
            return false;
    }
    cout << " total "<<rgbList.size() << " RGB images, " << depthList.size() << " DEPTH images" << endl;

    if(rgbList.size() != depthList.size())
    {
        cout << RED "rgbList.size() != depthList.size()" << endl;
        return false;
    }
    return true;
}
