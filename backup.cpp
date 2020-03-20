//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>

#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_types.h>
#include <boost/thread/thread.hpp>

#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

#include <ceres/rotation.h>
#include <ceres/problem.h>
#include <ceres/ceres.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>

#include "tinydir.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <typeinfo>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#define OUTPUT_NAME "/home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/Viewer/structures.yml"

using std::cin;
using namespace cv;
using namespace std;
using namespace ceres;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

void get_matched_points(
        vector<KeyPoint>& p1,
        vector<KeyPoint>& p2,
        vector<DMatch> matches,
        vector<Point2f>& out_p1,
        vector<Point2f>& out_p2
        )
{
        out_p1.clear();
        out_p2.clear();
        for (int i = 0; i < matches.size(); ++i)
        {
                out_p1.push_back(p1[matches[i].queryIdx].pt);
                out_p2.push_back(p2[matches[i].trainIdx].pt);
        }
}
void get_matched_colors(
        vector<Vec3b>& c1,
        vector<Vec3b>& c2,
        vector<DMatch> matches,
        vector<Vec3b>& out_c1,
        vector<Vec3b>& out_c2
        )
{
        out_c1.clear();
        out_c2.clear();
        for (int i = 0; i < matches.size(); ++i)
        {
                out_c1.push_back(c1[matches[i].queryIdx]);
                out_c2.push_back(c2[matches[i].trainIdx]);
        }
}

void get_objpoints_and_imgpoints(
        vector<DMatch>& matches,
        vector<int>& struct_indices,
        vector<Point3d>& structure,
        vector<KeyPoint>& key_points,
        vector<Point3f>& object_points,
        vector<Point2f>& image_points)
{
        object_points.clear();
        image_points.clear();

        cout << endl << "get_objpoints_and_imgpointsssssssssssssssssss " << endl;
        cout << "matches = " << matches.size() << endl;
        cout << "structure = " << structure.size() << endl;
        cout << "KeyPoint = " << key_points.size() << endl;
 
        for (int i = 0; i < matches.size(); ++i)
        {
            int query_idx = matches[i].queryIdx;
            int train_idx = matches[i].trainIdx;

            //cout << "query_idx = " << query_idx << "train_idx" << train_idx << endl;
 
            int struct_idx = struct_indices[query_idx];

            if (struct_idx < 0)
            {
                //cout << "melbu tolak" << endl;
                continue;
            }

            object_points.push_back(structure[struct_idx]);
            image_points.push_back(key_points[train_idx].pt);

            //cout << "melbu last" << endl;
        }

        cout << "object_points = " << object_points.size() << endl;
        cout << "image_points = " << image_points.size() << endl;
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
        vector<Point2f> p1_copy = p1;
        p1.clear();
 
        for (int i = 0; i < mask.rows; ++i)
        {
                if (mask.at<uchar>(i) > 0)
                        p1.push_back(p1_copy[i]);
        }
}
void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
        vector<Vec3b> p1_copy = p1;
        p1.clear();
        for (int i = 0; i < mask.rows; ++i)
        {
                if (mask.at<uchar>(i) > 0)
                        p1.push_back(p1_copy[i]);
        }
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
        
        double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
        Point2d principle_point(K.at<double>(2), K.at<double>(5));
 
        //
        Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
        if (E.empty()) return false;
 
        double feasible_count = countNonZero(mask);
        //cout << (int)feasible_count << " -in- " << p1.size() << endl;
        //
        if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
                return false;
 
        int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
 
        if (((double)pass_count) / feasible_count < 0.7)
                return false;
        return true;
}


void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)
{
        Mat proj1(3, 4, CV_32FC1);
        Mat proj2(3, 4, CV_32FC1);
 
        R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
        T1.convertTo(proj1.col(3), CV_32FC1);
 
        R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
        T2.convertTo(proj2.col(3), CV_32FC1);
 
        Mat fK;
        K.convertTo(fK, CV_32FC1);
        proj1 = fK*proj1;
        proj2 = fK*proj2;

        cout << "fK" << endl << fK << endl;

        Mat s;
        triangulatePoints(proj1, proj2, p1, p2, s);
 
        structure.clear();
        structure.reserve(s.cols);
        for (int i = 0; i < s.cols; ++i)
        {
                Mat_<float> col = s.col(i);
                col /= col(3);
                structure.push_back(Point3f(col(0), col(1), col(2)));
        }
}

void fusion_structure(
        vector<DMatch>& matches,
        vector<int>& struct_indices,
        vector<int>& next_struct_indices,
        vector<Point3d>& structure,
        vector<Point3d>& next_structure,
        vector<Vec3b>& colors,
        vector<Vec3b>& next_colors
        )
{
        for (int i = 0; i < matches.size(); ++i)
        {
                int query_idx = matches[i].queryIdx;
                int train_idx = matches[i].trainIdx;
 
                int struct_idx = struct_indices[query_idx];
                if (struct_idx >= 0)
                {
                        next_struct_indices[train_idx] = struct_idx;
                        continue;
                }
                
                structure.push_back(next_structure[i]);
                colors.push_back(next_colors[i]);
                struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
        }
}


void init_structure(
        Mat K,
        vector<vector<KeyPoint>>& key_points_for_all,
        vector<vector<Vec3b>>& colors_for_all,
        vector<vector<DMatch>>& matches_for_all,
        vector<Point3d>& structure,
        vector<vector<int>>& correspond_struct_idx,
        vector<Vec3b>& colors,
        vector<Mat>& rotations,
        vector<Mat>& motions
        )
{
        //
        vector<Point2f> p1, p2;
        vector<Vec3b> c2;
        Mat R, T;       
        Mat mask;       
        get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
        get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
        find_transform(K, p1, p2, R, T, mask);
 
        
        maskout_points(p1, mask);
        maskout_points(p2, mask);

        maskout_colors(colors, mask);
 
        Mat R0 = Mat::eye(3, 3, CV_64FC1);
        Mat T0 = Mat::zeros(3, 1, CV_64FC1);
        reconstruct(K, R0, T0, R, T, p1, p2, structure);
        //
        rotations = { R0, R };
        motions = { T0, T };
 
        //
        correspond_struct_idx.clear();
        correspond_struct_idx.resize(key_points_for_all.size());
        for (int i = 0; i < key_points_for_all.size(); ++i)
        {
                correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
        }
 
        //
        int idx = 0;
        vector<DMatch>& matches = matches_for_all[0];
        for (int i = 0; i < matches.size(); ++i)
        {
                if (mask.at<uchar>(i) == 0)
                        continue;
 
                correspond_struct_idx[0][matches[i].queryIdx] = idx;
                correspond_struct_idx[1][matches[i].trainIdx] = idx;
                ++idx;
        }
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
    int n = (int)rotations.size();

    FileStorage fs(file_name, FileStorage::WRITE);
    fs << "Camera Count" << n;
    fs << "Point Count" << structure.cols;
    
    fs << "Rotations" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << rotations[i];
    }
    fs << "]";

    fs << "Motions" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << motions[i];
    }
    fs << "]";

    fs << "Points" << "[";
    for (size_t i = 0; i < structure.cols; ++i)
    {
        Mat_<float> c = structure.col(i);
        c /= c(3);  // Koordinat homogen, yang perlu dibagi dengan elemen terakhir untuk menjadi nilai koordinat sebenarnya
        fs << Point3f(c(0), c(1), c(2));
    }
    fs << "]";

    fs << "Colors" << "[";
    for (size_t i = 0; i < colors.size(); ++i)
    {
        fs << colors[i];
    }
    fs << "]";

    fs.release();
}

struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d& observation)
        : observation(observation)
    {
    }

    template <typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
    {
        const T* r = extrinsic;
        const T* t = &extrinsic[3];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }
};


// bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
// instrinsic selalu sama
// correspond_struct_idx = sama kayak init_structure
// key_points_for_all = sama kayak init_structure
// structure = sama kayak init_structure

void bundle_adjustment(
    Mat& intrinsic,
    vector<Mat>& extrinsics,
    vector<vector<int> >& correspond_struct_idx,
    vector<vector<KeyPoint> >& key_points_for_all,
    vector<Point3d>& structure
)
{
    Problem problem;

    // load extrinsics (rotations and motions)
    for (size_t i = 0; i < extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);//Add a parameter block with appropriate size and parameterization to the problem.
                //Repeated calls with the same arguments are ignored. Repeated calls with the same double pointer but a different size results in undefined behavior.
    }
    // fix the first camera.
    problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());//Hold the indicated parameter block constant during optimization.保持第一个外惨矩阵不变

    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

    // load points
    LossFunction* loss_function = new HuberLoss(4);   // loss function make bundle adjustment robuster.

    //cout << "correspond_struct_idx = " <<  correspond_struct_idx.size() << endl;
    //cout << "key_points_for_all = " <<  key_points_for_all .size() << endl;

    for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
    {
        vector<int>& point3d_ids = correspond_struct_idx[img_idx];
        vector<KeyPoint>& key_points = key_points_for_all[img_idx];

        //cout << "point3d_ids = " <<  point3d_ids.size() << endl;
        //cout << "key_points = " <<  key_points.size() << endl;

        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;

            Point2d observed = key_points[point_idx].pt;//corresponding 2D points coordinates with feasible 3D point
            CostFunction* cost_function = new AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
            problem.AddResidualBlock(//adds a residual block to the problem,implicitly adds the parameter blocks(This causes additional correctness checking) if they are not present
                cost_function,
                loss_function,
                intrinsic.ptr<double>(),            // Intrinsic
                extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
                &(structure[point3d_id].x)          // Point in 3D space
            );
        }
    }

    // Solve BA
    Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = SILENT;
    ceres_config_options.num_threads = 1;//Number of threads to be used for evaluating the Jacobian and estimation of covariance.
    ceres_config_options.preconditioner_type = JACOBI;
    ceres_config_options.linear_solver_type = DENSE_SCHUR;
    // ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//ype of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm
    // ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    Solver::Summary summary;
    Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }
}


int main( int argc, char** argv )
{
    // STAGE 1
    vector<KeyPoint> key_points1, key_points2;
    vector<vector<KeyPoint>> key_points_for_all;
    vector<Mat> descriptor_for_all;
    vector<vector<Vec3b>> colors_for_all;
    vector<vector<DMatch>> matches_for_all;
    vector<vector<DMatch>> knn_matches;
    Mat descriptor1;
    Mat descriptor2;
    /*Mat K(Matx33d(
            2759.48, 0, 1520.69,
            0, 2764.16, 1006.81,
            0,      0,      1)); */
    vector<Point3d> structure;
    vector<Point3d> structure_copy;
    vector<vector<int>> correspond_struct_idx;
    vector<Vec3b> colors;
    vector<Mat> rotations;
    vector<Mat> motions;
    vector<Point3d> save_structure[100];
    vector<Point3d> new_structure;

    Mat ngehe = Mat::eye(3, 3, CV_64FC1);

    // outt2.mp4
    /*Mat K(Matx33d(
            718.85, 0, 607.19,
            0, 718.85, 185.21,
            0,      0,      1));*/

    // video 1
    /*Mat K(Matx33d(
            1100, 0, 639.19,
            0, 1100, 359.21,
            0,      0,      1));*/

    /*Mat K(Matx33d(
            1100, 0, 637,
            0, 1100, 317,
            0,  0, 1)); */

    // belok.mp4
    /*Mat K(Matx33d(
            718, 0, 607.19,
            0, 718, 185.21,
            0,      0,      1)); */
        
    
    VideoCapture cap("/home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/build/video_1.avi");
    //VideoCapture cap("/home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/Video_Accident/accident_15.mp4");

    int cx, cy, fx, fy;

    Mat whread;
    cap >> whread;
    cout << "width = " << whread.size().width << endl;

    cx = ((whread.size().width/2) - 1);
    cy = ((whread.size().height/2) - 1);

    fx = 0.57*whread.size().width;
    fy = 0.57*whread.size().width;

    Mat K(Matx33d(
            fx, 0, cx,
            0, fy, cy,
            0,  0, 1));

    cout << K << endl;


    Mat currImage_c, currImage, prevImage_c, prevImage, prevImages;
    int p = 0;
    int w = 0;
    int g = 1;
    int l = 0;

    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    typedef pcl::PointXYZRGB PointT;                                    // CREATE POINT STRUCTURE PCL
    typedef pcl::PointCloud<PointT> PointCloud;                         // BASE CLASS IN PCL FOR STORING COLECTIONS OF 3D POINTS
    PointCloud::Ptr pointcloud( new PointCloud );

    // Read bbox initialization
    int h = 0;
    int x1[100], x2[100], y1[100], y2[100];
    int xx1, xx2, yy1, yy2;


    while(1)
    {
        cap >> currImage_c;
        cout << endl << "g" << g << endl;
        //cout << endl << "i" << i << endl;

        if (currImage_c.empty())
        {
            break;
        }


        // BBOX Initialization
        Mat currImage = Mat(currImage_c.size(), currImage_c.type(), Scalar::all(0));
        string file_name = string("//home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/re3_bbox/fix4_") + to_string(g) + ".txt";
        //string file_name = string("//home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/new_bbox/accident15_") + to_string(g) + ".txt";
        h = 0;
        ifstream file;
        file.open(file_name);

        cout <<"File name = " << file_name << endl;

        while (file >> xx1 >> yy1 >> xx2 >> yy2)
        {
            //cout << "In the while" << endl;
            // MASK R-CNN FILE
            //cout << "y1 = " << yy1 << " x1 = " << xx1 << " y2 = " << yy2 << " x2 = " << xx2 << endl;
            //y1[h] = yy1;
            //x1[h] = xx1;
            //y2[h] = yy2 - yy1;
            //x2[h] = xx2 - xx1;
            //h++;

            cout << "x1 = " << xx1 << " x2 = " << xx2 << " y1 = " << yy1 << " y2 = " << yy2 << endl;
            if(xx1 <= 0)
            {
                xx1 = 0;
            }
            if(yy1 <= 0)
            {
                yy1 = 0;
            }
            if(xx2 > currImage_c.cols)
            {
                xx2 = currImage_c.cols;
            }
            if(yy2 > currImage_c.rows)
            {
                yy2 = currImage_c.rows;
            }
            x1[h] = xx1;
            y1[h] = yy1;
            x2[h] = xx2 - xx1;
            y2[h] = yy2 - yy1;
            h++;
        }

        cout << " " << endl;
        cout << "continue = " << currImage_c.cols << " " << currImage_c.rows << endl;
        // cols = x
        // rows =  y

        for (int n = 0; n < h; ++n)
        {   
            cout << "x1 = " << x1[n] << " y1 = " << y1[n] << " x2 = " << x2[n] << " y2 = " << y2[n] << endl;
            Rect r1(x1[n], y1[n], x2[n], y2[n]);
            cout << "check 1" << endl;
            currImage_c(r1).copyTo(currImage(r1));
            cout << "check 2" << endl;
        }

        //Rect r1(84,81,330,376);
        //currImage_c(r1).copyTo(currImage(r1));
        //imshow("ehehhe", currImage);


        if(prevImage.empty())
        {
            currImage.copyTo(prevImage);
            //cvtColor(prevImage, prevImage, COLOR_BGR2GRAY);

            cout << endl << "i" << p << endl;

            Ptr<AKAZE> feature = AKAZE::create();
            //Ptr<ORB> feature = ORB::create();
            assert(!prevImage.empty());
            feature->detect(prevImage, key_points1);
            feature->compute(prevImage, key_points1, descriptor1);
            cout << "Jumlah KeyPoint = " << key_points1.size() << endl;
            Mat output_key1;
            drawKeypoints(prevImage, key_points1, output_key1);
            

            vector<Vec3b> colors_for_all11(key_points1.size());
            for (int i = 0; i < key_points1.size(); ++i)
            {
                Point2f& p = key_points1[i].pt;
                colors_for_all11[i] = prevImage.at<Vec3b>(p.y, p.x);
            }

            key_points_for_all.push_back(key_points1);
            descriptor_for_all.push_back(descriptor1);
            colors_for_all.push_back(colors_for_all11);

            prevImages = prevImage.clone();

            p++;
            g++;

            cap >> currImage;
        }

        if(g%3 == 0)
        {
            cout << endl << "i" << p << " g" << g << endl;;
            //cvtColor(currImage, currImage, COLOR_BGR2GRAY);

            Ptr<AKAZE> feature = AKAZE::create();
            //Ptr<ORB> feature = ORB::create();
            assert(!currImage.empty());
            feature->detect(currImage, key_points2);
            feature->compute(currImage, key_points2, descriptor2);
            cout << "Jumlah KeyPoint = " << key_points2.size() << endl;
            Mat output_key2;
            drawKeypoints(currImage, key_points2, output_key2);
            //imshow("ngeee", output_key2);

            imshow("hehffe", output_key2);
            //waitKey(0);

            vector<Vec3b> colors_for_all22(key_points2.size());
            for (int i = 0; i < key_points2.size(); ++i)
            {
                Point2f& p = key_points2[i].pt;
                colors_for_all22[i] = currImage.at<Vec3b>(p.y, p.x);
            }

            key_points_for_all.push_back(key_points2);
            descriptor_for_all.push_back(descriptor2);
            colors_for_all.push_back(colors_for_all22);

            vector<DMatch> matches;

            // First matcher methods
            //BFMatcher matcher(NORM_L2, false);
            //matcher.knnMatch(descriptor_for_all[p-1], descriptor_for_all[p], knn_matches, 2);

            // Second matcher method
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
            matcher->knnMatch(descriptor_for_all[p-1], descriptor_for_all[p], knn_matches, 2);

            float min_dist = FLT_MAX;
            for (int r = 0; r < knn_matches.size(); ++r)
            {
                //Ratio Test
                if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
                    continue;

                float dist = knn_matches[r][0].distance;
                if (dist < min_dist) min_dist = dist;
            }
        
            matches.clear();

            for (size_t r = 0; r < knn_matches.size(); ++r)
            {
                /*if (
                    knn_matches[r][0].distance > 0.7*knn_matches[r][1].distance ||
                    knn_matches[r][0].distance > 6 * max(min_dist, 10.0f)
                    )
                    continue;
                matches.push_back(knn_matches[r][0]); */

                if(knn_matches[r][0].distance <= 0.7 * knn_matches[r][1].distance) 
                {
                    matches.push_back(knn_matches[r][0]);
                } 
            }
            matches_for_all.push_back(matches);
            knn_matches.clear();

            // Display matches
            Mat img_matches;
            //drawMatches(prevImages, key_points_for_all[p-1], currImage, key_points_for_all[p], matches_for_all[p-1], img_matches);
            //imwrite("asoe.jpg", img_matches);


            //imshow("Current", currImage);
            //imshow("Previous", prevImages);


            //imshow("asoe.jpg", img_matches);
            //imwrite("1_1.jpg", img_matches);
            //waitKey(0);

            if(p >=1 )
            {
                vector<Point2f> p1, p2;
                vector<Vec3b> c2;
                Mat init_R, init_T;       
                Mat mask;    
                get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);

                // for VO
                //get_matched_points(key_points_for_all[p-1], key_points_for_all[p], matches_for_all[p-1], p1, p2);

                cout << "K sekarang = " << endl << K << endl;

                find_transform(K, p1, p2, init_R, init_T, mask);

                if(p == 1)
                {
                    get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
                    maskout_points(p1, mask);
                    maskout_points(p2, mask);
                    maskout_colors(colors, mask);
                    Mat R0 = Mat::eye(3, 3, CV_64FC1);
                    Mat T0 = Mat::zeros(3, 1, CV_64FC1);

                    cout << "R0 T0 " << endl;
                    //cout << "R = " << endl << R0.size() << endl;
                    //cout << "T = " << endl << T0.size() << endl;
                    cout << "init_R = " << endl << init_R << endl;
                    cout << "init_T = " << endl << init_T << endl;

                    reconstruct(K, R0, T0, init_R, init_T, p1, p2, structure);

                    rotations = { R0, init_R };
                    motions = { T0, init_T };

                    cout << "Pergerakan " << endl;
                    cout << "R = " << endl << rotations[1] << endl;
                   //cout << "T = " << endl << motions << endl;
                }

                correspond_struct_idx.resize(key_points_for_all.size());

                for (int i = 0; i < key_points_for_all.size(); ++i)
                {
                    correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
                }

                int idx = 0;
                vector<DMatch>& matches = matches_for_all[0];
                //cout << "Jumlah matches = " << matches.size() << "/" << key_points_for_all[p].size() << endl;

                for (int i = 0; i < matches.size(); ++i)
                {
                    if (mask.at<uchar>(i) == 0)
                        continue;
                    correspond_struct_idx[0][matches[i].queryIdx] = idx;
                    correspond_struct_idx[1][matches[i].trainIdx] = idx;
                    ++idx;
                }

                if(p >= 2)
                {
                    vector<Point3f> object_points;
                    vector<Point2f> image_points;
                    Mat r, R, T;

                    cout << "lebih dari dua" << endl;

                    object_points.clear();
                    image_points.clear();

                    for (int g = 0; g < matches_for_all[p-1].size(); ++g)
                    {
                        int query_idx = matches_for_all[p-1][g].queryIdx;
                        int train_idx = matches_for_all[p-1][g].trainIdx;
    
                        int struct_idx = correspond_struct_idx[p-1][query_idx];

                        if (struct_idx < 0)
                        {
                            continue;
                        }

                        object_points.push_back(structure[struct_idx]);
                        image_points.push_back(key_points_for_all[p][train_idx].pt);
                    }

                    cout << "object_points = " << object_points.size() << endl;
                    cout << "image_points = " << image_points.size() << endl;

                    solvePnPRansac(object_points, image_points, K, noArray(), r, T);
                    Rodrigues(r, R);

                    rotations.push_back(R);
                    motions.push_back(T);
 
                    vector<Point2f> p1, p2;
                    vector<Vec3b> c1, c2;
                    get_matched_points(key_points_for_all[p-1], key_points_for_all[p], matches_for_all[p-1], p1, p2);
                    get_matched_colors(colors_for_all[p-1], colors_for_all[p], matches_for_all[p-1], c1, c2);

                    vector<Point3d> next_structure;
                    reconstruct(K, rotations[p-1], motions[p-1], R, T, p1, p2, next_structure);
 
                    cout << "Colors 1 = " << colors.size() << endl;

                    fusion_structure(
                        matches_for_all[p-1],
                        correspond_struct_idx[p-1],
                        correspond_struct_idx[p],
                        structure,
                        next_structure,
                        colors,
                        c1
                        );

                    structure_copy = structure;

                    // -------------------------------------------------------------------------------------------------------------
                    // Bundle Adjustment
                    // GTSAM
                    /*
                    gtsam::Values result;
                    using namespace gtsam;

                    double cx_new = ((whread.size().width/2) - 1);
                    double cy_new = ((whread.size().height/2) - 1);

                    Cal3_S2 K(fx, fy, 0 /* skew *///, cx, cy);
                    /*noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0);

                    NonlinearFactorGraph graph;
                    Values initial;


                    cout << endl << " R = " << endl << R << endl;
                    cout << " T = " << endl << T << endl;

                    cout << endl << " R2 = " << endl << rotations[p] << endl;
                    cout << " T2 = " << endl << motions[p] << endl;

                    Rot3 R_new(
                        rotations[p].at<double>(0,0),
                        rotations[p].at<double>(0,1),
                        rotations[p].at<double>(0,2),

                        rotations[p].at<double>(1,0),
                        rotations[p].at<double>(1,1),
                        rotations[p].at<double>(1,2),

                        rotations[p].at<double>(2,0),
                        rotations[p].at<double>(2,1),
                        rotations[p].at<double>(2,2)
                    );

                    cout << "R GTSAM" << R_new << endl;

                    Point3 t;
                    t(0) = motions[p].at<double>(0,0);
                    t(1) = motions[p].at<double>(0,1);
                    t(2) = motions[p].at<double>(0,2);

                    cout << "T GTSAM" << t << endl;

                    Pose3 pose(R_new, t);

                    if (l == 0) {
                        noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                        graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), pose, pose_noise); // add directly to graph
                        cout << "Init noise" << endl;
                    }

                    initial.insert(Symbol('x', l), pose);

                    initial.insert(Symbol('K', 0), K);
                    noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 100, 100, 0.01 /*skew*///, 100, 100).finished());
                    /*
                    graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);

                    bool init_prior = false;
                    */
                    // -------------------------------------------------------------------------------------------------------------

    
                    
                    // -------------------------------------------------------------------------------------------------------------
                    // Bundle Adjustment
                    // CERES SOLVER

                    /*for(int q = 0; q <= key_points_for_all.size(); q++)
                    {
                        for(int w = 0; w <= key_points_for_all[q].size(); w++)
                        {
                            key_points_for_all_new[q][w] = key_points_for_all[q][w];
                        }
                    }*/

                    /*for(int q = 0; q <= correspond_struct_idx.size(); q++)
                    {
                        for(int w = 0; w <= correspond_struct_idx[q].size(); w++)
                        {
                            cout << "correspond_struct_idx = " << correspond_struct_idx.size() << endl;
                            cout << "correspond_struct_idx[q] = " << correspond_struct_idx[q].size() << endl;
                            cout << "correspond_struct_idx[q][w] = " << correspond_struct_idx[q][w] << endl;
                            //correspond_struct_idx_new[q][w] = correspond_struct_idx[q][w];
                        }
                    } */

                    if(p%30 == 0)
                    {
                        Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
                        vector<Mat> extrinsics;
                        vector<vector<KeyPoint>> key_points_for_all_new;
                        vector<vector<int>> correspond_struct_idx_new;

                        cout << "total rotations = " << rotations.size() << endl;

                        for(size_t i = p-10; i < p; i++)
                        {
                            cout <<"size - " << i << endl;
                            Mat extrinsic(6, 1, CV_64FC1);
                            Mat r;
                            Rodrigues(rotations[i], r);
                            r.copyTo(extrinsic.rowRange(0, 3));
                            motions[i].copyTo(extrinsic.rowRange(3, 6));
                            extrinsics.push_back(extrinsic);

                            key_points_for_all_new.push_back(key_points_for_all[i]);
                            correspond_struct_idx_new.push_back(correspond_struct_idx[i]);
                        }

                        /*for(size_t i = 0; i < rotations.size(); i++)
                        {
                            cout <<"size - " << i << endl;
                            Mat extrinsic(6, 1, CV_64FC1);
                            Mat r;
                            Rodrigues(rotations[i], r);
                            r.copyTo(extrinsic.rowRange(0, 3));
                            motions[i].copyTo(extrinsic.rowRange(3, 6));
                            extrinsics.push_back(extrinsic);

                            key_points_for_all_new.push_back(key_points_for_all[i]);
                            correspond_struct_idx_new.push_back(correspond_struct_idx[i]);
                        }*/


                        cout << "total extrinsics = " << extrinsics.size() << endl;
                        cout << "total key_points_for_all_new = " << key_points_for_all_new.size() << endl;
                        cout << "total correspond_struct_idx_new = " << correspond_struct_idx_new.size() << endl;

                        cout <<"intrinsic 1 = " << endl << intrinsic << endl;
                        bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx_new, key_points_for_all_new, structure_copy);
                        cout <<"intrinsic 2 = " << endl << intrinsic << endl;

                        float fx_new = intrinsic.at<double>(0, 0);
                        float fy_new = intrinsic.at<double>(0, 1);
                        float cx_new = intrinsic.at<double>(0, 2);
                        float cy_new = intrinsic.at<double>(0, 3);

                        /*cout << "fx_new" << fx_new << endl;
                        cout << "fy_new" << fy_new << endl;
                        cout << "cx_new" << cx_new << endl;
                        cout << "cy_new" << cy_new << endl;
                        Mat K(Matx33d(
                            fx_new, 0, cx_new,
                            0, fy_new, cy_new,
                            0,  0, 1));
                        ngehe = K;
                        cout << "K baru = " << endl << ngehe << endl; */
                    
                        K.at<double>(0, 0) = fx_new;
                        K.at<double>(1, 1) = fy_new;
                        K.at<double>(0, 2) = cx_new;
                        K.at<double>(1, 2) = cy_new;
                        cout << "K anyar" << endl << K << endl;
                    } 
                    cout << "K paling anyar" << endl << K << endl;
                    // -------------------------------------------------------------------------------------------------------------
                    



                    // -------------------------------------------------------------------------------------------------------------
                    // Bundle Adjustment
                    // CERES SOLVER
                    /*if(p%30 == 0)
                    {
                        Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
                        vector<Mat> extrinsics;
                        
                        /*for(size_t i = 0; i < rotations.size(); i++)
                        {
                            cout <<"size - " << i << endl;
                            Mat extrinsic(6, 1, CV_64FC1);
                            Mat r;
                            Rodrigues(rotations[i], r);
                            r.copyTo(extrinsic.rowRange(0, 3));
                            motions[i].copyTo(extrinsic.rowRange(3, 6));
                            extrinsics.push_back(extrinsic);

                        } */

                        
                        /*cout <<"intrinsic 1 = " << endl << intrinsic << endl;
                        bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure_copy);
                        cout <<"intrinsic 2 = " << endl << intrinsic << endl;

                        float fx_new = intrinsic.at<double>(0, 0);
                        float fy_new = intrinsic.at<double>(0, 1);
                        float cx_new = intrinsic.at<double>(0, 2);
                        float cy_new = intrinsic.at<double>(0, 3);

                        /*cout << "fx_new" << fx_new << endl;
                        cout << "fy_new" << fy_new << endl;
                        cout << "cx_new" << cx_new << endl;
                        cout << "cy_new" << cy_new << endl;
                        Mat K(Matx33d(
                            fx_new, 0, cx_new,
                            0, fy_new, cy_new,
                            0,  0, 1));
                        ngehe = K;
                        cout << "K baru = " << endl << ngehe << endl; */
                    
                        /*K.at<double>(0, 0) = fx_new;
                        K.at<double>(1, 1) = fy_new;
                        K.at<double>(0, 2) = cx_new;
                        K.at<double>(1, 2) = cy_new;
                        cout << "K anyar" << endl << K << endl;
                    } 
                    cout << "K paling anyar" << endl << K << endl;
                    

                    */
                    // -------------------------------------------------------------------------------------------------------------





                l++;
                }

            }

            // R and T out
            
            /*
            // Bundle Adjustment
            using namespace gtsam;

            double cx_new = ((whread.size().width/2) - 1);
            double cy_new = ((whread.size().height/2) - 1);

            Cal3_S2 K(fx, fy, 0 /* skew *//*, cx, cy);
            /*noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

            cout << "K = " << endl << K << endl;

            NonlinearFactorGraph graph;
            Values initial; */

            /*Rot3 R(
                img_pose.T.at<double>(0,0),
                img_pose.T.at<double>(0,1),
                img_pose.T.at<double>(0,2),

                img_pose.T.at<double>(1,0),
                img_pose.T.at<double>(1,1),
                img_pose.T.at<double>(1,2),

                img_pose.T.at<double>(2,0),
                img_pose.T.at<double>(2,1),
                img_pose.T.at<double>(2,2)
            );

            cout << "R GTSAM" << R << endl; 
            */


            // -----------------------------------------------------------------------------------------------------------
            // Here add PCL
            /*if(p == 32)
            {
                cout << "jumlah struktur after BA = " << structure.size() << endl;
                //cout << "structure =  " << endl;
                //cout << structure << endl;

                typedef pcl::PointXYZRGB PointT;
                typedef pcl::PointCloud<PointT> PointCloud;
                PointCloud::Ptr pointcloud( new PointCloud );
                for (size_t i = 0; i < structure.size(); ++i)
                {
                    /*PointT p;
                    p.x = structure[i].x;
                    p.y = structure[i].y;
                    p.z = structure[i].z;
                    p.b = colors[i][0];
                    p.g = colors[i][1];
                    p.r = colors[i][2];
                    pointcloud->points.push_back( p );*/
                    /*
                    if(structure[i].x < 150 && structure[i].y < 150 && structure[i].z < 150 && structure[i].x > -150 && structure[i].y > -150 && structure[i].z > -150)
                    {
                        PointT p;
                        p.x = structure[i].x;
                        p.y = structure[i].y;
                        p.z = structure[i].z;
                        p.b = colors[i][0];
                        p.g = colors[i][1];
                        p.r = colors[i][2];
                        pointcloud->points.push_back( p );
                    }
                    
                }
                pointcloud->is_dense = false;
                pcl::visualization::CloudViewer viewer("Cloud Viewer");
                viewer.showCloud(pointcloud); 

                int user_data;
                while(!viewer.wasStopped())
                {
                    user_data++;
                }
            }
            */
            // -----------------------------------------------------------------------------------------------------------

            prevImages = currImage.clone();
            p++;
        }
        prevImages = currImage.clone();
        g++;
        waitKey(1);
    }

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    
    for (size_t i = 0; i < structure_copy.size(); ++i)
    { 
        if(structure[i].x < 200 && structure[i].y < 200 && structure[i].z < 200 && structure[i].x > -200 && structure[i].y > -200 && structure[i].z > -200)
        {
            PointT p;
            p.x = structure[i].x;
            p.y = structure[i].y;
            p.z = structure[i].z;
            p.b = colors[i][0];
            p.g = colors[i][1];
            p.r = colors[i][2];
            pointcloud->points.push_back( p );
        }
        else
        {
            continue;
        } 
    }

    // ORIGINAL
    //pointcloud->is_dense = false;
    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
    //viewer.showCloud(pointcloud); 

    // Cobaan
    pointcloud->is_dense = false;
    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Cloud Viewer"));

    // Define the bounding box here
    pcl::MomentOfInertiaEstimation <pcl::PointXYZRGB> feature_extractor;
    feature_extractor.setInputCloud(pointcloud);
    feature_extractor.compute ();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZRGB min_point_AABB;
    pcl::PointXYZRGB max_point_AABB;
    pcl::PointXYZRGB min_point_OBB;
    pcl::PointXYZRGB max_point_OBB;
    pcl::PointXYZRGB position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia (moment_of_inertia);
    feature_extractor.getEccentricity (eccentricity);
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter (mass_center);

    //pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->addPointCloud<pcl::PointXYZRGB> (pointcloud, "sample cloud");
    viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB", 0);

    viewer->setRepresentationToWireframeForAllActors();

    // Used to show the cloud
    /*int user_data;
    while(!viewer.wasStopped())
    {
        user_data++;
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }*/


    // Adder
    viewer->addPointCloud (pointcloud, "cloud");

    int user_data;
    //viewer.showCloud(pointcloud); 
    while(!viewer->wasStopped())
    {
        user_data++;
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }













    /*typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    for (size_t i = 0; i < structure_copy.size(); ++i)
    { 
        if(structure_copy[i].x < 150 && structure_copy[i].y < 150 && structure_copy[i].z < 150)
        {
            PointT p;
            p.x = structure_copy[i].x;
            p.y = structure_copy[i].y;
            p.z = structure_copy[i].z;
            p.b = colors[i][0];
            p.g = colors[i][1];
            p.r = colors[i][2];
            pointcloud->points.push_back( p );
        }
        else
        {
            continue;
        } 
    }
    pointcloud->is_dense = false;
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(pointcloud); 

    int user_data;
    while(!viewer.wasStopped())
    {
        user_data++;
    }*/


    // Bundle Adjustment at the end
    /*
    Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
    vector<Mat> extrinsics;

    cout << "total rotations = " << rotations.size() << endl;

    for (size_t i = 0; i < rotations.size(); ++i)
    {
        cout <<"size - " << i << endl;
        Mat extrinsic(6, 1, CV_64FC1);
        Mat r;
        Rodrigues(rotations[i], r);
        r.copyTo(extrinsic.rowRange(0, 3));
        motions[i].copyTo(extrinsic.rowRange(3, 6));
        extrinsics.push_back(extrinsic);
    }
    cout << "total extrinsics = " << extrinsics.size() << endl;
    cout << "total key_points_for_all = " << key_points_for_all.size() << endl;
    cout <<"intrinsic 1 = " << endl << intrinsic << endl;
    bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure_copy);
    cout <<"intrinsic 2 = " << endl << intrinsic << endl;*/


    // ----------------------------------------------------------------------------------
    // PCL at the end of frame
    /*typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr pointcloud( new PointCloud );
    for (size_t i = 0; i < structure.size(); ++i)
    {               
        if(structure[i].x < 150 && structure[i].y < 150 && structure[i].z < 150 && structure[i].x > -150 && structure[i].y > -150 && structure[i].z > -150)
        {
            PointT p;
            p.x = structure[i].x;
            p.y = structure[i].y;
            p.z = structure[i].z;
            p.b = colors[i][0];
            p.g = colors[i][1];
            p.r = colors[i][2];
            pointcloud->points.push_back( p );
        }         
    }
    pointcloud->is_dense = false;
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(pointcloud); 

    int user_data;
    while(!viewer.wasStopped())
    {
        user_data++;
    }*/
    // ----------------------------------------------------------------------------------

    cout << "done" << endl;

    return 0;
}