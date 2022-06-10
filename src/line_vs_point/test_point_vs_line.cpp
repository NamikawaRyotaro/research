#include <iostream>
#include <cmath>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <limits>
#include <map>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <istream>
#include <ostream>
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>
#include <opencv2/calib3d.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>

#include "../cerial/eigen_wrapper.hpp"
#include "../calib3d/calib.h"

const double PI = 3.141592653589;

namespace Util
{
  std::string toString(const Eigen::Vector3d &vect)
  {
    return "[" + std::to_string(vect[0]) + "," + std::to_string(vect[1]) + "," + std::to_string(vect[2]) + "]";
  }
  Eigen::Matrix3d getBaseConversionMatrix(const std::vector<Eigen::Vector3d> &src, const std::vector<Eigen::Vector3d> &dst)
  {
    Eigen::Matrix3d A = Eigen::Matrix3d::Ones();
    A.row(0) = src[0].transpose();
    A.row(1) = src[1].transpose();
    A.row(2) = src[2].transpose();

    Eigen::Matrix3d B = Eigen::Matrix3d::Ones();
    B.row(0) = dst[0].transpose();
    B.row(1) = dst[1].transpose();
    B.row(2) = dst[2].transpose();

    Eigen::FullPivLU<Eigen::Matrix3d> rank_a(A);
    Eigen::FullPivLU<Eigen::Matrix3d> rank_b(B);
    assert(rank_a.rank() == 3 && rank_b.rank() == 3);

    Eigen::Matrix3d P = B * A.inverse();
    return P;
  }

  Eigen::Matrix3d GetLookAtRotation(const Eigen::Vector3d target, const Eigen::Vector3d pose, const Eigen::Vector3d up)
  {
    Eigen::Vector3d z = (target - pose).normalized();
    Eigen::Vector3d x = z.cross(up).normalized();
    Eigen::Vector3d y = z.cross(x).normalized();

    Eigen::Matrix3d rot;
    rot(0, 0) = x(0);
    rot(0, 1) = y(0);
    rot(0, 2) = z(0);
    rot(1, 0) = x(1);
    rot(1, 1) = y(1);
    rot(1, 2) = z(1);
    rot(2, 0) = x(2);
    rot(2, 1) = y(2);
    rot(2, 2) = z(2);

    return rot.transpose();
  }

  struct CameraParameter
  {

    Eigen::Matrix3d matrix;
    template <class Archive>
    void serialize(Archive &archive)
    {
      archive(CEREAL_NVP(matrix));
    }
  };

  template <typename Scalar, int row, int col>
  class Matrix
  {
  private:
    cv::Mat mat;
    Eigen::Matrix<Scalar, row, col> eigen;

  public:
    auto getCvMat()
    {
      return mat.clone();
    }
    auto getEigen()
    {
      return eigen.
    }
  }
}

int main(int argc, char *argv[])
{

  double f0 = 0.3;

  std::ofstream line_error_txt("./result/line_error.txt", std::ios::trunc);
  std::ofstream point_error_txt("./result/point_error.txt", std::ios::trunc);

  if (!line_error_txt || !point_error_txt)
  {
    std::cout << "ファイルが開けませんでした。" << std::endl;
    return 0;
  }

  Util::CameraParameter cp;
  {
    std::ifstream camera_param_json("./resource/camera_pram.json");
    if (!camera_param_json)
    {
      std::cout << "ファイルが開けませんでした。" << std::endl;
      return 0;
    }
    cereal::JSONInputArchive archive(camera_param_json);
    archive(cp);
  }

  Eigen::Matrix3d cameraMat;
  cameraMat = cp.matrix;
  cv::Mat cameraMat_cv;
  cv::eigen2cv(cameraMat, cameraMat_cv);

  double fx = cameraMat(0, 0);
  double fy = cameraMat(1, 1);
  double cx = cameraMat(0, 2);
  double cy = cameraMat(1, 2);
  int imageX = 2 * cx;
  int imageY = 2 * cy;

  // std::cout << cameraMat_cv << std::endl;

  cv::Mat src(cv::Size(imageX, imageY), CV_8UC1, cv::Scalar(0));

  Eigen::Vector3d lu = cameraMat.inverse() * Eigen::Vector3d(0, 0, 1);
  Eigen::Vector3d ll = cameraMat.inverse() * Eigen::Vector3d(0, imageY, 1);
  Eigen::Vector3d ru = cameraMat.inverse() * Eigen::Vector3d(imageX, 0, 1);
  Eigen::Vector3d rl = cameraMat.inverse() * Eigen::Vector3d(imageX, imageY, 1);

  std::vector<Eigen::Vector3d> corners{lu, ll, ru, rl};
  std::vector<cv::Point3f> corners_cv;
  for (auto &&x : corners)
  {
    x = f0 * (x / x(2));
    corners_cv.push_back(cv::Point3f(x(0), x(1), x(2)));
  }

  std::vector<cv::Point> p{cv::Point(320, 45), cv::Point(200, 140), cv::Point(240, 300), cv::Point(400, 300), cv::Point(440, 140)};
  // cv::line(src, cv::Point(100, 100), cv::Point(260, 260), cv::Scalar(255), 10, 4);
  // cv::line(src, cv::Point(260, 100), cv::Point(100, 260), cv::Scalar(255), 10, 4);
  // cv::line(src, cv::Point(50, 100), cv::Point(200, 100), cv::Scalar(255), 10, 4);
  fillConvexPoly(src, p.data(), p.size(), cv::Scalar(255), cv::LINE_AA);

  Eigen::Matrix<double, 3, 4> identify3x4;
  identify3x4 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0;

  double pos_x, pos_y, pos_z;
  while (!std::cin.eof())
  {

    std::cin >> pos_x >> pos_y >> pos_z;
    std::stringstream camera_pos_stream;
    std::stringstream camera_pos_csv;
    camera_pos_stream << "[" << pos_x << "," << pos_y << "," << pos_z << "]";
    camera_pos_csv << pos_x << "," << pos_y << "," << pos_z << ",";
    auto target = Eigen::Vector3d(0, 0, f0);
    auto pose = Eigen::Vector3d(pos_x, pos_y, pos_z);
    auto up = Eigen::Vector3d(0, -1, 0);
    auto pre_rad = Util::GetLookAtRotation(target, pose, up);
    auto tv = -pre_rad * pose;

    // std::cout << tv << std::endl;

    // std::cout << pre_rod << std::endl;
    // cv::Mat rod = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat rod;
    cv::Mat rad;

    cv::eigen2cv(pre_rad, rad);
    std::vector<Eigen::Vector3d> corners_dist;
    for (auto &&c : corners)
    {
      Eigen::Vector3d xyz = pre_rad * c + tv;
      Eigen::Vector3d uv = cameraMat * xyz;
      uv = uv / uv(2);
      corners_dist.push_back(Eigen::Vector3d(uv(0), uv(1), 1));
    }

    std::vector<cv::Point2f> pts_src, pts_dst, pts_prj;
    // std::cout << cameraMat << std::endl;
    // std::cout << cameraMat.inverse() << std::endl;
    for (const Eigen::Vector3d &c : corners)
    {
      Eigen::Vector3d point2d = cameraMat * c;
      point2d = point2d / point2d(2);
      pts_src.push_back(cv::Point2f(point2d(0), point2d(1)));
    }
    for (const Eigen::Vector3d &c : corners_dist)
    {
      pts_dst.push_back(cv::Point2f(c[0], c[1]));
    }

    // std::cout << pts_src << std::endl;
    // std::cout << pts_dst << std::endl;
    // std::cout << pts_dst << std::endl;

    cv::Mat h = cv::getPerspectiveTransform(pts_src, pts_dst);
    Eigen::MatrixXd perspective;
    cv::cv2eigen(h, perspective);
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::warpPerspective(src, dst, h, dst.size(), cv::INTER_AREA);
    // std::cout << h << std::endl;
    // std::cout << perspective << std::endl;
    auto algorithm = cv::xfeatures2d::SURF::create();
    // (4)extract SURF
    std::vector<cv::KeyPoint> src_kps, dst_kps;
    std::vector<cv::Vec4f> src_klines, dst_klines;
    // std::vector<float> desc_vec;
    // cv::calc_surf(src, cv::Mat(), kp_vec, desc_vec);
    // std::vector<cv::KeyPoint> keypoint1, keypoint2;
    cv::Mat src_dscriptor, dst_discriptor;
    algorithm->detect(src, src_kps);
    algorithm->compute(src, src_kps, src_dscriptor);
    algorithm->detect(dst, dst_kps);
    algorithm->compute(dst, dst_kps, dst_discriptor);

    // (5)draw keypoints
    // cout << "Image Keypoints: " << kp_vec.size() << endl;

    // std::vector<cv::KeyPoint>::iterator it = kp_vec.begin(), it_end = kp_vec.end();
    cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector();
    fld->detect(src, src_klines);
    fld->detect(dst, dst_klines);

    using Line = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
    std::vector<Eigen::Vector3d> src_pts;
    std::vector<Line> src_lines;
    std::vector<Eigen::Vector3d> dst_pts;
    std::vector<Line> dst_lines;
    std::vector<Eigen::Vector3d> predict_pts;
    std::vector<Line> predict_lines;
    std::map<size_t, size_t> match_pt_indexs;
    std::map<size_t, size_t> match_line_indexs;

    cv::Mat src_kp, dst_kp, src_line, dst_line;
    src.copyTo(src_kp);
    src.copyTo(src_line);
    dst.copyTo(dst_kp);
    dst.copyTo(dst_line);

    for (auto kp : src_kps)
    {
      src_pts.push_back(Eigen::Vector3d(kp.pt.x, kp.pt.y, 1));
      circle(src_kp, cv::Point(kp.pt.x, kp.pt.y),
             cv::saturate_cast<int>(kp.size) / 4, cv::Scalar(128), 2);
    }
    for (auto line : src_klines)
    {
      cv::Point pt1 = cv::Point2f(line[0], line[1]);
      cv::Point pt2 = cv::Point2f(line[2], line[3]);
      src_lines.push_back(std::make_pair(Eigen::Vector3d(line[0], line[1], 1), Eigen::Vector3d(line[2], line[3], 1)));
      cv::line(src_line, pt1, pt2, cv::Scalar(128), 2);
    }

    for (auto kp : dst_kps)
    {
      // std::cout << "dst_kp : (" << kp.pt.x << "," << kp.pt.y << ")" << std::endl;
      dst_pts.push_back(Eigen::Vector3d(kp.pt.x, kp.pt.y, 1));
      circle(dst_kp, cv::Point(kp.pt.x, kp.pt.y),
             cv::saturate_cast<int>(kp.size) / 4, cv::Scalar(128), 2);
    }

    for (auto line : dst_klines)
    {
      cv::Point pt1 = cv::Point2f(line[0], line[1]);
      cv::Point pt2 = cv::Point2f(line[2], line[3]);
      dst_lines.push_back(std::make_pair(Eigen::Vector3d(line[0], line[1], 1), Eigen::Vector3d(line[2], line[3], 1)));
      cv::line(dst_line, pt1, pt2, cv::Scalar(128), 2);
    }

    // std::cout << h << std::endl;

    for (auto origin : src_pts)
    {
      Eigen::Vector3d predict = perspective * origin;
      predict_pts.push_back(predict / predict(2));
      // circle(dst, cv::Point(predict_pts.back()[0], predict_pts.back()[1]),
      //        2, cv::Scalar(128), 2);
      // std::cout << "src(" << origin[0] << ", " << origin[1] << ") -> (" << predict[0] << "," << predict[1] << ")" << std::endl;
    }

    for (auto line : src_lines)
    {
      Eigen::Vector3d origin1 = line.first;
      Eigen::Vector3d origin2 = line.second;
      Eigen::Vector3d predict1 = perspective * origin1;
      Eigen::Vector3d predict2 = perspective * origin2;
      predict1 = predict1 / predict1(2);
      predict2 = predict2 / predict2(2);
      predict_lines.push_back(std::make_pair(predict1, predict2));
      auto pt1 = cv::Point(predict1[0], predict1[1]);
      auto pt2 = cv::Point(predict2[0], predict2[1]);
      // cv::line(dst, pt1, pt2, cv::Scalar(255), 2);
    }

    std::vector<double> matchd_pt_distance(src_pts.size(), std::numeric_limits<double>::infinity());
    std::vector<double> matchd_line_distance(src_lines.size(), std::numeric_limits<double>::infinity());

    for (int j = 0; j < dst_pts.size(); j++)
    {
      auto pt = dst_pts[j];
      auto match_pt = Eigen::Vector3d(0, 0, 0);
      double distance = std::numeric_limits<double>::infinity();
      int index = -1;

      for (int i = 0; i < predict_pts.size(); i++)
      {
        auto p = predict_pts[i];
        auto s = src_pts[i];
        double d = std::pow(pt[0] - p[0], 2) + std::pow(pt[1] - p[1], 2);
        if (distance > d)
        {
          distance = d;
          match_pt = p;
          index = i;
        }
      }
      if (index < 0)
        continue;
      if (distance > matchd_pt_distance[index]) //|| 10 < distance)
      {
        continue;
      }

      matchd_pt_distance[index] = distance;
      auto s = src_pts[index];
      match_pt_indexs.insert_or_assign(index, j);
    }

    // for (auto [key,val] : match_pt_indexs){
    //   std::cout << "matched(" << key << "," << val << ") -> (" << Util::toString(src_pts[key]) << ")(" << Util::toString(dst_pts[val]) << std::endl;
    // }

    //* std::cout << "---------------------line-------------------------------------" <<std::endl;
    for (int j = 0; j < dst_lines.size(); j++)
    {
      auto pt1 = dst_lines[j].first;
      auto pt2 = dst_lines[j].second;
      auto match_pt = Eigen::Vector3d(0, 0, 0);
      double distance = std::numeric_limits<double>::infinity();
      int index = -1;

      for (int i = 0; i < predict_lines.size(); i++)
      {
        auto p1 = predict_lines[i].first;
        auto p2 = predict_lines[i].second;
        double d1 = std::pow(pt1[0] - p1[0], 2) + std::pow(pt1[1] - p1[1], 2);
        double d2 = std::pow(pt2[0] - p1[0], 2) + std::pow(pt2[1] - p1[1], 2);
        double d3 = std::pow(pt1[0] - p2[0], 2) + std::pow(pt1[1] - p2[1], 2);
        double d4 = std::pow(pt2[0] - p2[0], 2) + std::pow(pt2[1] - p2[1], 2);
        auto d = std::min(d1, d2) + std::min(d3, d4);
        if (distance > d)
        {
          distance = d;
          match_pt = p1;
          index = i;
        }
      }
      if (index < 0)
        continue;
      if (distance > matchd_line_distance[index])
        continue;

      matchd_line_distance[index] = distance;
      auto s = src_lines[index].first;
      match_line_indexs.insert_or_assign(index, j);
    }

    //*/

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> imagePredictionPoints;
    cv::Mat distortion = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    cv::Mat rvec, tvec, rmat;

    for (const auto &[si, di] : match_pt_indexs)
    {
      Eigen::Vector3d op = cameraMat.inverse() * src_pts[si];
      op = (f0 / op(2)) * op;
      objectPoints.push_back(cv::Point3f(op(0), op(1), op(2)));
      imagePoints.push_back(cv::Point2f(dst_pts[di](0), dst_pts[di](1)));
      imagePredictionPoints.push_back(cv::Point2f(predict_pts[si](0), predict_pts[si](1)));
    }

    std::vector<cv::Point2f> pts_prd_kp;
    std::vector<cv::Point2f> pts_prd_line;
    Eigen::Matrix<double, 3, 3> r;
    Eigen::Matrix<double, 3, 1> t;
    try
    {
      cv::solvePnP(objectPoints, imagePoints, cameraMat_cv, distortion, rvec, tvec);
      cv::Rodrigues(rvec, rmat);
      cv::cv2eigen(rmat, r);
      cv::cv2eigen(tvec, t);

      // std::cout << "predict rmat:" << std::endl;
      // std::cout << r << std::endl;
      // std::cout << "predict camera forward:" << std::endl;
      // std::cout << r * target + t << std::endl;
      point_error_txt << camera_pos_csv.str() << (pose + r.transpose() * t).norm() << std::endl;

      std::vector<Eigen::Vector3d> corners_pred;
      for (auto &&c : corners)
      {
        Eigen::Vector3d xyz = r * c + t;
        Eigen::Vector3d uv = cameraMat * xyz;
        uv = uv / uv(2);
        corners_pred.push_back(Eigen::Vector3d(uv(0), uv(1), 1));
      }

      for (const Eigen::Vector3d &c : corners_pred)
      {
        pts_prd_kp.push_back(cv::Point2f(c[0], c[1]));
      }
    }
    catch (...)
    {
      std::vector<cv::Point2f> pts_prd_kp(pts_src);
    }
    // cv::solvePnP(objectPoints, imagePredictionPoints, cameraMat_cv, distortion, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    cv::Mat h_p = cv::getPerspectiveTransform(pts_src, pts_prd_kp);
    Eigen::Matrix3d perspective_p;
    cv::cv2eigen(h_p, perspective_p);
    cv::Mat prd_kp = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::warpPerspective(src, prd_kp, h_p, prd_kp.size());

    cv::Mat dif_kp = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::absdiff(dst, prd_kp, dif_kp);
    // for (size_t i = 0; i < 3; i++)
    // {
    //   Eigen::Vector3d op;
    //   op << objectPoints[i].x, objectPoints[i].y,objectPoints[i].z;
    //   Eigen::Vector3d p = cameraMat*((r*op)+t);
    //   Eigen::Vector3d po = cameraMat * op;
    //   // std::cout << "[" << imagePoints[i].x << "," << imagePoints[i].y << "] -> [" << p[0] / p[2] << "," << p[1] / p[2] << "]" << std::endl;
    //   // circle(src, cv::Point(po[0] / po[2], po[1] / po[2]),
    //   //        4, cv::Scalar(128), 2);
    //   // circle(dst, cv::Point(imagePoints[i].x ,imagePoints[i].y),
    //   //        4, cv::Scalar(128), 2);
    // }

    //*
    Eigen::MatrixXd nV(3, match_line_indexs.size()); //n
    Eigen::MatrixXd dV(3, match_line_indexs.size()); //predict
    Eigen::MatrixXd P(3, match_line_indexs.size());  //original pos

    int tmp = 0;
    Eigen::Matrix3d camera_base;
    camera_base << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    Eigen::Matrix3d world_base;
    world_base << 1, 0, 0, 0, 0, 1, 0, -1, 0;
    for (const auto &[si, di] : match_line_indexs)
    {
      auto s = src_lines[si];
      auto d = dst_lines[di];
      // auto direction_s = (camera_base * s.second - camera_base * s.first).normalized();
      auto direction_s = (cameraMat.inverse() * s.second - cameraMat.inverse() * s.first).normalized();
      // Eigen::Vector3d ss = camera_base * s.first;

      // auto direction_d = (camera_base * d.second - camera_base * d.first).normalized();
      auto direction_d = (cameraMat.inverse() * d.second - cameraMat.inverse() * d.first).normalized();
      Eigen::Vector3d ss = s.first;
      Eigen::Vector3d sp = f0 * cameraMat.inverse() * ss;
      Eigen::Vector3d dd = d.first;
      Eigen::Vector3d dp = cameraMat.inverse() * dd;
      auto direction_n = direction_d.cross(dp).normalized();
      nV(0, tmp) = direction_n(0);
      nV(1, tmp) = direction_n(1);
      nV(2, tmp) = direction_n(2);
      dV(0, tmp) = direction_s(0);
      dV(1, tmp) = direction_s(1);
      dV(2, tmp) = direction_s(2);
      P(0, tmp) = sp(0);
      P(1, tmp) = sp(1);
      P(2, tmp) = sp(2);

      tmp++;
      // std::cout << s.first(0) << std::endl;
    }

    // // std::cout << typeid(P).name() << std::endl;
    // std::cout << cameraMat * P << std::endl;
    // // std::cout << typeid(nV).name() << std::endl;
    // std::cout << nV << std::endl;
    // // std::cout << typeid(dV).name() << std::endl;
    // std::cout << dV << std::endl;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd T = Eigen::VectorXd::Zero(3);

    // try
    // {
    double err = calc_motion(nV, dV, P, R, T, std::numeric_limits<double>::epsilon());
    line_error_txt << camera_pos_csv.str() << (pose + R.transpose() * T).norm() << std::endl;
    // // }
    // catch (...)
    // {
    //   continue;
    // }

    // std::cout << "predict rmat:" << std::endl;
    // std::cout << R << std::endl;
    // std::cout << "predict camera forward:" << std::endl;
    // std::cout << R * target + T << std::endl;

    for (auto &&c : corners)
    {
      Eigen::Vector3d xyz = R * c + T;
      Eigen::Vector3d uv = cameraMat * xyz;
      uv = uv / uv(2);
      pts_prd_line.push_back(cv::Point2f(uv(0), uv(1)));
    }

    cv::Mat h_pl = cv::getPerspectiveTransform(pts_src, pts_prd_line);
    Eigen::Matrix3d perspective_pl;
    cv::cv2eigen(h_pl, perspective_pl);
    cv::Mat prd_line = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::warpPerspective(src, prd_line, h_pl, prd_line.size());

    cv::Mat dif_line = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::absdiff(dst, prd_line, dif_line);

    // cv::imshow("src", src);
    // cv::imshow("dst", dst);
    // cv::imshow("src_line", src_line);
    // cv::imshow("dst_line", dst_line);
    // cv::imshow("src_kp", src_kp);
    // cv::imshow("dst_kp", dst_kp);
    // cv::imshow("prd_kp", prd_kp);
    // cv::imshow("dif_kp", dif_kp);
    // cv::imshow("prd_line", prd_line);
    // cv::imshow("dif_line", dif_line);

    std::cout << "original camera pos" << camera_pos_stream.str() << std::endl;
    std::cout << "camera pos error estimated by point:" << (pose + r.transpose() * t).norm() << std::endl;
    std::cout << "camera pos error estimated by line:" << (pose + R.transpose() * T).norm() << std::endl;
    std::cout << "camera pos estimated by point:" << -r.transpose() * t << std::endl;
    std::cout << "camera pos estimated by line:" << -R.transpose() * T << std::endl;

    cv::imwrite("./result/src" + camera_pos_stream.str() + ".png", src);
    cv::imwrite("./result/dst" + camera_pos_stream.str() + ".png", dst);
    cv::imwrite("./result/src_line" + camera_pos_stream.str() + ".png", src_line);
    cv::imwrite("./result/dst_line" + camera_pos_stream.str() + ".png", dst_line);
    cv::imwrite("./result/src_kp" + camera_pos_stream.str() + ".png", src_kp);
    cv::imwrite("./result/dst_kp" + camera_pos_stream.str() + ".png", dst_kp);
    cv::imwrite("./result/prd_kp" + camera_pos_stream.str() + ".png", prd_kp);
    cv::imwrite("./result/dif_kp" + camera_pos_stream.str() + ".png", dif_kp);
    cv::imwrite("./result/prd_line" + camera_pos_stream.str() + ".png", prd_line);
    cv::imwrite("./result/dif_line" + camera_pos_stream.str() + ".png", dif_line);
  }
  // cv::waitKey(0);
  //    }
  // }

  // std::cout << R << std::endl;
  // std::cout << T << std::endl;

  // Eigen::VectorXd z = Eigen::VectorXd::Zero(3);
  // z(2) = 1;
  // std::cout << R*T << std::endl;
  //*/

  // cv::imshow("src", src);
  // cv::imshow("dst", dst);
  // cv::imshow("prd", prd);
  // cv::imshow("dif", dif);
  // cv::waitKey(0);
  return 0;
}
