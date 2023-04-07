#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2d> project(
    const std::vector<cv::Point3d>& points,
    std::vector<double>* weights, 
    cv::Mat rvec, cv::Mat tvec,
    const cv::Mat& intrinsic,
    const std::vector<float>& distortion) {
  std::vector<cv::Point2d> result;

  cv::Mat transformation = cv::Mat(cv::Size(4, 3), CV_64F);
  cv::Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);
  rot_mat.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(transformation(cv::Rect(3, 0, 1, 3)));

  //cv::Mat new_intrinsic = cv::getOptimalNewCameraMatrix(intrinsic, distortion, cv::Size(800, 600), 0);

  cv::Mat pinhole_model = intrinsic * transformation;

  weights->resize(points.size());

  for (int i = 0; i < points.size(); i++) {
    
    cv::Point2d image_point;
    cv::Mat world_point = cv::Mat(cv::Size(1, 4), CV_64F);
    world_point.at<double>(0, 0) = points[i].x;
    world_point.at<double>(1, 0) = points[i].y;
    world_point.at<double>(2, 0) = points[i].z;
    world_point.at<double>(3, 0) = 1;

    cv::Mat projected_homogeneous = pinhole_model * world_point;
    (*weights)[i] = projected_homogeneous.at<double>(2, 0);
    
    image_point.x = projected_homogeneous.at<double>(0, 0) / (*weights)[i];
    image_point.y = projected_homogeneous.at<double>(1, 0) / (*weights)[i];
    //std::cout << "origin point " << points[i] << " to " << image_point << std::endl;
    result.push_back(image_point);
  }

  return result;
}

// Assume world z is 0
std::vector<cv::Point3d> unproject(
    const std::vector<cv::Point2d>& points,
    cv::Mat rvec, cv::Mat tvec,
    cv::Mat& intrinsic,
    const std::vector<float>& distortion) {

  // Reconstruct pinhole-camera model
  cv::Mat transformation = cv::Mat(cv::Size(4, 3), CV_64F);
  cv::Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);
  rot_mat.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(transformation(cv::Rect(3, 0, 1, 3)));
  cv::Mat new_intrinsic = cv::getOptimalNewCameraMatrix(intrinsic, distortion, cv::Size(800, 600), 0);
  cv::Mat pinhole_model = new_intrinsic * transformation;

  std::vector<cv::Point3d> world_points(points.size());
  for (int i = 0; i < points.size(); i++) {
    const double a = pinhole_model.at<double>(2, 0) * points[i].x - pinhole_model.at<double>(0, 0);
    const double b = pinhole_model.at<double>(2, 1) * points[i].x - pinhole_model.at<double>(0, 1);
    const double c = pinhole_model.at<double>(0, 3) - pinhole_model.at<double>(2, 3) * points[i].x;
    const double d = pinhole_model.at<double>(2, 0) * points[i].y - pinhole_model.at<double>(1, 0);
    const double e = pinhole_model.at<double>(2, 1) * points[i].x - pinhole_model.at<double>(1, 1);
    const double f = pinhole_model.at<double>(1, 3) - pinhole_model.at<double>(2, 3) * points[i].y;

    double world_x, world_y;
    world_y = (a * f - c * d) / (a * e - b * d);
    world_x = (c - world_y * b) / a;
    world_points[i].x = world_x; 
    world_points[i].y = world_y;
    world_points[i].z = 0;
    std::cout << "image point " << points[i] << " to " << world_points[i] << std::endl;
  }
  return world_points;
}

int main() {
  const int image_count = 2;
  const cv::Size pattern_size = cv::Size(6, 4);
  const int checker_length = 40; // 40 mm
  const cv::Size image_size = cv::Size(800, 600);

  std::vector<cv::Mat> origin_imgs(image_count);
  std::vector<cv::Mat> gray_imgs(image_count);
  std::vector< std::vector<cv::Vec2f> > imgs_corners(image_count, 
    std::vector<cv::Vec2f>(pattern_size.width * pattern_size.height));
  std::vector< std::vector<cv::Vec3f> > corners_real_pos(image_count, 
    std::vector<cv::Vec3f>(pattern_size.width * pattern_size.height));

  for (int i = 0; i < image_count; i++) {
    // Init real corners pos
    for (int x = 0; x < pattern_size.width; x++) {
      for (int y = 0; y < pattern_size.height; y++) {
        // define top-left corner as (0,0), and the dpeth(z) are all 0.
        const int index = y * pattern_size.width + x;
        corners_real_pos[i][index] = cv::Vec3f(
          x * checker_length, y * checker_length, 0
        );
      }
    }

    // Load image
    std::string file = "C:/Users/woose/calib" + std::to_string(i) + ".jpg";
    std::cout << "Loading file=" << file << std::endl;
    origin_imgs[i] = cv::imread(file);
    cv::resize(origin_imgs[i], origin_imgs[i], image_size);
    cv::cvtColor(origin_imgs[i], gray_imgs[i], cv::COLOR_BGR2GRAY);

    // Find checker patterns
    bool pattern_found 
      = cv::findChessboardCorners(gray_imgs[i], pattern_size, imgs_corners[i]);
    if (pattern_found) {
      cv::cornerSubPix(gray_imgs[i], imgs_corners[i], cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(
          cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    }

    cv::drawChessboardCorners(origin_imgs[i], pattern_size, imgs_corners[i], pattern_found);
    
  }

  cv::Mat camera_mat = cv::Mat(cv::Size(3, 3), CV_32F);
  // camera_mat.at<float>(0, 0) = focal_length_x
  // camera_mat.at<float>(1, 1) = focal_length_y
  // camera_mat.at<float>(0, 2) = principal_point_x
  // camera_mat.at<float>(1, 2) = principal_point_x
  std::vector<float> dist_coeffs;
  std::vector<cv::Mat> rotation_vector(image_count);
  std::vector<cv::Mat> translate_vector(image_count);
  cv::calibrateCamera(corners_real_pos, imgs_corners, image_size,
    camera_mat, dist_coeffs, rotation_vector, translate_vector);
  
  std::cout << "TEST 3D -> 2D" << std::endl;
  for (int i = 0; i < image_count; i++) {
    std::vector<cv::Point3d> world_points(corners_real_pos[i].size()); 
    for (int j = 0; j < world_points.size(); j++) {
      world_points[j].x = corners_real_pos[i][j][0];
      world_points[j].y = corners_real_pos[i][j][1];
      world_points[j].z = corners_real_pos[i][j][2];
    }
    std::vector<double> weights;
    std::vector<cv::Point2d> image_points = project(
      world_points, &weights, rotation_vector[i], translate_vector[i],
      camera_mat, dist_coeffs);
    for (int j = 0; j < image_points.size(); j++) {
      //std::cout << "COMPARE " << imgs_corners[i][j] << " -> " << image_points[j] << std::endl;
      cv::circle(origin_imgs[i], image_points[j], 4, cv::Scalar(0, 0, 0), 10);
    }
    cv::imshow(std::to_string(i), origin_imgs[i]);
  }

  // Test 2D -> 3D
  std::cout << "TEST 2D -> 3D" << std::endl;
  for (int i = 0; i < image_count; i++) {
    std::vector<cv::Point2d> image_points(imgs_corners[i].size());
    for (int j = 0; j < image_points.size(); j++) {
      image_points[j].x = imgs_corners[i][j][0];
      image_points[j].y = imgs_corners[i][j][1];
    }
    std::vector<cv::Point3d> world_points = unproject(
      image_points, rotation_vector[i], translate_vector[i], camera_mat, dist_coeffs
    );
  }
  
  cv::waitKey(0);
  return 0;
}
