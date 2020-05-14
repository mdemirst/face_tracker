#include <stdio.h>

#include <chrono>
#include <deque>
#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

class FaceTracker {
 public:
  FaceTracker()
      : kf_(6 /*state size*/, 4 /* measurement size */, 0 /* control size */) {
    InitKf();
    InitDetector();
  }

  void DetectAndTrack() {
    capture_.open(-1);  // open webcam
    assert(capture_.isOpened());

    Mat frame;
    std::vector<Rect> faces;

    while (capture_.read(frame)) {
      assert(!frame.empty());

      DetectFaces(frame, faces);

      TrackFaces(faces);

      DisplayTrajectories(frame);

      DisplayTrackedFace(frame);

      int c = waitKey(10);
      if ((char)c == 27) break;
    }
  }

 private:
  void InitKf() {
    last_measurement_time_ = Now();

    // measurement matrix
    // 1 0 0 0 0 0
    // 0 1 0 0 0 0
    // 0 0 0 0 1 0
    // 0 0 0 0 0 1
    kf_.measurementMatrix.setTo(Scalar(0.0));
    kf_.measurementMatrix.at<float>(0, 0) = 1.0;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0;
    kf_.measurementMatrix.at<float>(2, 4) = 1.0;
    kf_.measurementMatrix.at<float>(3, 5) = 1.0;

    // process noise
    // n_x 0 0 0 0 0
    // 0 n_x 0 0 0 0
    // 0 0 n_v 0 0 0
    // 0 0 0 n_v 0 0
    // 0 0 0 0 n_w 0
    // 0 0 0 0 0 n_w
    kf_.processNoiseCov.at<float>(0, 0) = 1e-1;  // x variance
    kf_.processNoiseCov.at<float>(1, 1) = 1e-1;  // y variance
    kf_.processNoiseCov.at<float>(2, 2) = 1e-2;  // vx variance
    kf_.processNoiseCov.at<float>(3, 3) = 1e-2;  // vy variance
    kf_.processNoiseCov.at<float>(4, 4) = 1e-1;  // width variance
    kf_.processNoiseCov.at<float>(5, 5) = 1e-1;  // height variance

    // measurement noise
    // n_x 0
    // 0 n_y
    setIdentity(kf_.measurementNoiseCov, Scalar::all(1e-2));  // x,y variance

    // set initial state
    randn(kf_.statePost, Scalar::all(0), Scalar::all(0.1));
    randn(kf_.statePre, Scalar::all(0), Scalar::all(0.1));

    // set initial covariance
    setIdentity(kf_.errorCovPost, Scalar::all(10.0));
    setIdentity(kf_.errorCovPre, Scalar::all(10.0));
  }

  void CalcTransitionMatrix(double dt) {
    // transition matrix
    // 1 0 dt 0 0 0
    // 0 1 0 dt 0 0
    // 0 0 1 0 0 0
    // 0 0 0 1 0 0
    // 0 0 0 0 1 0
    // 0 0 0 0 0 1
    setIdentity(kf_.transitionMatrix, Scalar::all(1.0));
    kf_.transitionMatrix.at<float>(0, 2) = dt;
    kf_.transitionMatrix.at<float>(1, 3) = dt;
  }

  void InitDetector() {
    assert(face_cascade_.load("../haarcascade_frontalface_alt.xml"));
  }

  int DetectFaces(Mat frame, std::vector<Rect>& faces) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    face_cascade_.detectMultiScale(frame_gray, faces, 1.1, 2,
                                   0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    for (int i = 0; i < faces.size(); i++) {
      Point center(faces[i].x + faces[i].width / 2,
                   faces[i].y + faces[i].height / 2);
      ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0,
              0, 360, Scalar(255, 0, 255), 4, 8, 0);
    }

    return faces.size();
  }

  void TrackFaces(const std::vector<Rect>& faces) {
    double now = Now();
    CalcTransitionMatrix(now - last_measurement_time_);

    int matched_face = MatchTrackedFace(faces);

    if (matched_face != -1) {
      auto& face = faces[matched_face];
      double face_x = face.x + face.width / 2;
      double face_y = face.y + face.height / 2;
      Mat measurement =
          (Mat_<float>(4, 1) << face_x, face_y, face.width, face.height);

      trajectory_measurement_.push_back(Point(face_x, face_y));
      while (trajectory_measurement_.size() > 30)
        trajectory_measurement_.pop_front();

      kf_.correct(measurement);
      last_measurement_time_ = now;
    } else {  // no matched face found
              // gradually decrease velocity to zero
      kf_.statePost.at<float>(2, 0) *= 0.9;
      kf_.statePost.at<float>(3, 0) *= 0.9;
    }
  }

  // returns index of the most similar face to tracked face
  // returns -1 if not match
  int MatchTrackedFace(const std::vector<Rect>& faces) {
    int min_id = -1;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < faces.size(); i++) {
      double dist = CalcSimilarity(faces[i]);

      if (dist < min_dist) {
        min_dist = dist;
        min_id = i;
      }
    }

    const double kSimilarityThreshold = 5000.0;
    const double kUncertaintyThreshold = 10.0;

    std::cout << min_dist << std::endl;
    if (min_dist < kSimilarityThreshold ||
        hypot(kf_.errorCovPre.at<float>(0, 0),
              kf_.errorCovPre.at<float>(1, 1)) > kUncertaintyThreshold)
      return min_id;
    else
      return -1;
  }

  double CalcSimilarity(const Rect& face) {
    double face_x = face.x + face.width / 2.0;
    double face_y = face.y + face.height / 2.0;

    const double kFacePositionWeight = 1.0;
    const double kFaceSizeWeight = 3.0;
    return sqrt(
        kFacePositionWeight * pow(kf_.statePre.at<float>(0, 0) - face_x, 2) +
        kFacePositionWeight * pow(kf_.statePre.at<float>(1, 0) - face_y, 2) +
        kFaceSizeWeight * pow(kf_.statePre.at<float>(4, 0) - face.width, 2) +
        kFaceSizeWeight * pow(kf_.statePre.at<float>(5, 0) - face.height, 2));
  }

  void DisplayTrackedFace(Mat frame) {
    double now = Now();
    CalcTransitionMatrix(now - last_measurement_time_);

    Mat predicted_state = kf_.predict();

    Rect bbox(
        predicted_state.at<float>(0, 0) - predicted_state.at<float>(4, 0) / 2,
        predicted_state.at<float>(1, 0) - predicted_state.at<float>(5, 0) / 2,
        predicted_state.at<float>(4, 0), predicted_state.at<float>(5, 0));
    rectangle(frame, bbox, Scalar(0, 255, 0));

    trajectory_filtered_.push_back(Point(predicted_state.at<float>(0, 0),
                                         predicted_state.at<float>(1, 0)));
    while (trajectory_filtered_.size() > 30) trajectory_filtered_.pop_front();

    imshow("Face tracking", frame);
  }

  double Now() {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
               .count() /
           1000.0;
  }

  void DisplayTrajectories(Mat frame) {
    for (int i = 0; i < (int)trajectory_measurement_.size() - 1; i++)
      line(frame, trajectory_measurement_[i], trajectory_measurement_[i + 1],
           Scalar(0, 0, 255), 2, 8);

    Rect frame_rect(Point(), frame.size());

    for (int i = 0; i < (int)trajectory_filtered_.size() - 1; i++) {
      if (frame_rect.contains(trajectory_filtered_[i]) &&
          frame_rect.contains(trajectory_filtered_[i + 1]))
        line(frame, trajectory_filtered_[i], trajectory_filtered_[i + 1],
             Scalar(0, 255, 0), 2, 8);
    }
  }

  KalmanFilter kf_;
  double last_measurement_time_;
  CascadeClassifier face_cascade_;
  VideoCapture capture_;
  std::deque<Point> trajectory_measurement_;
  std::deque<Point> trajectory_filtered_;
};

int main(void) {
  FaceTracker face_tracker;
  face_tracker.DetectAndTrack();
  return 0;
}
