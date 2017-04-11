#include<iostream>
#include <fstream>
#include <io.h>
#include "conio.h"
#include <sys/time.h>
using namespace std;

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_detection.h"
#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace seeta;

int main(int argc, char* argv[])
{
  // Initialize face detection model
  seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  std::ofstream infoout;
  infoout.open("./infoout/dlib.txt", ofstream::out);
  infoout << "0  empty. " << endl;
  //double t = cv::cvGetTickCount();

  struct timeval starttime, endtime;
  gettimeofday(&starttime, 0);

  int ncount = 0;
  string test_dir = "./infoout/picname.txt";
  ifstream  ist;
  string    st;
  ist.open(test_dir, ios_base::in);
  while (getline(ist, st))
  {
    istringstream iss(st);
    string imgname;
    iss >> imgname;
    imgname = "~/testspeed2/" + imgname;

    cv::Mat img_color = cv::imread(imgname, 1);
    if (img_color.empty())
    {
      cout << "0  empty  " << imgname << endl;
      infoout << "0  empty  " << imgname << endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

    ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
    img_data_gray.data = img_gray.data;

    // Detect faces
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_data_gray);
    int32_t face_num = static_cast<int32_t>(faces.size());

    cout << face_num << "  good  " << imgname << endl;
    infoout << face_num << "  good  " << imgname << endl;
  }//while

  gettimeofday(&endtime, 0);
  double timeuse = 1000000 * (endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  cout    << "Face detection and landmark consuming: " << timeuse / 1000 << "ms" << endl;
  infoout << "Face detection and landmark consuming: " << timeuse / 1000 << "ms" << endl;
  infoout.close();
  return 0;
}


