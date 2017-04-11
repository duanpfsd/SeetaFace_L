
#include<iostream>
#include <fstream>
//#include <io.h>
#include "conio.h"
using namespace std;

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace seeta;

int main(int argc, char* argv[]) {
  // Initialize face detection model
  seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");

  // Initialize face Identification model 
  FaceIdentification face_recognizer("seeta_fr_v1.0.bin");
  std::string test_dir =  "./images/test/";
  string gallery_dir = "./images/";

  while (1)
  {
	  printf("Input address of picture: ");
	  char str2[100] = { 0 };
	  scanf("%s", str2);
	  printf("Picture address£º %s\n", str2);
	  string imgpath2(str2);
	  cv::Mat probe_img_color = cv::imread(test_dir + imgpath2, 1);
	  if (probe_img_color.empty())
	  {
		  continue;
	  }
    cv::Mat probe_img_gray;
	  cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

    cv::imshow("image2", probe_img_color);
	  cvWaitKey(0);


	  ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
	  probe_img_data_color.data = probe_img_color.data;

	  ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
	  probe_img_data_gray.data = probe_img_gray.data;

	  // Detect faces
	  double t = cvGetTickCount();
	  std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
	  t = cvGetTickCount() - t;
	  cout << "face detetion consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
	  int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

	  //show detect results

    for (int i = 0; i < probe_faces.size(); i++)
	  {
		  cv::Rect rc;
		  rc.x = probe_faces[i].bbox.x;
		  rc.y = probe_faces[i].bbox.y;
		  rc.width = probe_faces[i].bbox.width;
		  rc.height = probe_faces[i].bbox.height;
		  cv::rectangle(probe_img_color, rc, cv::Scalar(255, 0, 0));
      cout << i << "  " << rc.width << "  " << rc.height << endl;
	  }
	  imshow("image2", probe_img_color);
	  cvWaitKey(0);

    if (probe_face_num == 0)
	  {
		  std::cout << "Faces are not detected.";
		  //   return 0;
		  continue;
	  }
	  // Detect 5 facial landmarks
	  seeta::FacialLandmark probe_points[10][5];
    int probe_size = (probe_faces.size() < 10) ? probe_faces.size() : 10;
	  t = cvGetTickCount();
    for (int i = 0; i < probe_size; i++)
    {
      point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[i], probe_points[i]);
    }
	  t = cvGetTickCount() - t;
	  cout << "face alignment consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;

    for (int i = 0; i < probe_size; i++)
	  {
      for (int j = 0; j < 5; j++)
      {
        cv::circle(probe_img_color, cv::Point(probe_points[i][j].x, probe_points[i][j].y), 7,
          cvScalar((0), (255), (0), 0));
      }
	  }
	  imshow("image2", probe_img_color);
    cvWaitKey(0);

    // cv::imwrite("probe_point_result.jpg", probe_img_color);

	  // Extract face identity feature
	  float probe_fea[10][2048];
	  t = cvGetTickCount();
    for (int i = 0; i < probe_size; i++)
    {
      face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points[i], probe_fea[i]);
    }
    t = cvGetTickCount() - t;
	  cout << "face ExtractFeatureWithCrop consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
    cout << endl;
    for (int i = 0; i < 2048; i++)
    {
      //cout << probe_fea[0][i] << " ";
    }
    cout << endl;

	  // Caculate similarity of two faces
    t = cvGetTickCount();
    string name_chinese, name_pinyin;
    int ncount = 0;
    float gallery_fea[2048] = { 0.f };
    float sim = -1.0f;
    float maxsim = -1.0f;
    string likeliname;

    int linecount = 0, maxindex = -1;
    ifstream  istreamgallery;
    string    strgallery;
    istreamgallery.open(gallery_dir+"/gallery_fea_v1/gallery_features.txt", ios_base::in);
    while (getline(istreamgallery, strgallery))
    {
      istringstream iss(strgallery);
      iss >> name_pinyin;
      ncount = 0;
      while (iss >> gallery_fea[ncount])
      {
        ncount++;
      }
      // calculate sim
      for (int i = 0; i < probe_size; i++)
      {
        sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea[i]);
        if (sim > maxsim)
        {
          maxsim = sim;
          maxindex = linecount;
          likeliname = name_pinyin;
        }
      }
      linecount++;
    }
    if (linecount == 0) cout << "No gallery input!!" << endl;
    t = cvGetTickCount() - t;
    cout << "Face computing similarity and identifying consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
    if (maxsim < 0.4) {
      likeliname = "not matched with gallery.";
    }
    std::cout << maxindex<<"  ÄãÊÇ " << likeliname << ". Cosine similarity:" << maxsim << endl;
	  _getch();

  }
  return 0;
}


