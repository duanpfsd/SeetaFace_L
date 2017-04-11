/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/
/*
* This file is to store feature of the chosen gallery pictures.  
*
*/

#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include "conio.h"
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

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

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif

// Read facial information and store in files 
void readFaceFeature(string path)
{
  // The *.txt file to store Face Information.
  std::ofstream faceInfoOut;
  faceInfoOut.open("GalleryFaceInfo.txt", ofstream::out);

  // Some vars of  folders and picture files.
  long hFolder = 0;
  struct _finddata_t folderInfo;
  string pathName;
  long hFile = 0;
  struct _finddata_t fileInfo;
  string subpathName;

  // Read picture files and store Face Information.
  if ((hFolder = _findfirst(pathName.assign(path).append("\\*").c_str(), &folderInfo)) == -1) {
    return;
  }
  _findnext(hFolder, &folderInfo);
  while (_findnext(hFolder, &folderInfo) == 0)
  {
    cout << folderInfo.name << (folderInfo.attrib&_A_SUBDIR ? "[folder]" : "[file]") << endl;
    if (folderInfo.attrib&_A_SUBDIR)
    {
      if ((hFile = _findfirst(subpathName.assign(path).append("\\").append(folderInfo.name)
        .append("\\*").c_str(), &fileInfo)) == -1) {
        continue;
      }
      while (_findnext(hFile, &fileInfo) == 0)
      {
        if (!(fileInfo.attrib&_A_SUBDIR))
        {
          cout << "         " << fileInfo.name << (fileInfo.attrib&_A_SUBDIR ? "[folder]" : "[file]") << endl;
          // Store Face Information.
          {
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
            std::string test_dir = "./images/test/";

            //while (1)
            {
              string imgpath1(subpathName.assign(path).append("\\").append(folderInfo.name)
                .append("\\").append(fileInfo.name).c_str());
              //load image
              cv::Mat gallery_img_color = cv::imread(imgpath1, 1);
              if (gallery_img_color.empty())
              {
                continue;
              }

              cv::Mat gallery_img_gray;
              cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);
              cv::imshow("image1", gallery_img_color);
              cvWaitKey(0);
              
              ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
              gallery_img_data_color.data = gallery_img_color.data;

              ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
              gallery_img_data_gray.data = gallery_img_gray.data;

              // Detect faces
              double t = cvGetTickCount();
              std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
              t = cvGetTickCount() - t;
              cout << "face detetion consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
              int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

              //show detect results
              for (int i = 0; i < gallery_faces.size(); i++)
              {
                cv::Rect rc;
                rc.x = gallery_faces[i].bbox.x;
                rc.y = gallery_faces[i].bbox.y;
                rc.width = gallery_faces[i].bbox.width;
                rc.height = gallery_faces[i].bbox.height;
                cv::rectangle(gallery_img_color, rc, cv::Scalar(255, 0, 0));
              }
              imshow("image1", gallery_img_color);
              cvWaitKey(0);

              if (gallery_face_num == 0 )
              {
                std::cout << "Faces are not detected.";
                //   return 0;
                continue;
              }

              // Detect 5 facial landmarks
              seeta::FacialLandmark gallery_points[5];
              t = cvGetTickCount();
              point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
              t = cvGetTickCount() - t;
              cout << "face alignment consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
              for (int i = 0; i < 5; i++)
              {
                cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
                  CV_RGB(0, 255, 0));
              }
              imshow("image1", gallery_img_color);
              cvWaitKey(0);

              // Extract face identity feature
              float gallery_fea[2048];

              t = cvGetTickCount();
              face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
              t = cvGetTickCount() - t;
              cout << "face ExtractFeatureWithCrop consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
                            
              faceInfoOut << folderInfo.name << "  ";
              for (int i = 0; i < 2048; i++)
              {
                faceInfoOut << gallery_fea[i] << " ";
              }
              faceInfoOut << " " << endl;
            }
          }
        }
      }
      _findclose(hFile);
    }
  }
  _findclose(hFolder);
  faceInfoOut.close();
  return;
}


int main(int argc, char* argv[]) {
   readFaceFeature("D:\\work\\国家领导人\\leaders");

  return 0;
}


