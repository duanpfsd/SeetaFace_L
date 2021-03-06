#include<iostream>
#include <fstream>
#include <io.h>
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

int main(int argc, char* argv[])
{
  std::string path = "D:\\pythonPath\\march 14\\true\\";
  string gallery_dir = "./images/gallery_fea_v1/";
  float gallery_fea[105][2048] = { 0.0f };
  float gallery_threhold[105] = { 0.60f };
  string gallery_name[105];
  {
    int linecount = 0, stringcount = 0;
    ifstream  istreamgallery;
    string    strgallery;
    istreamgallery.open(gallery_dir + "/gallery_features.txt", ios_base::in);
    while (getline(istreamgallery, strgallery))
    {
      istringstream iss(strgallery);
      iss >> gallery_name[linecount] >> gallery_threhold[linecount];
      stringcount = 0;
      cout << linecount << "  " << gallery_threhold[linecount];
      while (iss >> gallery_fea[linecount][stringcount])
      {
        stringcount++;
      }
      linecount++;
    }
  }

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

  {
    long hFile = 0;
    struct _finddata_t fileInfo;
    string pathName;

    if ((hFile = _findfirst(pathName.assign(path).append("/*").c_str(), &fileInfo)) == -1) 
    {
      return 0;
    }

    std::ofstream faceInfoOut;
    faceInfoOut.open(pathName.assign(path).append("/checkInfo2.txt"), ofstream::out);
    int ncount = 0;
    while (_findnext(hFile, &fileInfo) == 0)
    {
      ncount++;
      if (ncount % 5 == 0)
      {
        cout << ncount << endl;
      }
      if (!(fileInfo.attrib&_A_SUBDIR))
      {
        string imgpath1(pathName.assign(path).append(fileInfo.name).c_str());
        cv::Mat gallery_img_color = cv::imread(imgpath1, 1);
        if (gallery_img_color.empty())
        {
          cout << "No picture!" << fileInfo.name << endl;
          faceInfoOut << "Nopicture 0 0 " << fileInfo.name << endl;
          continue;
        }
        cv::Mat gallery_img_gray;
        cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

        ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
        gallery_img_data_color.data = gallery_img_color.data;

        ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
        gallery_img_data_gray.data = gallery_img_gray.data;

        // Detect faces
        std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
        int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

        if (gallery_face_num == 0)
        {
          cout << "Faces are not detected." << fileInfo.name << endl;
          faceInfoOut << "Nofacedet 0 0" << fileInfo.name << endl;
          continue;
        }
        float maxsim = -1.f;
        int maxindex = -1;
        int maxjndex = -1;
        for (int i = 0; i < gallery_face_num; i++)
        {
          // Detect 5 facial landmarks
          seeta::FacialLandmark gallery_points[5];
          point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);

          float probe_fea[2048];
          face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, probe_fea);

          for (int j = 0; j < 105; j++)
          {
            float sim = face_recognizer.CalcSimilarity(gallery_fea[j], probe_fea);
            if (maxsim<sim)
            {
              maxsim = sim;
              maxindex = j;
              maxjndex = i;
            }
            if (maxsim > 0.8) break;
          }
        }
        //if (maxsim > gallery_threhold[maxindex])
        if (maxsim>gallery_threhold[maxindex])
        {
          cout << gallery_name[maxindex] << " " << maxjndex << " " << maxindex << " " << maxsim << " " << fileInfo.name << endl;
          faceInfoOut << gallery_name[maxindex] << " " << maxjndex << " " << maxindex << " " << maxsim << " " << fileInfo.name << endl;
        }
        else
        {
          cout << "Nomatchgallery 0 0 " << fileInfo.name << endl;
          faceInfoOut << "Not_match 0 0 " << fileInfo.name << endl;
        }

      }//if _A_SUBDIR
    }//while
  }//namespace
  return 0;
}


