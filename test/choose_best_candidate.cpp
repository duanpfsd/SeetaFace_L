#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
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

std::string GALLERY_DIR = "./images/train/34zlj/train/f/";
//std::string GALLERY_DIR = "C:\\python\\crawler\\image3\\";
bool isshowpic = false;

// Read facial information and store in files 
void readFaceFeature(string path)
{
  // Some vars of  folders and picture files.
  long hFile = 0;
  struct _finddata_t fileInfo;
  string pathName;

  // The *.txt file to store Face Information.
  std::ofstream faceInfoOut;
  faceInfoOut.open(pathName.assign(path).append("/GalleryFaceInfo.txt"), ofstream::out);


  // Read picture files and store Face Information.
  if ((hFile = _findfirst(pathName.assign(path).append("/*").c_str(), &fileInfo)) == -1) {
    return;
  }
  int ncount = 0;
  while (_findnext(hFile, &fileInfo) == 0)
  {
    double t = cvGetTickCount();
    ncount++;
    if (ncount % 5 == 0) {
      cout << ncount << endl; 
      t = cvGetTickCount() - t;
      cout << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
    }
    if (!(fileInfo.attrib&_A_SUBDIR))
    {
      //cout << "         " << fileInfo.name << (fileInfo.attrib&_A_SUBDIR ? "[folder]" : "[file]") << endl;
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

        //while (1)
        {
          string imgpath1(pathName.assign(path).append(fileInfo.name).c_str());
          //load image
          cv::Mat gallery_img_color = cv::imread(imgpath1, 1);
          if (gallery_img_color.empty())
          {
            continue;
          }

          cv::Mat gallery_img_gray;
          cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);
          //cv::imshow("image1", gallery_img_color);
              
          ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
          gallery_img_data_color.data = gallery_img_color.data;

          ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
          gallery_img_data_gray.data = gallery_img_gray.data;

          // Detect faces

          std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
          t = cvGetTickCount() - t;
          //cout << "face detetion consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
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
          //imshow("image1", gallery_img_color);

          if (gallery_face_num == 0 )
          {
            std::cout << "Faces are not detected."<<endl;
            //   return 0;
            continue;
          }

          // Detect 5 facial landmarks
          seeta::FacialLandmark gallery_points[5];
          t = cvGetTickCount();
          point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
          t = cvGetTickCount() - t;
          //cout << "face alignment consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
          for (int i = 0; i < 5; i++)
          {
            cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2, CV_RGB(0, 255, 0));
          }
          if (isshowpic){
            imshow("image1", gallery_img_color);
            cvWaitKey(0);
          }

          // Extract face identity feature
          float gallery_fea[2048];
          t = cvGetTickCount();
          face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
          t = cvGetTickCount() - t;
          //cout << "face ExtractFeatureWithCrop consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;

          faceInfoOut << fileInfo.name << "  " << gallery_faces[0].score << " ";
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
  faceInfoOut.close();
  return;
}


int main(int argc, char* argv[]) {
  readFaceFeature(GALLERY_DIR);

  while (1)
  {
    // Initialize face Identification model 
    FaceIdentification face_recognizer("seeta_fr_v1.0.bin");
    std::string test_dir =  "./images/test/";

    // Load gallerry face information
    int ncount = 0;
    float gallery_fea[70][2048] = {0.f};
    bool isAnyGallery = false;
    cout << "ookk1" << endl;
    string pathName;
    string  gallery_name[70];
    ifstream  galleryFaceInfoIn;
    string    strGalleryFace;
    galleryFaceInfoIn.open(pathName.assign(GALLERY_DIR).append("/GalleryFaceInfo.txt"), ios_base::in);
    cout << "ookk2" << endl;
    double t = cvGetTickCount();
    float detectscore;
    while (getline(galleryFaceInfoIn, strGalleryFace))
    {
      cout << "ookk3" << endl;
      istringstream iss(strGalleryFace);
      iss >> gallery_name[ncount] >> detectscore;
      int ncount2 = 0;
      while (iss >> gallery_fea[ncount][ncount2])
      {
        ncount2 ++;
      }
      ncount++;
      if (ncount >= 69){
        cout << "Too  much pics in gallery: " << ncount << endl;
        break;
      }
      cout << "ookk4" << endl;
    }
    cout << "ookk5" << endl;
    galleryFaceInfoIn.close();
    t = cvGetTickCount() - t;
    cout << "load face information consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;

    //calculate similarity
    float score[70] = { 0.f }, sim = 0.f;
    vector<string> vsimilarpic;
    vsimilarpic.clear();
    vector<float> vsimilarscore;
    vsimilarscore.clear();
        for (int i = 0; i < ncount-1; i++){
      for (int j = i + 1; j<ncount; j++){
        sim = face_recognizer.CalcSimilarity(gallery_fea[i], gallery_fea[j]);
        if (sim>0.9f){
          vsimilarpic.push_back(gallery_name[i]);
          vsimilarpic.push_back(gallery_name[j]);
          vsimilarscore.push_back(sim);
        }
        score[i] += sim;
        score[j] += sim;
        cout << i << " " << j << " " << gallery_name[i] << " " << gallery_name[j] << " : " << sim << endl;
      }
    }
    
    //sort by socre
    float temp_score;
    string temp_name;
    for (int i = ncount-1; i >0; i--){
      for (int j = 0; j < i; j++){
        if (score[j + 1]>score[j]){
          temp_score = score[j];
          score[j] = score[j + 1];
          score[j + 1] = temp_score;
          temp_name = gallery_name[j];
          gallery_name[j] = gallery_name[j + 1];
          gallery_name[j + 1] = temp_name;
        }
      }
    }

    //print by score and similar pics
    for (int i = 0; i < ncount; i++){
      cout << i << " " << gallery_name[i] << " : " << score[i] / (ncount - 1) << endl;
    }
    for (int i = 0; i < vsimilarpic.size()/2; i++){
      cout << vsimilarpic.size()<<" "<<i << " " << vsimilarpic[2 * i] << " : " << vsimilarpic[2 * i + 1] << " : " << vsimilarscore[i] << endl;
    }
    vsimilarpic.clear();
    vector<string>().swap(vsimilarpic);
    vsimilarscore.clear();
    vector<float>().swap(vsimilarscore);
    _getch();
  }
  return 0;
}


