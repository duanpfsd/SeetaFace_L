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

string leader[35] = {  "0mzd",  "1dxp",  "2jzm",  "3hjt",  "4wjb",  "5wbg",  "6jql",  "7lcc",  "8hgq",  "9zyk",
                      "10xjp", "11lkq", "12zdj", "13yzs", "14lys", "15wqs", "16zgl", "17fcl", "18gjl", "19hch",
                      "20hz" , "21ljg", "22lqb", "23lyc", "24lyd", "25lzs", "26mjz", "27mk" , "28scl", "29szc",
                      "30whn", "31wy" , "32xql", "33zcx", "34zlj" };
float gallery_fea[3][2048];

void readFaceFeature(int numleader)
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

  long hFile = 0;
  struct _finddata_t fileInfo;
  string pathName;
  string path = "D:\\learn\\FaceID\\FaceID\\images\\train\\";

  {
    std::ofstream faceInfoOut;
    faceInfoOut.open(pathName.assign(path).append(leader[numleader]).append("\\train\\gallery_fea.txt"), ofstream::out);

    if ((hFile = _findfirst(pathName.assign(path).append(leader[numleader]).append("\\train\\target\\*").c_str(), &fileInfo)) == -1) {
      return;
    }
    int ncount = 0;
    while (_findnext(hFile, &fileInfo) == 0)
    {
      //cout << ncount << endl; 

      if (!(fileInfo.attrib&_A_SUBDIR))
      {
        string imgpath1(pathName.assign(path).append(leader[numleader]).append("\\train\\target\\").append(fileInfo.name).c_str());
        cv::Mat gallery_img_color = cv::imread(imgpath1, 1);
        if (gallery_img_color.empty())
        {
          cout << "WRONG !! 1 " << endl;
          continue;
        }
        cv::Mat gallery_img_gray;
        cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);
        ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
        gallery_img_data_color.data = gallery_img_color.data;
        ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
        gallery_img_data_gray.data = gallery_img_gray.data;

        std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
        int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
        if (gallery_face_num != 1 )
        {
          cout << "WRONG !! 2" << endl;
          cout << imgpath1 << "  " << gallery_face_num << endl;
          continue;
        }
        seeta::FacialLandmark gallery_points[5];
        for(int i=0;i<gallery_face_num;i++)
        {
          point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);
          face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea[ncount-1]);

          faceInfoOut << fileInfo.name << "  " << i << "  " << gallery_faces[i].score << "  ";
          for (int j = 0; j < 2048; j++)
          {
            faceInfoOut << gallery_fea[ncount-1][j] << " ";
          }
          faceInfoOut << " " << endl;
        }
      }
      ncount++;
    }
    _findclose(hFile);
    faceInfoOut.close();
  }
  return;
}

void readFaceFeature(char* path, int i)
{
  cout << path << endl;

  float detectscore;
  int faceindex;
  string namepic;
  int linecount = 0, stringcount = 0;

  ifstream  galleryFaceInfoIn;
  string    strGalleryFace;
  galleryFaceInfoIn.open(path, ios_base::in);
  while (getline(galleryFaceInfoIn, strGalleryFace))
  {
    istringstream iss(strGalleryFace);
    iss >> namepic ;
    if (linecount<3 * i || linecount>3 * i + 2)
    {
      linecount++;
      continue;
    }
    else
    {
      while (iss >> gallery_fea[linecount % 3][stringcount])
      {
        stringcount++;
      }
      stringcount = 0;
      linecount++;
    }
  }
  return;
}

int main(int argc, char* argv[])
{
  // Initialize face Identification model 
  FaceIdentification face_recognizer("seeta_fr_v1.0.bin");

  for (int i = 0; i < 35; i++)
  {
    readFaceFeature("D:\\learn\\FaceID\\FaceID\\images\\gallery_fea_v1\\gallery_features.txt", i);

    //readFaceFeature(i);
    if (true)
    {
      for (int i = 0; i < 2048; i++)
      {
        //cout << gallery_fea[0][i] << " "<<gallery_fea[1][i] << " " << gallery_fea[2][i] << endl;
      }
      vector<string> test_dir;
      test_dir.clear();
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\head400clean\\gallery_fea.txt");
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\outdoor100clean\\gallery_fea.txt");
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\speech300clean\\gallery_fea.txt");
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\face340clean\\gallery_fea.txt");
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\zhejiang900clean\\gallery_fea.txt");
      test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\certificate560clean\\gallery_fea.txt");
      //test_dir.push_back("E:\\MyDownloads\\Download\\fktpxzq\\pic\\beauty1000clean\\gallery_fea.txt");

      string namedir;
      int hprobescore[2][50] = { 0 };
      for (int j = 0; j < 35; j++)
      {
        namedir.assign("D:\\learn\\FaceID\\FaceID\\images\\train\\").append(leader[j]).append("\\train\\total\\gallery_fea.txt");
        test_dir.push_back(namedir);
        //namedir.assign("D:\\learn\\FaceID\\FaceID\\images\\train\\").append(leader[j]).append("\\test\\gallery_fea.txt");
        //test_dir.push_back( namedir );   
      }

      //namedir.assign("D:\\learn\\FaceID\\FaceID\\images\\train\\").append(leader[i]).append("\\results.txt");
      //std::ofstream resultsOut;
      //resultsOut.open(namedir, ofstream::out);
      for (int j = 0; j < test_dir.size(); j++)
      {
        ifstream  galleryFaceInfoIn;
        string    strGalleryFace;
        galleryFaceInfoIn.open(test_dir[j], ios_base::in);

        float detectscore;
        int faceindex;
        string namepic;
        float probe_fea[2048] = { 0.f };
        cout << test_dir[j] << endl;
        while (getline(galleryFaceInfoIn, strGalleryFace))
        {
          istringstream iss(strGalleryFace);
          iss >> namepic >> faceindex >> detectscore;
          //if (j==2) cout << namepic <<"  "<< detectscore <<"  "<< faceindex << endl;
          int ncount2 = 0;
          while (iss >> probe_fea[ncount2])
          {
            ncount2++;
          }
          ncount2 = 0;

          int istrueeve;
          if (j - test_dir.size() + 35 == i) istrueeve = 0;
          else istrueeve = 1;

          float maxsim = -1.0f;
          for (int k = 0; k < 3; k++)
          {
            float sim = face_recognizer.CalcSimilarity(gallery_fea[k], probe_fea);
            //if(j==1)cout << endl;
            //if(j==1)cout << namepic << "  ";
            for (int n = 0; n<20; n++)
            {
              //if(j==1)cout << gallery_fea[k][n] << "," << probe_fea[n] << " ";
            }
            maxsim = (maxsim>sim) ? maxsim : sim;
          }
          //resultsOut << j << "  " << istrueeve << "  " << namepic << "  " << maxsim << endl;
          //if (j == 2) cout << j << "  " << istrueeve << "  " << namepic << "  " << maxsim << endl;
          for (int k = 0; k < 50; k++)
          {
            if (maxsim < (k + 1)*1.0 / 50)
              hprobescore[istrueeve][k] ++;
          }

        }
      }
      //resultsOut.close();

      double maxsum = 0;
      double maxjq = 0;
      double maxzh = 0;
      int maxindex = 0;
      //cout << endl;
      char buf[10];
      sprintf(buf, "%d", i);
      namedir.assign("D:\\learn\\FaceID\\FaceID\\images\\gallery_fea_v1\\threhold.txt").append(buf).append(".txt");
      std::ofstream threholdOut;
      threholdOut.open(namedir, ofstream::out);
      for (int j = 0; j < 40; j++)
      {
        //cout << hprobescore[0][j] << " " << hprobescore[1][j] << " " << hprobescore[0][49] << " " << hprobescore[1][49] << endl;
        double jingque = (1.*hprobescore[0][49] - hprobescore[0][j]) / (0.000001 + 1.*hprobescore[0][49] - hprobescore[0][j] + hprobescore[1][49] - hprobescore[1][j]);
        double zhaohui = 1 - 1.*hprobescore[0][j] / hprobescore[0][49];
        threholdOut << j << "  " << hprobescore[0][j] << " " << hprobescore[0][49] << " " << hprobescore[1][j] << " " << hprobescore[1][49] << "  " << jingque << "  " << zhaohui << "  " << (jingque + zhaohui) << " " << endl;
        //cout <<j<<"  "<<jingque<<"  "<<zhaohui<<"  "<<(jingque + zhaohui) << endl;
      }
      threholdOut.close();
    }
  }
  return 0;
}


