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

int main(int argc, char* argv[]) {

  // Train dir for 35 targets.
  string TEST_DIR = "D:\\learn\\FaceID\\FaceID\\images\\train\\";
  string leader[35] = {  "0mzd",  "1dxp",  "2jzm",  "3hjt",  "4wjb",  "5wbg",  "6jql",  "7lcc",  "8hgq",  "9zyk",
                        "10xjp", "11lkq", "12zdj", "13yzs", "14lys", "15wqs", "16zgl", "17fcl", "18gjl", "19hch",
                        "20hz" , "21ljg", "22lqb", "23lyc", "24lyd", "25lzs", "26mjz", "27mk" , "28scl", "29szc",
                        "30whn", "31wy" , "32xql", "33zcx", "34zlj" };
  string leader_chinese_name[35] = { "毛泽东", "邓小平", "江泽民", "胡锦涛", "温家宝", "吴邦国", "贾庆林", "李长春", "贺国强", "周永康",
    "习近平", "李克强", "张德江", "俞正声", "刘云山", "王岐山", "张高丽", "范长龙", "郭金龙", "胡春华",
    "韩正", "李建国", "刘奇葆", "李源潮", "刘延东", "栗战书", "孟建柱", "马凯", "孙春兰", "孙政才",
    "王沪宁", "汪洋", "许其亮", "张春贤", "赵乐际" };
  string leader_pinyin_name[35] = { "Maozedong", "Dengxiaoping", "Jiangzemin", "Hujintao", "Wenjiabao", "Wubangguo", "Jiaqinglin", "Lichangchun", "Heguoqiang", "Zhouyongkang",
    "Xijinping", "Likeqiang", "Zhangdejiang", "Yuzhengsheng", "Liuyunshan", "Wangqishan", "Zhanggaoli", "Fanchanglong", "Guojinlong", "Huchunhua",
    "Hanzheng", "Lijianguo", "Liuqibao", "Liyuanchao", "Liuyandong", "Lizhanshu", "Mengjianzhu", "Makai", "Sunchunlan", "Sunzhengcai",
    "Wanghuning", "Wangyang", "Xuqiliang", "Zhangchunxian", "Zhaoleji" };

  float threhold[35] = { 0.62, 0.60, 0.62, 0.76, 0.72, 0.68, 0.68, 0.66, 0.72, 0.68, 
                         0.70, 0.64, 0.72, 0.72, 0.68, 0.68, 0.72, 0.74, 0.72, 0.72,
                         0.66, 0.66, 0.64, 0.70, 0.72, 0.64, 0.66, 0.66, 0.70, 0.66,
                         0.64, 0.68, 0.68, 0.70, 0.74};
  float threhold_corr[35] = { 0.62, 0.60, 0.62, 0.73, 0.72, 0.68, 0.68, 0.66, 0.72, 0.68,
                              0.70, 0.64, 0.72, 0.72, 0.68, 0.68, 0.72, 0.72, 0.72, 0.72,
                              0.66, 0.66, 0.64, 0.70, 0.72, 0.64, 0.66, 0.66, 0.70, 0.66,
                              0.64, 0.68, 0.68, 0.70, 0.72 };
  string name;
  string pathName;

  float feature_point;
  // The *.txt file to store gallery features.
  std::ofstream resultsOut;
  resultsOut.open("D:\\learn\\FaceID\\FaceID\\images\\gallery_fea.txt", ofstream::out);

  for(int i=0; i<35; i++)
  {
    int ncount = 0;
    float gallery_fea[3][2048] = {0.f};

    std::ifstream  galleryFaceInfoIn;
    string    strGalleryFace;
    galleryFaceInfoIn.open(pathName.assign(TEST_DIR).append(leader[i]).append("\\train\\gallery_fea.txt"), ios_base::in);

    while (getline(galleryFaceInfoIn, strGalleryFace))
    {
      istringstream iss(strGalleryFace);
      float detectscore;
      int num;
      iss >> name >> num >> detectscore;
      resultsOut << leader_pinyin_name[i] << "  " << threhold_corr[i] << "  ";
      //resultsOut << leader_chinese_name[i] << "  " << leader_pinyin_name[i] << "  ";
      while (iss >> feature_point)
        resultsOut << feature_point << " ";
      resultsOut << endl;
    }
    galleryFaceInfoIn.close();
  }
  resultsOut.close();
  return 0;
}
