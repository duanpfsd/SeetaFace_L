/*****************************************************************************
*  @COPYRIGHT NOTICE
*  @Copyright (c) 2017, zhangyang
*  @All rights reserved
*  @file     : FaceIdentifyRecoger.cpp
*  @version  : ver 1.0
*  @author   : zhangyang
*  @date     : 2017/03/02 14:43
*  @brief    : 人脸图片识别算法的C++函数实现
*****************************************************************************/

#include "FaceIdentifyRecoger.h"
#include <iostream>
using namespace std;

#include <fstream>
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
using namespace cv;

/********************************************************
*  @function :  LoadDictionary
*  @brief    :  加载特征字典
*  @input    :  string path -- 特征字典文件的文件夹路径
*  @output   :
*  @return   :  成功返回0，失败返回非0

*  @author   :  zhangyang  2017/03/02 14:44
********************************************************/
bool FaceIdentifyRecoger::LoadDictionary(std::string model_path)
{
	string modelPath = model_path + "/seeta_fd_frontal_v1.0.bin";
	detector_ = new seeta::FaceDetection(modelPath.c_str());
	detector_->SetMinFaceSize(40);
	detector_->SetScoreThresh(2.f);
	detector_->SetImagePyramidScaleFactor(0.8f);
	detector_->SetWindowStep(4, 4);
	modelPath = model_path + "/seeta_fa_v1.1.bin";
	point_detector_ = new seeta::FaceAlignment(modelPath.c_str());
	modelPath = model_path + "/seeta_fr_v1.0.bin";
	face_recognizer_ = new seeta::FaceIdentification(modelPath.c_str());

	if (!dicFlag_)
	{
		//memset(gallery_threhold_, 0.6, kNumGallery*sizeof(float));
		int linecount = 0, streamcount = 0;
		ifstream  streamgallery;
		string    strgallery;
		modelPath = model_path + "/gallery_features.txt";
		streamgallery.open(modelPath, ios_base::in);
		while (getline(streamgallery, strgallery))
		{
			istringstream iss(strgallery);
			//iss >> name_pinyin[linecount];
			iss >> name_pinyin_[linecount] >> gallery_threhold_[linecount];
			streamcount = 0;
			while (iss >> gallery_fea_[linecount][streamcount])
			{
				streamcount++;
			}
			linecount++;
			if (linecount >= kNumGallery)
			{
				break;
			}
		}
		dicFlag_ = true;
	}	
  return dicFlag_;
}

/********************************************************
*  @function :  Java_com_bj58_zlsf_impl_FaceIdentifyRecoger_DetectAndRecogFaceFromImage
*  @brief    :  检测图片中是否含字典中的对象（人脸识别匹配）
*  @input    :  imData 图像数据， jint 图像数据长度
*  @output   :
*  @return   :  成功返回匹配的对象名字，失败返回未匹配原因的描述

*  @author   :  zhangyang  2017/03/02 14:47
********************************************************/
std::string FaceIdentifyRecoger::DetectAndRecogFaceFromImage(const char* imData, int len)
{
  string image_data = imData;
	string matchid = "";
	cv::Mat probe_img_color;
	cv::Mat rawData = cv::Mat(1, len, CV_8UC1, (unsigned char*)imData);
	probe_img_color = imdecode(rawData, 1);
	if (!probe_img_color.empty())
	{
		cv::Mat probe_img_gray;
		cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);
		ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
		probe_img_data_color.data = probe_img_color.data;
		ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
		probe_img_data_gray.data = probe_img_gray.data;

		vector<seeta::FaceInfo> probe_faces = detector_->Detect(probe_img_data_gray);		
		int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());
		probe_face_num = (probe_faces.size() < kMaxFaceNum) ? probe_faces.size() : kMaxFaceNum;
		if (probe_face_num > 0)
		{
			seeta::FacialLandmark probe_points[5] = { 0 };
      float probe_fea[kMaxFaceNum][2048] = { 0 };
			for (int i = 0; i < probe_face_num; i++)
			{
				point_detector_->PointDetectLandmarks(probe_img_data_gray, probe_faces[i], probe_points);
				face_recognizer_->ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea[i]);
			}

			float sim = -1.0f, maxsim = -1.0f;
			int maxindex = -1;
			for (int i = 0; i < kNumGallery; i++)
			{
				for (int j = 0; j < probe_face_num; j++)
				{
					sim = face_recognizer_->CalcSimilarity(gallery_fea_[i], probe_fea[j]);
					if (sim > maxsim)
					{
						maxsim = sim;
						maxindex = i;
					}
				}
			}
			if (maxsim > gallery_threhold_[maxindex]) 
			{
				matchid = to_string(maxsim)+"-"+name_pinyin_[maxindex];
			}
			probe_faces.clear();
			probe_img_gray.release();
		}
	}
	rawData.release();
	probe_img_color.release();
	return matchid;
}

