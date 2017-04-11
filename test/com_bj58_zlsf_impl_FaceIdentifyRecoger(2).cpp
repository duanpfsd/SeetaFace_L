#include "FaceIdentifyRecoger.h"
/*****************************************************************************
*  @COPYRIGHT NOTICE
*  @Copyright (c) 2017, zhangyang
*  @All rights reserved
*  @file     : FaceIdentifyRecoger.cpp
*  @version  : ver 1.0
*  @author   : zhangyang
*  @date     : 2017/03/02 14:43
*  @brief    : ����ͼƬʶ���㷨��C++����ʵ��
*****************************************************************************/

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

/********************************************************
*  @function :  LoadDictionary
*  @brief    :  ���������ֵ�
*  @input    :  string path -- �����ֵ��ļ����ļ���·��
*  @output   :
*  @return   :  �ɹ�����0��ʧ�ܷ��ط�0

*  @author   :  zhangyang  2017/03/02 14:44
********************************************************/
bool LoadDictionary(std::string model_path)
{
  bool flag = false;
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

	if (!dicFlag)
	{
		//memset(gallery_threhold_, 0.6, kNumGallery*sizeof(float));
		int linecount = 0, streamcount = 0;
		ifstream  streamgallery;
		string    strgallery;
		modelPath = model_path + "/gallery_features.txt";
		streamgallery.open(modelPath, ios_base::in);
		while (getline(streamgallery, strgallery))
		{
			flag = true;
			istringstream iss(strgallery);
			//iss >> name_pinyin[linecount];
			iss >> name_pinyin_[linecount] >> gallery_threhold_[linecount];
			streamcount = 0;
			while (iss >> gallery_fea[linecount][streamcount])
			{
				streamcount++;
			}
			linecount++;
			if (linecount >= NUM_GALLARY)
			{
				flag = false;
				break;
			}
		}
		dicFlag = true;
	}	
	return flag;
}

/********************************************************
*  @function :  Java_com_bj58_zlsf_impl_FaceIdentifyRecoger_DetectAndRecogFaceFromImage
*  @brief    :  ���ͼƬ���Ƿ��ֵ��еĶ�������ʶ��ƥ�䣩
*  @input    :  imData ͼ�����ݣ� jint ͼ�����ݳ���
*  @output   :
*  @return   :  �ɹ�����ƥ��Ķ������֣�ʧ�ܷ���δƥ��ԭ�������

*  @author   :  zhangyang  2017/03/02 14:47
********************************************************/
std::string com_bj58_zlsf_impl_FaceIdentifyRecoger_DetectAndRecogFaceFromImage(
  const char* imData, 
  int len, 
  int model_id);
{
	string matchid = "";
	cv::Mat probe_img_color;
	cv::Mat rawData = cv::Mat(1, len, CV_8UC1, imData);
	probe_img_color = imdecode(rawData, 1);
	if (!probe_img_color.empty())
	{
		cv::Mat probe_img_gray;
		cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);
		ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
		probe_img_data_color.data = probe_img_color.data;
		ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
		probe_img_data_gray.data = probe_img_gray.data;

		vector<seeta::FaceInfo> probe_faces = detectors[idx]->Detect(probe_img_data_gray);		
		int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());
		probe_face_num = (probe_faces.size() < MAX_FACE_NUM) ? probe_faces.size() : MAX_FACE_NUM;
		if (probe_face_num > 0)
		{
			seeta::FacialLandmark probe_points[5] = { 0 };
			float probe_fea[MAX_FACE_NUM][2048] = { 0 };
			for (int i = 0; i < probe_face_num; i++)
			{
				point_detectors[idx]->PointDetectLandmarks(probe_img_data_gray, probe_faces[i], probe_points);
				face_recognizers[idx]->ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea[i]);
			}

			float sim = -1.0f, maxsim = -1.0f;
			int maxindex = -1;
			for (int i = 0; i < NUM_GALLARY; i++)
			{
				for (int j = 0; j < probe_face_num; j++)
				{
					sim = face_recognizers[idx]->CalcSimilarity(gallery_fea[i], probe_fea[j]);
					if (sim > maxsim)
					{
						maxsim = sim;
						maxindex = i;
					}
				}
			}
			if (maxsim > gallery_threhold[maxindex]) 
			{
				matchid = to_string(maxsim)+"-"+name_pinyin[maxindex];
			}
			probe_faces.clear();
			probe_img_gray.release();
		}
	}
	rawData.release();
	probe_img_color.release();
	return matchid;
}

