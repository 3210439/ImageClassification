#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifndef CV_TERMCRIT_ITER
#define CV_TERMCRIT_ITER cv::TermCriteria::MAX_ITER
#endif

using namespace cv;
using namespace std;

int main()
{
	Mat samples(50, 2, CV_32F);

	for (int y = 0; y < samples.rows; y++) {
		samples.at<float>(y, 0) = (rand() % 255);
		samples.at<float>(y, 1) = (rand() % 255);
	}
	Mat dst(256, 256, CV_8UC3);

	for (int y = 0; y < samples.rows; y++) {
		float x1 = samples.at<float>(y, 0);
		float x2 = samples.at<float>(y, 1);
		circle(dst, Point(x1, x2), 3, Scalar(255, 0, 0));
	}
	imshow("dst", dst);

	Mat result;
	Mat labels(50, 1, CV_8UC1);
	Mat centers;
	result = Mat::zeros(Size(256, 256), CV_8UC3);
	kmeans(samples, 2, labels, TermCriteria(CV_TERMCRIT_ITER | cv::TermCriteria::EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < samples.rows; y++) {
		float x1 = samples.at<float>(y, 0);
		float x2 = samples.at<float>(y, 1);
		int cluster_idx = labels.at<int>(y, 0);
		if (cluster_idx == 0)
			circle(result, Point(x1, x2), 3, Scalar(255, 0, 0));
		else
			circle(result, Point(x1, x2), 3, Scalar(255, 255, 0));
	}
	imshow("result", result);
	waitKey();
}

/*영상 분류란 셩상 안의 어떤 특징을 이용하여 영상 안의 물체를 분류하는 것이다. 영상 인식이라고도 한다.
영상을 분류할 때는 먼저 카메라가 영상을 캡처한 후에 전처리 하고 이어서 영상 분할 기법을 사
용하여 각 물체를 분류한다. 이들 물체의 특징값을 측정하고 이것이 분류기로 보내져서 최종 판단을 
내리게 된다. 
kNN은 테스트 샘플이 입력되면 인접한 k개의 학습 데이터를 선택하여 k개 중에서 가장 많이 들어 있는 
클래스로 테스트 샘플을 분류하는 알고리즘이다. 
OpenCV에서는 kNN을 위하여 KNearest클래스를 제공한다. 포인터 참조 방법을 사용하여 객체를 생성
하며, train()함수로 학습을 시킨다. 학습이 종료되면 findNearest() 함수를 이용하여 분류를 수행한다.
k-means클러스터링은 선생님이 없어도 분류를 할 수 있는 알고리즘이다. 입력 데이터를 몇 개로 분류
할 것인지만 주어지면 각 클래스의 유사도는 증가시키고 다른 클래스와의 유사도는 감소시키는 방식으로 분류
가 진행된다.
OpenCV에서는 k-means클러스터링을 위하여 kmeans()함수를 제공한다. */