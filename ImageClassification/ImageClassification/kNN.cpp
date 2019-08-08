#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	Mat train_features(5, 2, CV_32FC1);
	Mat labels(5, 1, CV_32FC1);

	// 점의 좌표를 train_features에 입력한다.
	train_features.at<float>(0, 0) = 10, train_features.at<float>(0, 1) = 10;
	train_features.at<float>(1, 0) = 10, train_features.at<float>(1, 1) = 20;
	train_features.at<float>(2, 0) = 20, train_features.at<float>(2, 1) = 10;
	train_features.at<float>(3, 0) = 30, train_features.at<float>(3, 1) = 30;
	train_features.at<float>(4, 0) = 40, train_features.at<float>(4, 1) = 30;

	// 원하는 레이블을 labels에 입력한다.
	labels.at<float>(0, 0) = 1;
	labels.at<float>(1, 0) = 1;
	labels.at<float>(2, 0) = 1;
	labels.at<float>(3, 0) = 2;
	labels.at<float>(4, 0) = 2;

	// 학습 과정
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ROW_SAMPLE, labels);
	knn->train(trainData);

	// 테스트 과정
	Mat sample(1, 2, CV_32FC1);
	Mat predictedLabels;

	// 테스트 데이터를 입력한다.
	sample.at<float>(0, 0) = 28, sample.at<float>(0, 1) = 28;
	knn->findNearest(sample, 2, predictedLabels);

	float prediction = predictedLabels.at<float>(0, 0);
	cout << "테스트 샘플의 라벨 = " << prediction << endl;
	return 0;
}