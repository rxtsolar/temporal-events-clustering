#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int histSize = 16;

Mat getHistogram(const Mat& image)
{
	vector<Mat> channels;
	Mat hist;
	Mat result;

	split(image, channels);

	for (int i = 0; i < channels.size(); i++) {
		calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, 0);
		if (result.empty())
			hist.copyTo(result);
		else
			vconcat(result, hist, result);
	}

	return result;
}

Mat getHistImage(const Mat& histogram)
{
	Mat hist;
	Mat histImage = Mat::ones(600, 1200, CV_8U) * 255;
	normalize(histogram, hist, 0, histImage.rows, CV_MINMAX, CV_32F);

	histImage = Scalar::all(255);
	int binW = cvRound(static_cast<double>(histImage.cols / hist.rows));

	for (int i = 0; i < hist.rows; i++) {
		rectangle(histImage, Point(i * binW, histImage.rows),
				Point((i + 1) * binW, histImage.rows - cvRound(hist.at<float>(i))),
				Scalar::all(0), -1, 8, 0);
	}

	return histImage;
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		cerr << "usage: " << argv[0] << " <input-image> <output-hist>" << endl;
		return -1;
	}

	Mat image = imread(argv[1], 1);
	Mat hist = getHistogram(image);
	cerr << hist.rows << ' ' << hist.cols << endl;
	Mat histImage = getHistImage(hist);
	imwrite(argv[2], histImage);
	//Mat fImage;
	//image.convertTo(fImage, CV_32F, 1.0/255);
	//Mat dctImage;
	//dct(fImage, dctImage);
	//imshow("image", image);
	//imshow("dct", dctImage);
	//while (waitKey(1000) != 27);
	return 0;
}
