all:
	#g++ -o show show.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
	#g++ -o process cooper.cpp feature.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
	#g++ -o train train.cpp feature.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
	#g++ -o test test.cpp feature.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
	g++ content.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
