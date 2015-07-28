all:
	g++ -o show show.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
	g++ -o process cooper.cpp -lopencv_core -lopencv_imgproc
