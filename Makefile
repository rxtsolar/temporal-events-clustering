all:
	#g++ -o train train.cpp feature.cpp parser.cpp `pkg-config opencv --libs`
	#g++ -o test test.cpp feature.cpp parser.cpp `pkg-config opencv --libs`
	#g++ -o get-feature content.cpp get-feature.cpp parser.cpp `pkg-config opencv --libs`
	#g++ -o cluster cluster.cpp spectral.cpp parser.cpp `pkg-config opencv --libs`
	g++ -o main main.cpp spectral.cpp parser.cpp feature.cpp content.cpp `pkg-config opencv --libs`
