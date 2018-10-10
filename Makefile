all: denoise readImage

denoise: main.cpp
	g++ -g -pthread $^ -o $@
readImage: readImage.cpp
	g++ `pkg-config --cflags opencv` -g $^ `pkg-config --libs opencv` -o $@
run: readImage
	./readImage

clean:
	rm -Rf readImage denoise
