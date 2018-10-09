

all: denoise

denoise: main.cpp
	g++ -g -pthread $^ -o $@
run: denoise
	./denoise