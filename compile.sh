OUT_FILE=main.o
nvcc -std=c++11 src/main.cu src/graph/graph.cpp -o $OUT_FILE \
&& echo "Compiled the program and created executable named $OUT_FILE."

