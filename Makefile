SRC_DIR := src
exe = run_bfs

cc = "$(shell which nvcc)" 
flags = -I. -O3 -Xptxas -dlcm=ca

ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst $(SRC_DIR)/%.cu,$(SRC_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cu))

deps =  $(wildcard $(SRC_DIR)/*/*.hpp) \
	Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

clean:
	rm -rf $(exe) $(objs) 
