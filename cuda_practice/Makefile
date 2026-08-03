NVCC = /usr/local/cuda-10.2/bin/nvcc
ARCH = -arch=sm_53
CFLAGS = -O2

TARGETS = vector_add matmul

all: $(TARGETS)

vector_add: vector_add.cu
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

matmul: matmul.cu
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
