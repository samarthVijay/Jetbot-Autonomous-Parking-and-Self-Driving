NVCC = /usr/local/cuda-10.2/bin/nvcc
ARCH = -arch=sm_53
FLAGS = -O2

LIB_TARGET = libcuda_preprocess.so
TEST_TARGET = test_preprocess

all: $(LIB_TARGET) $(TEST_TARGET)

$(LIB_TARGET): preprocess_kernel.cu
	$(NVCC) $(ARCH) --shared -Xcompiler -fPIC $(FLAGS) -o $@ $<

$(TEST_TARGET): test_preprocess.cu preprocess_kernel.cu
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $^

clean:
	rm -f $(LIB_TARGET) $(TEST_TARGET)

.PHONY: all clean
