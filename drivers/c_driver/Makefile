CC = gcc
CFLAGS = -Wall -O2 -fPIC
LDFLAGS_LIB = -shared -lm
LDFLAGS_TEST = -lm

LIB_TARGET = libpca9685.so
TEST_TARGET = test_pca9685

LIB_SRCS = pca9685.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

TEST_SRCS = test_pca9685.c pca9685.c

# Default: build the shared library
all: $(LIB_TARGET)

# Build shared library for Python ctypes
$(LIB_TARGET): $(LIB_OBJS)
	$(CC) $(LDFLAGS_LIB) -o $@ $^

# Build standalone test binary (statically links pca9685.c)
test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS_TEST)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(LIB_OBJS) $(LIB_TARGET) $(TEST_TARGET)

.PHONY: all test clean
