NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3
NVCC_LIBS   := -lnvidia-ml

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

SRC         := main.cu
TARGET      := app

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -arch=sm_$(GPU_ARCH) -o $@ $<

clean:
	rm -f $(TARGET)
