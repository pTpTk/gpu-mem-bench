NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3

SRC         := main.cu
TARGET      := HBM_LAT-BW

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_DEFS) -arch=sm_120 -o $@ $<

clean:
	rm -f $(TARGET)
