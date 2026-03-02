NVCC        := nvcc
NVCC_FLAGS  := -Xptxas -dlcm=cg -O3
NVCC_LIBS   := -lnvidia-ml

SRC         := main.cu
TARGET      := app

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -arch=sm_120 -o $@ $<

clean:
	rm -f $(TARGET)
