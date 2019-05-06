# Note: Automatically set when running Python code
INCLUDES=-I/usr/include/python3.6m -I/usr/local/lib/python3.6/dist-packages/numpy/core/include

# Common header files (just include everything always)
HEADERS=vector.hpp basis.hpp wfn.hpp atomCL.hpp

# C++ compiler information
CPP=g++
CFLAGS=-std=c++11 -O3 -ftree-vectorize -funroll-loops --fast-math -fopenmp


.PHONY: all
all: librot.so libwfn.so libcurrent.so

# Compiling object files
%.o: %.cpp $(HEADERS)
	$(CPP) $(CFLAGS) $(INCLUDES) -c -fPIC $< -o $@

# Compiling shared libraries
lib%.so: %.o $(HEADERS)
	$(CPP) $(CFLAGS) $(INCLUDES) -shared -Wl,-soname,$@ -o $@ $<

.PHONY: clean
clean:
	rm -f *.so
