# Task-Aware CUDA Library

## Building and installing
TACUDA uses the standard GNU automake and libtool toolchain. When cloning from a repository, the building environment must be prepared executing the bootstrap script:

```
$ ./bootstrap
```

Then, to configure, compile and install the library execute the following commands:

```
$ ./configure --prefix=$INSTALL_PREFIX --with-cuda=$CUDA_HOME ..other options..
$ make
$ make install
```

where $INSTALL_PREFIX is the directory into which to install TACUDA, and $CUDA_HOME is the prefix of the CUDA installation. There are two other optional configuration flags, which probably will not be needed if the user has a normal CUDA setup:

- `--with-cuda-include` : It lets the user specify the directory where the CUDA include files are installed.
- `--with-cuda-lib` : It lets the user specify the directory where the CUDA libraries are installed.

Once TACUDA is built and installed, e.g, in $TACUDA_HOME, the installation folder will contain the library in $TACUDA_HOME/lib (or $TACUDA_HOME/lib64) and the header in $TACUDA_HOME/include.

### Requirements
In order to install the TACUDA library, the main requirements are the following:

- C++ compiler (g++)
- CUDA, including the Runtime, Driver and cuBLAS APIs (with a version compatible with the g++ version).
- Boost library 1.59 or later
- ...

The cuBLAS API is not essential to the library, so if the user does the proper changes to the Makefile.am file (basically removing or commenting all the lines containing `src/c/cuBLASOperations.cpp` and a few compiler flags) it can be compiled and installed without it.
