# Fuid Advecction (OpenMP & CUDA)

Parallel advection solver over shared address space programming models. Programming models focus on OpenMP and CUDA implementations.


## Setup

This project contains a report called `ps-ass2Rep.pdf`,  which discusses questions posed in [docs/assignment_outline.md](./docs/assignment_outline.md). It also contains two sub-directories, `openmp` and `cuda`. The former contains a test program `testAdvect.c`, a file `serAdvect.c` containing serial advection functions, some header files, and a template OpenMP advection solver `parAdvect.c`. The test program can be built using the command `make`.

The usage for the test program is:

```bash
OMP_NUM_THREADS=p ./testAdvect [-P P] [-x] M N [r]
```

The `-P` option invokes an optimization where the parallel region is over all timesteps of the simulation, and *P* by *Q* block distribution is used to parallelize the threads, where *p=PQ*. The `-x` option is used to invoke an optional extra optimization.

The directory cuda is similar, except the test program is called `testAdvect.cu`, and the template CUDA parallel solver file is called `parAdvect.cu`. The usage for the test program is:

```bash
./testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] [-d d] M N [r]
```

with default values of `Gx=Gy=Bx=By=r=1` and `v=w=d=0`. `Gx,Gy` specifies the grid dimensions of the GPU kernels; `Bx,By` specifies the block dimensions.

The option `-h` runs the solver on the host; this may be useful for comparing the 'error' of the GPU runs (and for comparing GPU and CPU speeds). The option `-s` forces a serial implementation (run on a single GPU thread); all other options are ignored. If neither of `-h,-s,-o` are given, `Gx,Gy` thread blocks of size `Bx,By` are used in a 2D GPU parallelization of the solver. If `-o` is specified, an optimized GPU implementation is used, which may use the tuning parameter w as well.

The option `-d` can be used to specify the id of the GPU to be used (`stugpu2` has 4 GPUs, so you can use `d` equal to either `0`, `1`, `2`, or `3`). This may be useful if a particular GPU (e.g. GPU 0) is currently loaded.
