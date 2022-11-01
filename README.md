# OpenMP and MPI parallelization (task and data) of NPC problems
The NPC problem is generalized bisection width.

I have created basic sequential solution (recursion), then parallelized it using task parallelism in OpenMP. Afterward, I have changed the solution to iterative and used data parallelism in OpenMP. In the end, I combined both solution to take advantage of task parallelism in MPI and data parallelism in OpenMP to solve the problem faster on a supercomputer.
