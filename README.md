# Boundary First Flattening - Rust Reference Implementation

This is a reference implementation of the [Boundary First Flattening](https://geometrycollective.github.io/boundary-first-flattening/) algorithm published by Rohan Sawhney and Keenan Crane in 2017.  

## Implementation

The original [BFF code on Github](https://github.com/GeometryCollective/boundary-first-flattening) is written in C++ and dynamically links to the SuiteSparse library.  I've had success compiling it on Ubuntu 22.04 and 24.04 but have not been able to successfully build it on Windows or on other Linux distributions.  The code has a lot of extra features and is somewhat difficult to follow.

More recently, a [Numpy/SciPy based Python implementation](https://github.com/russelmann/confmap) by Ruslan Guseinov became available on Github.  It has a reduced feature set and is more compact.  However, it can still be difficult to follow because of the necessity of expressing every expensive computation as a vectorized `numpy` operation, and because of the indirection from a three-layer-deep inheritance hierarchy and the heavy use of function side effects to store and retrieve data.

This implementation is a reference for anyone interested in translating the BFF algorithm to Rust.  It is extremely barebones, only performing the minimum distortion version of the algorithm on disk topologies, and without the cone splitting features.  

Furthermore, I structured the code to be as straightforward and procedural as possible to make it easy to follow.  Self-contained operations are put into functions with clear inputs and outputs. Inputs that are used more than once are calculated in a common area and passed into the functions that use them as arguments.  It should be relatively easy to see exactly what quantities need to be computed and where they are used.

A commented version of the overall sequence is in `main.rs`, which compiles into a binary that will operate on any Wavefront `.obj` file, but will default to running on the `bumpcap.obj` sample file in the main repository root.

Also for reference, there is an `end_to_end()` test function in the main `lib.rs` which runs the entire algorithm on a msgpack encoded version of the `hyperboloid.obj` file from the [confmap](https://gitub.com/russelmann/confmap) repository. 

## Reference

To verify the correctness of this translation, I used the Python implementation as a baseline and modified it to save the results of critical intermediate computations to files.  These files are all included in the `src/test_data.zip` archive. The various modules have tests which read these files out of the archive and compare them to the results of calculation functions.  

The original test file was the `hyperboloid.obj` file from the [confmap](https://gitub.com/russelmann/confmap) repository, and the test confirms that the full process produces the same results within 1.0e-6 of the Python version.
