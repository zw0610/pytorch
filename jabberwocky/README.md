Instructions:

- pytorch setup.py develop
- mkdir pytorch/jabberwocky/build
- cd pytorch/jabberwocky/build
- CMAKE_PREFIX_PATH=<...>/pytorch/torch cmake ..
- make

Replace <...> in the above with the start of your path to the pytorch directory.
This should compile and dynamically link the scripts, which can be run from
the build directory.

To build again, delete the pytorch/jabberwocky/build directory and repeat the
process.

Relevant Jabberwocky files are in c10/core/jabberwocky, plus the interface
header (c10/core/Jabberwocky.h).