# AutoRCWA_Net
An evolving deep learning network combining RCWA.

This model contrain the [Simple_RCWA](https://github.com/GuoyaoShen/Simple_RCWA) package as a submodule. Changes might be applied in this [submodule](https://github.com/GuoyaoShen/AutoRCWA_Net/tree/main/Simple_RCWA) for different geometry & material structure.

This repo is still under construction, changes might be applied.

Can you "learn to design and design to learn" meta-material using neural networks? That's what we're doing now! In general, we're building a close-loop system combining RCWA method and neural networks to teach the network how to design meta-material with continuous data flow.

## Dependencies
* numpy
* matplotlib
* scipy
* torch
* [cupy](https://cupy.dev/), for GPU acceleration of RCWA
