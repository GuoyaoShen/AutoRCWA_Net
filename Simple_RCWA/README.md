# Simple_RCWA
A simple python version of RCWA (Rigorous Coupled‐Wave Analysis).


## GPU acceleration:
Install [cupy](https://cupy.dev/) to enable the GPU version of RCWA.


## How to use:
* "rcwa_Si_test.py" provides an example of how to use this package.
* To define a new structure, use the class "Material":
```python
from utils import rcwa_utils
new_material = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, device='cpu', use_logger=True)
```
freq: numpy array of frequencies to solve rcwa, (N_freq,)
params_eps: list of eps for all layers, each entry in the list is shape of (N_freq,)
params_geometry: [Lx,Ly,[d1,...,dn]], 2D geometry params and thickness for all layers
params_mesh: [Nx,Ny], mesh number for 2D geometry
PQ_order: a list of [PQ_x, PQ_y], each entry should be a singular value
list_layer_funcs: a list of functions [f1,...,fn] applied to each layer to define patterns inside each layer, deleting materials
device: 'cpu' for CPU using numpy; 'gpu' or 'cuda' for GPU using cupy
use_logger: printing solving progress percentage

* For each layer in the material, a corresponding layer function is needed in order to define patterns inside each layer (which is f1,...,fn above).
Function "layerfunc_Si_square_hole" and "layerfunc_absorber_ellipse_hole" provide a template of how to implement this. In general, applying a function
for each layer provides much higher flexibility for the patterns on each layer, for example, "layerfunc_Si_square_hole" contains a ellipse hole at the
center of the corresponding layer.


## Some example spectras:

Reflection:

![](https://github.com/GuoyaoShen/Simple_RCWA/blob/main/figs/R.png)

Transmission:

![](https://github.com/GuoyaoShen/Simple_RCWA/blob/main/figs/T.png)
