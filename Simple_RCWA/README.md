# Simple_RCWA
A simple python version of RCWA (Rigorous Coupled‐Wave Analysis).


## GPU acceleration:
Install [cupy](https://cupy.dev/) to enable the GPU version of RCWA.


## How to use:
* "rcwa_Si_test.py" provides an example of how to use this package.
* To define a new structure, use the class "Material":
```python
from utils import rcwa_utils
new_material = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, list_layer_params, source, device='cpu', use_logger=True)
```
        freq: numpy array of frequencies to solve rcwa, (N_freq,).
        params_eps: list of eps for all layers, each entry in the list is shape of (N_freq,).
        params_geometry: [Lx,Ly,[d1,...,dn]], 2D geometry params and thickness for all layers.
        params_mesh: [Nx,Ny], mesh number for 2D geometry.
        PQ_order: a list of [PQ_x, PQ_y], each entry should be a singular value.
        list_layer_funcs: a list of functions [f1,...,fn] applied to each layer to define patterns inside each layer,
                          deleting materials.
        list_layer_params: a list of params [[l1_p1,...,l1_pm1],...,[ln_p1,...,ln_pmn]], each entry is also a list for
                          the i th layer corresponding to the "list_layer_funcs". They should already contain units
                          inside (eg, millimeters, micrometres, etc).
        source: source of incident light, a list, [ginc, EP], each entry (ginc, EP) is also a list, both ginc and EP
                should be a unit vector.
        device: 'cpu' for CPU using numpy; 'gpu' or 'cuda' for GPU using cupy.
        use_logger: printing solving progress percentage.

* For each layer in the material, a corresponding layer function and input params are needed in order to define patterns inside each layer (which are [f1,...,fn] and [l1_p1,...,l1_pm1] shown above). Function "layerfunc_Si_square_hole" and "layerfunc_absorber_ellipse_hole" (both located at utils.rcwa_utils) provide a template of how to implement this. In general, applying a function for each layer provides much higher flexibility for the patterns. For example, "layerfunc_Si_square_hole" defines a ellipse
hole at the center of the corresponding layer.


## Some example spectras:

Reflection  |  Transmission
:-------------------------:|:-------------------------:
![](https://github.com/GuoyaoShen/AutoRCWA_Net/blob/main/Simple_RCWA/figs/R.png)  |  ![](https://github.com/GuoyaoShen/AutoRCWA_Net/blob/main/Simple_RCWA/figs/T.png)
![](https://github.com/GuoyaoShen/AutoRCWA_Net/blob/main/Simple_RCWA/figs/R_absorber_ellipse.png)  |  ![](https://github.com/GuoyaoShen/AutoRCWA_Net/blob/main/Simple_RCWA/figs/T_absorber_ellipse.png)
