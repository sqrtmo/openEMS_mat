# Material library for openEMS
example:
```python
from openEMS_mat.mat import Material
  
# substrate element
FR4 = Material.from_library( "FR4_lossy" )
substrate = FR4.add_to_csx( CSX, f0 )
substrate.AddBox( priority=0, start=[-25, -28, 0], stop=[25, 28, -substrate_thickness] )
```
