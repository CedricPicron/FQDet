/*
Binds CUDA deformable functions to python.
*/

#include "deformable.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("msda_3d_forward", &msda_3d_forward, "Forward method of 3D multi-scale deformable attention");
    m.def("msda_3d_backward", &msda_3d_backward, "Backward method of 3D multi-scale deformable attention");
}
