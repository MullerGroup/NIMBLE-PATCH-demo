# NIMBLE-PATCH-demo
Python implementation, simulation, and result parsing scripts for the NIMBLE-PATCH algorithm.

The repository includes two scripts to reproduce the results presented in the paper, and provide the infrastructure
to allow the comparison of NIMBLE-PATCH with other algorithms or adapt NIMBLE-PATCH for other use cases.

## NIMBLE-PATCH-demo.py
This script presents the simulation environment used to compare NIMBLE-PATCH with Gerchberg-Saxton (both GSx1 and GSx3 as described in the paper)
in the context of point cloud generation. The script is initially configured the comparison across four cases:

* 64x64 SLM, 4 targets
* 64x64 SLM, 16 targets
* 128x128 SLM, 4 targets
* 128x128 SLM, 16 targets

By default, GSx1 is selected and GS-based holograms will use this lighter weight variant of the GS algorithm.
`cgh_gs_pix_sampling` switch of the `system` dictionary can be set to `3` to switch to GSx3.
