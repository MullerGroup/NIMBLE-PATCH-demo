# NIMBLE-PATCH-demo
Python implementation, simulation, and result parsing scripts for the NIMBLE-PATCH algorithm.

The repository includes two scripts to reproduce the results presented in the paper and provide the infrastructure
to allow the comparison of NIMBLE-PATCH with other algorithms or adapt NIMBLE-PATCH for other use cases.

## NIMBLE-PATCH-demo.py
This script presents the simulation environment used to compare NIMBLE-PATCH with Gerchberg-Saxton (both GSx1 and GSx3 as described in the paper)
in the context of point cloud generation. The script is initially configured for comparison across four cases:

* 64x64 SLM, 4 targets
* 64x64 SLM, 16 targets
* 128x128 SLM, 4 targets
* 128x128 SLM, 16 targets

By default, GSx1 is selected and GS-based holograms will use this lighter-weight variant of the GS algorithm.
`cgh_gs_pix_sampling` key of the `system` dictionary can be set to `3` to switch to GSx3.

When run, the target depth planes (and two neighboring planes for each target plane) will be plotted and displayed, with targets marked as a red circle representing the spot size. Target planes without the red circles are also plotted and displayed, and can be saved and used for further processing.

## NIMBLE-PATCH-plots.py
This script reproduces the plots presented in the paper from the sweep result files located under the `results` directory.
Please refer to the paper for the legend of the plots.
The plots are, by default, saved under the `processed_results` directory once the script is run.
The names of the figure files indicate which figure number they correspond to within the paper.
