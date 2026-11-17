# OMEGA Scene5D — mathematical / computational contract

## The five axes

`F(X,Y,Z,t,L)`, where `X,Y,Z` are 3D world coordinates, `t` is seconds and `L` is the observed sRGB luminance proxy `(0.2126 R + 0.7152 G + 0.0722 B)/255`. `L` is not radiance in W/sr/m², a latent learned dimension, or an extra physical coordinate.

Each frame supplies calibrated `K` and camera-to-world `T_c2w`, images, positive optical-axis depth, pixel validity and time. For pixel `p=(u,v,1)`, `Xc = Z K^-1 p`, `Xw = T_c2w Xc`. If K is approximate, results are approximate. Extrinsics must be SE(3); input w2c must be explicitly inverted.

## Reconstruction

1. Verify dimensions, timestamps, positive depths, image sizes and legal provenance.
2. Backproject observed pixels using K and T_c2w; preserve masks.
3. Within each time slice, aggregate points into bounded spatial voxels by weighted averaging. Weight comes from logged uncertainty or an *uncalibrated* DA3 confidence rank.
4. For consecutive frames, find nearest 3D voxel centers (bounded KD-tree distance) and derive `v=(X_{t+1}-X_t)/dt`. These are **unverified** correspondences and may confuse camera occlusion, repeated surfaces, and moving objects. Unmatched velocities remain NaN.
5. For each frame, triangulate depth-connected pixels (no triangles across large depth discontinuities).
6. Optional projective TSDF per timestamp with limited voxel budget, preserve zero-weight unknown voxels, remove fake boundary surfaces, mesh via marching cubes. **Do not fuse dynamic frames into one static volume.**
7. Optional temporal interpolation warps only points with matched flow, then uses calibrated projection and nearest-point Z-buffer. Unknown output pixels remain black in visualization and NaN in depth+mask.
8. Export `scene5d.npz`, custom PLY time and flow properties, per-frame OBJ/TSDF, intermediate renders, and SHA-256 report.

## Scope and guarantees

This is a dynamic 3D + time + appearance **observational reconstruction**, not an optimized learned dynamic Gaussian Splatting, neural radiance field, physically accurate multi-view capture system, or reconstruction of unseen surfaces. Correct world positions depend on K, T and depth: an unanchored monocular or multiview reconstruction is only up to unknown similarity/scale. `arbitrary_scale` and `relative_estimated` must be retained until an independently verifiable reference is available.

Demo is a three-frame synthetic raycast with moving sphere and known camera pose. Its exact 5D output and test logs are evidence for computation and format validity, not for accuracy on real photography.
