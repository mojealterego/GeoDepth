# OMEGA GeoDepth — architecture and data flow

```text
                         [Local DA3 weights + config]
                                      |
                            [Integrity verification]
                                      |
[Local photos] --> [Official DA3 API adapter] --+--> [Prediction contract]
                                                       ^
[External DA3 NPZ] --> [Strict bounded NPZ reader] -----+
                                                       |
                                              [Pinhole geometry]
                                                       |
                                      [Multiview visibility + consistency]
                                                       |
                                         [Quality gates and scale policy]
                                                       |
                       [Metric anchors] --> [Robust scale calibration]
                                                       |
                                            [Optional conservative depth consensus]
                                                       |
                                            [Confidence quantile mask]
                                                       |
                                           [Bounded XYZ point sampling]
                                                       |
                                              [Deterministic voxel fuse]
                                                       |
                              [PLY + optional quality NPZ + SHA-256 report]
                                                       |
                                     [Atomic no-overwrite publication]
```

Trust boundaries: user input (images, NPZ, anchors, config, weights) is untrusted; only fixed-schema arrays and SafeTensors are read; external upstream inference engine is an explicit separately installed dependency. A quality gate rejects inadequate evidence rather than producing falsely certified measurements.

Complexity: reprojection is O(N * min(N-1,K) * H * W) with `K = max_targets_per_view`. Geometry export samples at most `max_points` eligible pixels; fusion uses deterministic voxel grouping. Input dimensions and uncompressed archive bytes are capped. No GPU benchmarks are claimed.

Remaining risks: the external DA3 implementation and its dependencies require separate supply-chain review; model quality is dataset-dependent; dynamic scenes and camera/depth inconsistencies may defeat reprojection checks; scale calibration is limited to anchors; relative coordinates are not a surveyed 3D model.
