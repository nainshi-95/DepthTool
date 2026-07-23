Analyze the uploaded codec source and create a concrete implementation plan for the following three independent GlobalMotion optimizations.

Do not modify the code yet. First inspect the actual source, identify the relevant files, functions, data structures, and encoder/decoder call paths, then provide a concise file-by-file modification plan.

All changes must be under #if GlobalMotion.

Add three compile-time switches as simple #defines in TypeDef.h:

#if GlobalMotion
#define GM_DEPTH_PLANE_2X2_AVG        0
#define GM_HOMOGRAPHY_SUBBLOCK_MV     0
#define GM_EXISTING_SUBBLOCK_MC       0
#endif

Each switch must be independently configurable. When a switch is 0, the corresponding legacy behavior must remain unchanged.

All eight combinations from 000 to 111 must build and run.

Do not add new bitstream syntax.

⸻

1. GM_DEPTH_PLANE_2X2_AVG

Current inverse-depth plane:

\rho(x,y)=\frac{1}{Z(x,y)}=ax+by+c

Keep the existing centered least-squares plane model, but replace pixel-wise fitting samples with one averaged sample per 2×2 region.

Requirements:

* Average only valid inverse-depth samples inside each 2×2 region.
* Skip a 2×2 region if it contains no valid samples.
* Use the mean coordinate of the valid samples as the sample position.
* Handle odd CU width or height correctly.
* Do not allocate a temporary downsampled depth map.
* Accumulate directly into the existing fitting statistics.
* Preserve all existing validity checks, fallback behavior, clipping, and quantization.
* If subsampling breaks assumptions such as symmetric coordinates or zero cross terms, use the existing general centered-LS handling or a fixed-size 3×3 solver. Do not introduce SVD, dynamic matrices, OpenCV, or heap allocation.

Macro behavior:

#if GM_DEPTH_PLANE_2X2_AVG
  // 2x2 averaged fitting
#else
  // exact legacy pixel-wise fitting
#endif

Measure:

* Plane-fitting time
* Number of original pixels and averaged samples
* Difference in a, b, and c
* Reconstructed inverse-depth error
* Final coding impact

⸻

2. GM_HOMOGRAPHY_SUBBLOCK_MV

The current method derives one MV per 4×4 subblock by projecting the four corners and averaging the four MVs.

Replace this optionally with:

CU inverse-depth plane + K/R/t
→ build one CU homography
→ evaluate the homography once at each 4×4 center
→ derive one MV per 4×4

Expected complexity change:

4N camera projections
→ 1 homography build + N homography evaluations

Requirements:

* Derive the homography from the exact camera and depth conventions used by the current code.
* Confirm pose direction, K convention, depth definition, pixel-center convention, MV direction, and internal MV precision.
* Support each used L0/L1 reference independently.
* Use a fixed-size 3×3 representation without dynamic matrix libraries.
* Evaluate the homography at the proper 4×4 center.
* Reuse the existing MV rounding, clipping, precision conversion, and boundary handling.
* Handle invalid denominator, NaN, infinity, invalid plane, and projection failure.
* Use the legacy corner-projection path as fallback where appropriate.

Macro behavior:

#if GM_HOMOGRAPHY_SUBBLOCK_MV
  // CU homography + 4x4 center evaluation
#else
  // exact legacy four-corner projection and MV averaging
#endif

Validation must compare:

A: legacy average of four corner MVs
B: direct camera projection at the 4x4 center
C: homography evaluation at the 4x4 center

B and C should match within numerical tolerance.

Measure:

* Legacy projection time
* Homography build time
* Homography evaluation time
* MV error
* Prediction SAD/SATD
* Coding impact

⸻

3. GM_EXISTING_SUBBLOCK_MC

Reuse the codec’s existing 4×4 subblock motion-compensation path instead of the current GlobalMotion-specific interpolation path.

Inspect and verify the actual path, expected to be similar to:

xSubPuMC()
→ motionCompensation()
→ xPredInterUni() / xPredInterBi()
→ xPredInterBlk()
→ InterpolationFilter

Confirm from the source that this path uses:

* 4×4 MotionInfo
* Existing SIMD dispatch
* Luma 12-tap interpolation
* Chroma 6-tap interpolation

Requirements:

* Fill the CU MotionBuf with one MotionInfo per 4×4 subblock.
* Correctly set interDir, refIdx, MV, slice information, BCW, LIC, and all required fields.
* Support L0, L1, and bi-prediction.
* Add an explicit GlobalMotion branch instead of pretending the CU is SUBPU_ATMVP.
* Prevent later spanMotionInfo() or similar code from overwriting the custom 4×4 MotionBuf.
* Preserve existing prediction buffers, intermediate precision, weighted prediction, clipping, chroma scaling, and reference boundary handling.
* Apply the same path in encoder mode evaluation, encoder final reconstruction, and decoder reconstruction.

Macro behavior:

#if GM_EXISTING_SUBBLOCK_MC
  // fill 4x4 MotionBuf and use existing subblock MC
#else
  // exact legacy GlobalMotion interpolation
#endif

Measure:

* MotionBuf fill time
* Legacy GlobalMotion interpolation time
* Existing subblock MC time
* Number of 4×4 calls
* Number of merged adjacent subblocks
* Prediction mismatches
* Coding impact

⸻

Compatibility requirements

The following configurations must all work:

000: legacy baseline
100: 2x2 fitting only
010: homography MV only
001: existing subblock MC only
110: 2x2 fitting + homography
101: 2x2 fitting + existing subblock MC
011: homography + existing subblock MC
111: all optimizations

When all macros are 0, results must remain bit-exact with the current baseline.

No new signaling is allowed.

Encoder and decoder must generate identical predictions for every enabled combination.

Avoid dynamic allocation in the CU hot path.

⸻

Profiling

Optionally add:

#if GlobalMotion
#define GM_OPT_PROFILE 0
#endif

When enabled, measure at least:

GlobalMotion total time
Plane-fitting time
Corner-projection time
Homography-build time
Homography-evaluation time
MotionBuf-fill time
Legacy interpolation time
Existing subblock-MC time
Processed CU count
Processed 4x4 subblock count

Use thread-local or worker-local profiling storage. Avoid atomics in the hot path.

⸻

Required output

Provide a concise implementation plan containing:

1. Current GlobalMotion call flow
2. Relevant files, classes, and functions
3. Exact location for the TypeDef.h macros
4. Modification plan for each optimization
5. Encoder and decoder changes
6. Interaction between the three macros
7. Profiling insertion points
8. Bit-exact and mismatch validation
9. Main implementation risks
10. Recommended implementation order

Use actual source names found in the uploaded code. Do not guess file or function names.
