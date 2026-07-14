Before writing any code, do not implement or modify anything.

First inspect the codec repository and the attached Python simulation script, then produce a detailed implementation plan for integrating the Camera+Depth prediction tool.

You may search and read repository files, trace call paths, and inspect existing implementations. Do not edit files, generate patches, or write implementation code.

1. Source-of-truth rules

The attached Python script is an algorithmic reference, not the final codec specification.

Use it to understand:

* Camera parameter representation and relative transforms
* The inverse-depth plane model
* MV-to-depth reconstruction
* Forward projection of reconstructed reference depth
* Plane fitting
* Plane quantization
* Camera projection
* Projection-domain warped-Y SATD evaluation

Do not copy its architecture directly.

The Python script contains experimental behavior that does not match the final codec design, including:

* Separate left, top, top_left, and spatial_all syntax candidates
* Separate fw_ref_<POC> candidates
* A manually maintained categorical probability model
* Generic best-reference selection using original-Y SATD
* Fixed block sizes
* Implicit zero-bit DepthReuseBuffer
* Zero-anchor simulation rules
* A single generic predictor-residual mode instead of separate C-only and full residual modes

Start your response with a table describing the differences between the Python simulator and the intended codec design.

2. Inspect the current codec first

Before proposing an architecture, locate and analyze the existing code related to:

* Camera projection or CamProj prediction
* Camera parameter storage and parsing
* CU-level CamProj flags and state
* Encoder inter-mode RDO
* Decoder inter reconstruction
* MotionInfo and MotionBuf
* spanMotionInfo
* Spatial Merge candidate derivation
* HMVP insertion and reuse
* L0/L1 reference-list handling
* CABAC writer, reader, estimator, and context definitions
* Existing plane, depth, or geometry-related utilities

The repository may already contain part of the Camera Projection tool. Reuse and extend the existing implementation rather than creating a parallel duplicate tool.

For every proposed change, identify exact existing:

* Files
* Classes
* Functions
* Call sites
* Data structures

Do not give only generic architectural recommendations.

3. Final predictor architecture

The predictor families have a permanently fixed logical order:

1. Forward Average
2. MV Sample Best
3. Forward Single
4. Neighbor Depth Plane
5. History Plane

The predictor order must never change by frame, slice, camera motion, or sequence.

Unavailable candidates are removed before candidate-index coding.

Example:

Logical order:

* Forward Average: unavailable
* MV Sample Best: available
* Forward Single: available
* Neighbor Depth Plane: available
* History Plane: unavailable

The coded available list becomes:

* index 0: MV Sample Best
* index 1: Forward Single
* index 2: Neighbor Depth Plane

The encoder and decoder must independently generate exactly the same available list in exactly the same order.

Propose:

* Candidate data structures
* Candidate availability rules
* Candidate-list generation functions
* Encoder/decoder shared utility placement
* Assertions or debug checks that verify candidate-list equality

4. Decoder determinism requirement

No candidate may contain a hidden encoder-only decision.

In particular, do not select an internal left, top, top-left, or other subcandidate using original-picture SATD unless that choice is explicitly signaled.

Review the intended meaning of MV Sample Best.

The preferred direction is one decoder-reproducible candidate derived from available neighboring MotionInfo, for example:

* Gather all valid neighboring MV observations
* Convert the observations to depth samples
* Reject unreliable samples using decoder-available checks
* Fit one plane using a deterministic rule

If a meaningful “best” selection cannot be reproduced by the decoder, identify the problem and propose either:

* A deterministic combined derivation, or
* Minimal additional syntax

Apply the same determinism requirement to:

* Neighbor Depth Plane
* History Plane
* Forward Average
* Forward Single

5. Forward Single

Forward Single is one predictor family.

It may internally have up to two alternatives:

* The nearest valid L0 reference
* The nearest valid L1 reference

For each list, select only the nearest reference using:

1. Minimum absolute POC distance from the current picture
2. Smaller refIdx as the deterministic tie-breaker

Do not signal the refIdx.

When both L0 and L1 alternatives are valid:

* The encoder evaluates both
* One CABAC-coded list flag selects L0 or L1

When only one list is valid:

* Infer the list
* Do not code the list flag

Propose:

* How the nearest reference is derived
* How both alternatives are represented during encoder RDO
* How they remain one predictor index
* Where the selected list is stored in CU state
* CABAC syntax and context for the list flag
* Decoder reconstruction behavior

6. Forward Average

Forward Average should use reconstructed reference depth, never GT depth.

Analyze the cleanest definition using the nearest valid L0 and L1 reconstructed-depth references.

Explicitly address:

* Whether Forward Average requires both L0 and L1
* Whether it is allowed when only one side is valid
* How duplication with Forward Single is avoided
* How valid and invalid forward-projected samples are combined
* How occlusion or multiple projected samples are resolved
* How its availability is reproduced identically by encoder and decoder

Recommend one exact normative rule.

7. Predictor source versus inter-prediction reference

Explicitly distinguish:

1. The reference picture used to generate a depth-plane predictor
2. The reference picture or pictures used for final Camera Projection inter prediction

The Python simulator currently evaluates candidate depth against several references and selects the lowest-SATD reference. Do not copy this behavior silently.

For each predictor family, define:

* Predictor source reference
* Final interDir
* Final L0/L1 refIdx
* Whether the predictor source and final prediction reference must be the same
* Which values are inferred and which values require syntax

This must be fully defined before implementation.

8. Plane representation and reconstruction

The depth plane is:

1 / z(x,y) = a * (x - cx) + b * (y - cy) + c

Analyze:

* The existing codec coordinate system
* CU-center definition
* Plane recentering
* Quantization of a, b, and c
* Fixed-point representation
* Rounding rules
* Clipping ranges
* Invalid inverse-depth handling
* Encoder/decoder numerical matching

The Python implementation uses floating point. The actual codec implementation must avoid encoder/decoder mismatch.

Propose one deterministic representation and identify where conversion from the Python formulas is required.

9. Plane coding modes

The final plane coding modes are:

* Predictor Only
* C-only Residual
* Full Residual
* Direct Plane

Definitions:

Predictor Only:

* Reconstructed plane equals the selected predictor
* No plane residual is coded

C-only Residual:

* a = predictor.a
* b = predictor.b
* c = predictor.c + delta_c

Full Residual:

* a = predictor.a + delta_a
* b = predictor.b + delta_b
* c = predictor.c + delta_c

Direct Plane:

* The absolute quantized a, b, and c are coded
* No predictor index is required

Propose the exact syntax tree.

In particular, determine whether plane mode should be coded before or after predictor index, while ensuring Direct Plane does not unnecessarily code a predictor index.

Propose:

* CABAC contexts
* Residual coefficient syntax
* Zero/nonzero coding
* Sign coding
* Magnitude coding
* Context sharing between a and b
* Separate treatment of c
* Encoder RDO bit estimation
* Decoder parsing order

10. Candidate-index CABAC coding

Candidate index must use the codec’s normal CABAC context adaptation.

Do not implement or retain the Python SimpleAdaptiveProb candidate model.

The preferred method is truncated unary coding over the compressed available-candidate list.

With five available candidates:

* index 0: 0
* index 1: 10
* index 2: 110
* index 3: 1110
* index 4: 1111

Use one CABAC context per decision level unless repository constraints suggest a better design.

Analyze and propose:

* Number of contexts
* Context initialization
* Writer implementation location
* Reader implementation location
* Estimator use during RDO
* Behavior when one candidate is available
* Behavior when only two or three candidates are available
* Context update behavior when the semantic candidate occupying index 0 changes because an earlier candidate is unavailable

The intended design learns the probability of the compressed ordinal index, not a separate probability for each predictor family.

Do not introduce availability-pattern-specific contexts in the first implementation.

11. Encoder processing flow

Describe the complete encoder flow and map every stage to exact repository functions.

Expected conceptual flow:

Depth Tool availability
→ Generate deterministic available predictor list
→ Generate plane coding alternatives
→ Reconstruct candidate plane
→ Generate candidate depth
→ Camera-project into reference picture
→ Generate warped Y predictor
→ Calculate projection-domain SATD
→ Estimate real CABAC fractional bits
→ Select promising Depth Tool candidates
→ Run the required normal inter residual/full-RD process
→ Select the final CU mode
→ Store reconstructed plane state
→ Generate final 4x4 MotionInfo
→ Allow normal Merge and HMVP reuse

Clarify the relationship between:

* Projection-domain SATD used by the simulator
* Fast encoder candidate pruning
* Existing codec full inter RD
* Transform and residual coding
* Final CU cost

Do not assume that the simulator’s simplified SATD + lambda * simulated_bits should directly replace the codec’s normal full RD.

12. Decoder processing flow

Describe the complete decoder call flow with exact insertion points:

Parse Depth Tool syntax
→ Generate the same available predictor list
→ Resolve Forward Single L0/L1 when applicable
→ Reconstruct the inverse-depth plane
→ Generate 4x4 depth
→ Camera projection
→ Generate 4x4 MV field
→ Populate MotionInfo
→ Perform normal inter prediction
→ Reconstruct residual
→ Update Merge/HMVP-related state

Identify which operations must be shared between encoder and decoder.

13. Motion generation and propagation

After the final plane is reconstructed:

* Generate depth for each 4x4 motion subblock
* Camera-project the subblock into the selected reference picture
* Generate codec-precision MV
* Set refIdx and interDir
* Store the result through the existing MotionBuf or spanMotionInfo path

Analyze whether the current Camera Projection implementation uses:

* Subblock center projection
* Four-corner projection and averaging
* Per-pixel projection followed by averaging
* Another existing method

Recommend one method consistent with the repository’s current CamProj behavior.

The final MotionInfo must automatically participate in:

* Spatial Merge
* HMVP
* Existing inter-neighbor lookup

Do not introduce a separate Depth Tool Merge candidate path unless the existing architecture makes it unavoidable.

Explain:

* How a nonuniform 4x4 MV field is stored
* How a representative HMVP MotionInfo is chosen
* When HMVP is updated
* Whether the existing HMVP maximum size remains unchanged
* How bi-pred MotionInfo is handled

14. History Plane

Define a decoder-reproducible History Plane mechanism.

Analyze and propose:

* History entry contents
* Maximum number of entries
* Insertion order
* Duplicate removal
* Reset points
* Slice/tile/subpicture restrictions
* Reference-list compatibility
* Recenter operation
* Candidate availability
* Encoder/decoder update timing

Do not select a history entry using original-picture distortion unless the selected history index is signaled.

If only one History Plane predictor is exposed, define a deterministic rule for selecting that entry.

15. Features from the Python simulator that are not automatically in scope

Do not assume the following experimental simulator features belong in the first codec implementation:

* Fixed block-size operation
* DepthReuseBuffer
* Implicit zero-bit depth reuse
* Zero-anchor POC rules
* Manual adaptive probability models
* Generic best-reference SATD selection
* Arbitrary numbers of forward references

For each feature, state whether it should be:

* Excluded
* Replaced by existing codec behavior
* Deferred to a later phase
* Retained for a specific reason

16. Required implementation roadmap

Split implementation into small, testable stages.

A suggested direction is:

1. Repository mapping and shared data structures
2. Deterministic fixed-point plane representation
3. One minimal predictor with Predictor-Only mode
4. Encoder/decoder candidate-list symmetry
5. Candidate-index CABAC
6. Forward Single and L0/L1 flag
7. Forward Average
8. MV Sample predictor
9. C-only, Full, and Direct plane coding
10. Projection-domain candidate RDO
11. 4x4 MotionInfo and spanMotionInfo
12. Spatial Merge reuse
13. HMVP reuse
14. Neighbor Depth Plane
15. History Plane
16. Complexity reduction and cleanup

You may change this order if repository dependencies require it.

For every stage provide:

* Exact files to modify
* Exact functions or classes to modify
* New data structures or functions
* Encoder behavior
* Decoder behavior
* CABAC changes
* Dependencies
* Expected debug output
* Unit or integration tests
* Completion criteria
* Main mismatch risks

17. Verification plan

Include a verification matrix covering at least:

* Encoder and decoder available-candidate lists are identical
* Candidate compressed indices are identical
* POC cases where Forward Average is unavailable
* Only L0 Forward Single is valid
* Only L1 Forward Single is valid
* Both L0 and L1 Forward Single are valid
* One available predictor, requiring zero candidate-index bits
* Predictor-Only reconstruction
* C-only residual reconstruction
* Full residual reconstruction
* Direct Plane reconstruction
* Fixed-point plane equality
* Per-4x4 depth equality
* Per-4x4 MV equality
* refIdx and interDir equality
* First reconstructed pixel mismatch tracing
* Bitstream encode/decode synchronization
* Spatial Merge reuse of Depth Tool MotionInfo
* HMVP insertion and later reuse
* Tile, slice, and CTU-boundary behavior

Recommend debug dumps that can compare encoder and decoder values block by block.

18. Required output format

Produce the plan in the following sections:

1. Existing repository architecture map
2. Python simulator versus final codec discrepancy table
3. Proposed normative Depth Tool design
4. Data structures
5. Syntax and CABAC design
6. Encoder call flow
7. Decoder call flow
8. Predictor derivation rules
9. MotionInfo, Merge, and HMVP integration
10. File-by-file modification plan
11. Incremental implementation stages
12. Verification and debugging plan
13. Risks, ambiguities, and decisions required before coding

Do not write implementation code.

Do not modify repository files.

Do not provide pseudocode longer than necessary to explain interfaces.

Every recommendation must be grounded in the actual repository structure and reference exact symbols where possible.
