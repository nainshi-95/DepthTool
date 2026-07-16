Analyze the entire codec source and prepare a detailed implementation plan for adding an MMVD-like 2D refinement mechanism to the existing camera/depth projection-based prediction tool.

Do not implement the feature yet.

Your task is to:

1. Analyze the current projection tool implementation and call flow.
2. Analyze the existing MMVD implementation and call flow.
3. Identify which MMVD design concepts can be reused.
4. Design a completely independent refinement framework dedicated to the projection tool.
5. Produce a file/function-level implementation plan.

Do not assume any projection-related flag names, mode names, function names, or data structures before analyzing the actual source code. Use the real identifiers only after locating them.

⸻

1. Goal

The existing tool roughly performs the following process:

Current pixel + depth
→ 3D camera transformation
→ Projection into the reference frame
→ Reference sampling
→ Prediction generation

One of the existing prediction paths roughly behaves as follows:

Use an existing merge candidate
→ Generate projection-based prediction
→ Skip residual coding

The goal is to extend this path by introducing a small image-space refinement.

Projected reference position
+ small horizontal/vertical displacement
→ final sampling position

The purpose is to compensate for small projection errors caused by camera parameter inaccuracies, depth inaccuracies, and quantization errors without transmitting residuals.

Initially, consider the following refinement candidates:

Directions
- right
- left
- down
- up
Steps
- 1/4 pel
- 1/2 pel
- 1 pel
- 2 pel

This gives

4 directions × 4 steps = 16 refinement candidates

If a better candidate structure exists after analyzing the actual implementation, explain why and propose it.

⸻

2. Most Important Design Rules

2.1 Never modify the existing MMVD implementation

The existing MMVD implementation must be used only as a reference.

Do not modify:

* MMVD data structures
* MMVD index definition
* MMVD candidate generation
* MMVD delta generation
* MMVD merge candidate setup
* MMVD encoder RDO
* MMVD CABAC writer
* MMVD CABAC reader
* MMVD context models
* MMVD decoder
* MMVD MotionInfo generation
* MMVD syntax
* MMVD candidate configuration

Do not insert projection-related conditions inside existing MMVD functions.

For example, the following approach is forbidden:

void ExistingMmvdFunction(...)
{
#if GlobalMotion
    if (projectionCondition)
    {
        ...
    }
#endif
    ...
}

The existing MMVD bitstream, encoder behavior, decoder behavior, and coding performance must remain completely unchanged.

⸻

2.2 Use MMVD only as a design reference

The following MMVD concepts may be reused conceptually:

* refinement index structure
* direction + step representation
* internal MV precision handling
* unary/truncated-unary syntax design
* SATD-based candidate pruning
* full-RD workflow
* encoder/decoder symmetric refinement reconstruction
* MotionInfo update strategy

However, all implementation must be newly created specifically for the projection tool.

If necessary, create entirely new:

* refinement index
* delta generation function
* candidate generation function
* CU state
* syntax
* CABAC writer
* CABAC reader
* context models
* encoder RDO
* decoder reconstruction
* debug/statistics

Do not directly reuse MMVD-specific classes or functions.

General codec utilities (for example MV precision constants or common helper functions) may be reused, but nothing should modify or depend on MMVD-specific states or probability models.

⸻

3. Every modification must be protected by #if GlobalMotion

This is mandatory.

Every new or modified code section must be enclosed by

#if GlobalMotion
...
#endif

This includes:

* constants
* enums
* new data structures
* CU members
* prediction states
* declarations
* implementations
* encoder candidate generation
* encoder RDO
* decoder code
* CABAC writer
* CABAC reader
* context models
* syntax
* initialization
* copy/reset functions
* MotionInfo update
* configuration
* tracing
* debug code
* statistics
* call-site modifications

For example,

#if GlobalMotion
// new CU members
#endif

Initialization:

#if GlobalMotion
...
#endif

Function declarations:

#if GlobalMotion
void NewProjectionRefinementFunction(...);
#endif

Function definitions:

#if GlobalMotion
void NewProjectionRefinementFunction(...)
{
    ...
}
#endif

When GlobalMotion == 0, the codec must:

* compile successfully
* expose no new symbols
* preserve identical behavior
* preserve identical bitstreams
* introduce no unused variables
* introduce no linker errors

For every proposed modification, explicitly specify where #if GlobalMotion should be placed.

⸻

4. MMVD Analysis

Analyze the complete MMVD pipeline.

The overall flow is approximately:

Merge candidate generation
→ base candidate selection
→ refinement index decoding
→ delta MV generation
→ motion refinement
→ motion compensation
→ SATD pruning
→ full RD
→ CABAC encoding
→ decoder reconstruction
→ MotionInfo generation

At minimum, trace:

* MMVD index definition
* base candidate generation
* delta MV generation
* CU setup
* encoder first-pass candidate generation
* SATD/SAD pruning
* full RD
* residual and skip paths
* bit estimation
* CABAC writer
* CABAC reader
* context models
* decoder reconstruction
* MotionInfo generation

For every stage, document:

File
Function
Inputs
Outputs
State changes
Next function

⸻

5. Projection Tool Analysis

Do not assume any existing identifiers.

Find the actual implementation and analyze:

State & Syntax

* tool enable conditions
* merge-related path
* residual-free path
* candidate indices
* reference list handling
* CU state
* initialization/reset/copy
* SPS/PPS/picture-level enable
* syntax order

Encoder

* candidate generation
* merge integration
* residual-free path
* first-pass distortion
* pruning
* full RD
* bit estimation
* final mode decision
* reconstruction

Prediction

* depth loading
* camera loading
* 3D transformation
* projection
* reference coordinate generation
* interpolation
* boundary handling
* invalid handling
* uni-pred
* bi-pred
* prediction buffer generation

MotionInfo

* how motion is stored
* block/subblock motion
* MotionInfo generation
* merge reuse
* temporal reuse
* HMVP
* affine interaction
* filtering dependencies

⸻

6. Projection Refinement Concept

Traditional MMVD performs

Final MV
= Base MV + Delta MV

The proposed refinement should instead perform

Final projection motion(x,y)
=
Original projection motion(x,y)
+
Common refinement delta

or equivalently,

Final projected coordinate(x,y)
=
Original projected coordinate(x,y)
+
Common refinement offset

Since projection motion varies spatially, simply modifying one block MV is insufficient.

The refinement must affect the actual projected coordinates or all subblock motions used for prediction.

⸻

7. Independent Projection Refinement Design

Dedicated refinement index

Create a new refinement index containing at least:

step
direction

If the existing implementation already signals a merge candidate separately, do not duplicate the base candidate index inside the refinement index unless necessary.

Initial refinement candidates:

Steps
- 1/4 pel
- 1/2 pel
- 1 pel
- 2 pel
Directions
- right
- left
- down
- up

Design the bit layout according to the codec coding style.

⸻

Dedicated delta generation

Create a completely new delta-generation function.

Do not reuse the MMVD delta-generation function directly.

The new function should approximately perform

Refinement index
→ Decode step
→ Decode direction
→ Return image-space delta using internal MV precision

Reuse only common codec utilities if appropriate.

⸻

Dedicated candidate generation

Create a completely independent candidate-generation path.

The intended flow is

Generate base projection prediction
→ Evaluate non-refined candidate
→ Evaluate all refinement candidates
→ First-pass cost
→ Candidate pruning
→ Full RD

The MMVD pruning strategy may be referenced conceptually, but candidate management should remain completely independent.

⸻

Dedicated syntax & CABAC

Do not modify or reuse MMVD syntax.

If required, introduce new syntax for:

Refinement enable
Refinement index

Create new CABAC contexts instead of sharing MMVD contexts.

The statistics of this tool must never affect MMVD probability adaptation.

⸻

8. Refinement Application Position

The preferred order is

Camera/depth projection
→ Projected coordinates
→ Add refinement
→ Boundary/validity check
→ Reference interpolation

Avoid

Projection
→ Boundary clipping
→ Refinement

Determine the safest implementation according to the actual coordinate representation used by the codec.

Prefer internal MV precision or fixed-point arithmetic whenever possible to guarantee encoder/decoder consistency.

⸻

9. MotionInfo Update

Updating only prediction while leaving MotionInfo unchanged is not acceptable.

The stored motion must satisfy

Stored motion
=
Original projection motion
+
Refinement delta

If subblock motion exists, apply the same refinement to every valid subblock.

Analyze the impact on:

* spatial merge
* temporal merge
* HMVP
* affine derivation
* motion-based filters
* deblocking
* subsequent candidate generation

Implement MotionInfo refinement independently.

Do not reuse or modify MMVD MotionInfo code.

⸻

10. Encoder RDO

The two-stage MMVD workflow may be used as a reference.

The intended workflow is

For each projection-based candidate
Evaluate base candidate
Evaluate all refinement candidates
Generate prediction
Compute distortion
Estimate syntax bits
Compute first-pass cost
Prune
Perform full RD

Cost should include

Prediction distortion
+
Lambda × Total syntax bits

Estimate every syntax bit actually transmitted by the projection tool.

If necessary, create a dedicated bit-estimation function instead of modifying the MMVD estimator.

⸻

11. Computational Optimization

Avoid recomputing the complete camera/depth projection for every refinement candidate.

Prefer

Generate projected coordinate map once
For every refinement candidate
Apply delta
Recheck validity
Perform reference interpolation
Compute distortion

Analyze whether:

* projected coordinates can be reused
* projection and sampling can be separated
* interpolation alone can be repeated
* invalid masks must be recomputed
* prediction buffers can be reused
* chroma can reuse the same refinement
* SATD can initially evaluate luma only

If appropriate, propose splitting the current implementation into

Projection coordinate generation
Reference sampling

using new functions or wrappers without modifying existing behavior.

All new functions must be protected by #if GlobalMotion.

⸻

12. Invalid Region Handling

Analyze:

* current invalid detection
* boundary handling
* interpolation margins
* validity changes after refinement
* invalid fill policy
* distortion treatment
* invalid penalties
* clipping behavior
* encoder/decoder consistency

Determine whether validity should be recomputed after refinement.

Also investigate whether candidates producing excessive invalid regions should be early rejected or penalized.

⸻

13. Bi-pred

Analyze the existing MMVD bi-pred strategy, but do not reuse it directly.

Compare:

1. Uni-pred only
2. Same image-space delta for both references
3. Newly implemented temporal scaling
4. Camera-motion-based scaling
5. Independent refinement for each reference

Initially, prioritize the uni-pred solution for simplicity and robustness.

⸻

14. Recommended Development Stages

Stage 1

* Analyze projection tool
* Analyze MMVD
* Identify projection generation
* Identify MotionInfo
* Identify syntax

Stage 2

* Encoder-only experiment
* Apply fixed refinement
* Measure distortion improvement
* No syntax or decoder changes yet

Stage 3

* Independent refinement index
* Candidate generation
* SATD pruning
* Full RD

Stage 4

* Dedicated syntax
* Dedicated CABAC
* Encoder/decoder synchronization

Stage 5

* MotionInfo refinement
* Merge/HMVP verification

Stage 6

* Coordinate reuse
* Interpolation reuse
* Buffer reuse
* Invalid rejection optimization

Stage 7

* More refinement steps
* Diagonal directions
* Bi-pred support
* Syntax optimization

Every stage must verify both:

* GlobalMotion == 0
* GlobalMotion == 1

⸻

15. Expected Deliverables

A. Projection Tool Flow

Provide the actual file/function call flow.

B. MMVD Flow

Provide the actual MMVD implementation flow.

C. Reusable MMVD Concepts

Clearly separate:

* reusable algorithms
* reusable coding concepts
* reusable RDO ideas
* reusable precision handling
* reusable validation strategies

D. MMVD Components That Must Never Be Modified

List everything that should remain untouched.

E. Dedicated Projection Refinement Design

Describe:

* new data structures
* new state variables
* new refinement index
* new delta generation
* new candidate generation
* new syntax
* new CABAC contexts
* new encoder RDO
* new decoder reconstruction
* new MotionInfo update

Use the project’s naming conventions after analyzing the source.

F. File/Function-Level Modification Plan

Provide a table containing:

File
Existing function or structure
Current responsibility
New addition
MMVD impact
Location of #if GlobalMotion
Notes

The MMVD impact should ideally be None for every item.

G. New Call Flow

Provide pseudocode such as

Base projection
Generate refinement candidates
Decode refinement delta
Apply delta
Validity check
Reference sampling
Cost evaluation
Store final motion

H. Validation Plan

Include at least:

* GlobalMotion=0 build
* GlobalMotion=1 build
* Existing MMVD bit-exact regression
* Verify unchanged MMVD behavior
* Encoder/decoder mismatch test
* Prediction shift verification
* Selected block ratio
* Direction distribution
* Step distribution
* Invalid-region statistics
* Distortion improvement over the original projection path
* BD-rate improvement
* Encoding time
* Decoding time
* MotionInfo verification
* HMVP/merge verification

⸻

16. Final Notes

* Never assume projection-related identifiers before analyzing the source.
* Do not modify existing MMVD code.
* Do not insert projection-specific logic into MMVD functions.
* Do not reuse MMVD contexts.
* Do not extend MMVD syntax.
* Do not attach the new functionality to MMVD candidate types.
* Use MMVD only as a conceptual design reference.
* Implement the projection refinement as a completely independent feature.
* Ensure that prediction and MotionInfo are updated consistently.
* Protect every new or modified code section with #if GlobalMotion.
* When GlobalMotion == 0, the codec must compile and behave exactly as before.
* Do not generate patches yet.
* Produce only the analysis, implementation plan, and high-level pseudocode.
* If any information is uncertain, explicitly identify it as something that must be verified in the source code.
