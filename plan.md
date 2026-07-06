# AMVP_Cam_Bias Implementation Plan Request

I want to add an `AMVP_Cam_Bias` feature as a sub-option of the existing `GlobalMotion` feature in the current codebase.

The goal is **not** to add the cam-projection MV to the AMVP candidate list. Instead, when a separate flag is true, the cam-projection MV should be used as the **motion search center** and the **MVD base** in AMVP inter mode.

The existing AMVP behavior is:

```text
MVP = existing AMVP candidate[mvpIdx]
motion search center = MVP
final MV = MVP + MVD
```

The new behavior when the flag is true should be:

```text
MVP = cam-projection MV
motion search center = cam-projection MV
final MV = cam-projection MV + MVD
```

In other words, the search method itself, such as diamond search or TZ search, should remain unchanged. Only the search center should be offset to the cam-projection MV. On the decoder side, when the same flag is true, the final MV should be reconstructed by adding the decoded MVD to the cam-projection MV.

---

## Macro Rules

Add the following macro to `TypeDef.h` or wherever feature macros are currently defined:

```cpp
#define AMVP_Cam_Bias 1
```

`AMVP_Cam_Bias` must be treated as a sub-feature of `GlobalMotion`.

If the new code is already inside a `#if GlobalMotion` block, use only:

```cpp
#if AMVP_Cam_Bias
```

If the new code is outside a `#if GlobalMotion` block, use:

```cpp
#if GlobalMotion && AMVP_Cam_Bias
```

---

## Required Constraints

For this step, implement **AMVP only**.

Do **not** modify:

```text
Affine
Affine AMVP
Merge
HMVP
MMVD
GPM
AMVP candidate list
AMVP candidate count
```

Especially important:

```text
Do not add the cam-projection MV to the AMVP candidate list.
Do not replace any existing AMVP candidate.
Do not break the existing mvpIdx structure.
```

When `amvp_cam_bias_flag` is false, the behavior must be exactly the same as the existing AMVP behavior.

When `amvp_cam_bias_flag` is true, the existing AMVP candidate and `mvpIdx` should not be used for MV reconstruction. Instead, the cam-projection MV should be used as the MVP.

---

## Intended Encoder Behavior

The encoder should be able to compare the regular AMVP path and the cam-bias AMVP path.

Regular path:

```text
MVP = existing AMVP candidate[mvpIdx]
search center = MVP
MVD = final MV - MVP
```

Cam-bias path:

```text
MVP = cam-projection MV
search center = cam-projection MV
MVD = final MV - cam-projection MV
```

The encoder should select the path with the better RD cost.

---

## Intended Decoder Behavior

The decoder should reconstruct the final MV according to the flag:

```text
if amvp_cam_bias_flag == false:
    MVP = existing AMVP candidate[mvpIdx]
    final MV = MVP + MVD

if amvp_cam_bias_flag == true:
    MVP = cam-projection MV
    final MV = cam-projection MV + MVD
```

For bi-prediction, L0 and L1 should be handled independently.

---

## Cam-Projection MV Requirements

The cam-projection MV must be derived identically on the encoder and decoder sides.

Therefore, the implementation must use:

```text
decoded depth
decoded camera parameters
the same refIdx / POC mapping
the same block-position convention
the same MV precision / rounding / clipping
```

Do **not** use encoder-only float depth or original NPZ data.

If the existing `GlobalMotion` code already has a function for deriving a block-level cam-projection MV, reuse it.

If such a function does not exist, design a shared helper in the common code so that both encoder and decoder can call the same derivation logic.

---

## Requested Output

Do not implement immediately. First, inspect the current codebase and write an implementation plan.

The plan must include:

```text
1. Files to modify
2. Main functions to modify
3. New syntax/state fields to add
4. Where to change the encoder-side search center and MVD base
5. Where to reconstruct the final MV on the decoder side
6. Whether to reuse an existing cam-projection MV derivation function or create a new shared helper
7. Possible encoder/decoder mismatch risks
8. Minimal test procedure
```

After presenting the plan, wait for approval before implementing.
