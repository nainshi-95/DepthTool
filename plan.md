Please do not modify the code yet. First, make a concrete implementation plan for restructuring the Camera Projection CABAC syntax and CU flag semantics.

Current implementation has these Camera Projection syntax/flags:

* camProjFlag
* camProjMergeFlag
* camProjMergeSkipFlag
* camProjMergeIdx

I want to simplify and restructure the design.

Target design:

There should be three Camera Projection modes:

1. Regular Camera Projection mode
    * camProjFlag = true
    * cu.skip = false
    * cu.mergeFlag = false
2. Camera Projection merge mode
    * camProjFlag = true
    * cu.skip = false
    * cu.mergeFlag = true
3. Camera Projection skip-merge mode
    * camProjFlag = true
    * cu.skip = true
    * cu.mergeFlag = true

Required changes:

1. Remove camProjMergeFlag as an independent mode flag.
    * Camera Projection merge mode should be represented by the existing cu.mergeFlag.
    * Do not signal a separate camProjMergeFlag.
2. Remove or stop using camProjMergeSkipFlag as an independent mode flag.
    * Camera Projection skip-merge mode should be represented by the existing cu.skip flag.
    * Since skip mode is already merge-like, do not signal a separate Camera Projection skip-mode flag unless there is absolutely no alternative.
3. Keep camProjFlag as the Camera Projection tool selection flag.
    * Its meaning depends on the syntax path:
        * in the skip path: Camera Projection skip-merge
        * in the merge path: Camera Projection merge
        * in the regular inter / AMVP-like path: regular Camera Projection
4. Keep camProjMergeIdx only for Camera Projection merge and Camera Projection skip-merge modes.
    * It should be coded only when camProjFlag == true and the current path is skip or merge.
    * It should not be coded for regular Camera Projection mode.
5. Place the Camera Projection flag as late as possible in each relevant syntax path:
    * In the existing skip-merge path, after existing skip/merge-related tools, place Camera Projection skip-merge as the final fallback candidate.
    * In the existing merge path, after existing merge-related tools, place Camera Projection merge as the final fallback candidate.
    * In the regular inter / AMVP-like path, after existing AMVP-like or regular inter tools, place regular Camera Projection as the final fallback candidate.
    * Do not put Camera Projection merge or skip-merge under the cu.mergeFlag == false path.
6. Decoder-side parsing must mirror encoder-side writing exactly.
    * CABACWriter.cpp and CABACReader.cpp must have identical syntax order.
    * The reconstructed CU state must satisfy the target mode definitions above.
7. Encoder-side decision code must set flags consistently:
    * Regular Camera Projection:
        * camProjFlag = true
        * skip = false
        * mergeFlag = false
    * Camera Projection merge:
        * camProjFlag = true
        * skip = false
        * mergeFlag = true
    * Camera Projection skip-merge:
        * camProjFlag = true
        * skip = true
        * mergeFlag = true
8. Decoder-side tool execution must use the same interpretation:
    * camProjFlag && cu.skip means Camera Projection skip-merge.
    * camProjFlag && !cu.skip && cu.mergeFlag means Camera Projection merge.
    * camProjFlag && !cu.skip && !cu.mergeFlag means regular Camera Projection.
9. Existing syntax must not be incorrectly parsed when camProjFlag is true.
    * For Camera Projection merge / skip-merge, parse camProjMergeIdx and do not parse unrelated normal merge syntax afterward.
    * For regular Camera Projection, do not parse normal ref_idx, mvd, mvp_idx, or other AMVP syntax unless the tool explicitly requires it.
    * Ensure imv, affine_amvr, bcw, lic, obmc, and similar syntax are skipped or handled consistently if Camera Projection bypasses normal AMVP coding.
10. Please identify all affected files/functions before coding, likely including:

* EncoderLib/CABACWriter.cpp
* DecoderLib/CABACReader.cpp
* EncoderLib/CABACWriter.h
* DecoderLib/CABACReader.h
* CU flag definitions / CodingUnit fields
* encoder CU mode decision code
* decoder reconstruction / inter prediction path
* any context definitions related to removed flags
* tracing/debug output
* any RDO bit-estimation code that estimates Camera Projection syntax cost

Expected output:

Please produce a staged implementation plan, not code yet.

The plan should include:

1. Current syntax flow summary.
2. Proposed target syntax flow for:
    * skip path
    * merge path
    * regular inter / AMVP-like path
3. List of flags to remove, keep, or reinterpret.
4. Exact files/functions to modify.
5. CABACWriter changes.
6. CABACReader changes.
7. Encoder-side CU flag-setting changes.
8. Decoder-side tool-dispatch changes.
9. RDO bit-estimation changes.
10. Risks and possible decoder mismatch points.
11. A validation checklist with expected CU flag states for all three modes.
