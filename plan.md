Please do not modify the code yet. First, make a concrete implementation plan for restructuring the Camera Projection tool syntax and internal CU flag semantics.

Important macro rule:
All Camera Projection / Global Motion related code must remain guarded by the existing #if GlobalMotion / #endif macro.
Do not remove, bypass, rename, or weaken the GlobalMotion macro.
Any new code added for this tool must also be inside the appropriate #if GlobalMotion guard.

Current implementation has these Camera Projection related flags:

* camProjFlag
* camProjMergeFlag
* camProjMergeSkipFlag
* camProjMergeIdx

New design goal:

Use existing syntax-level flags for mode signaling:

* cu.skip
* cu.mergeFlag
* camProjFlag

Keep camProjMergeFlag and camProjMergeSkipFlag as internal derived state flags only.
They should no longer be independently signaled in CABAC.

Target mode definitions:

1. Regular Camera Projection mode
    * camProjFlag = true
    * cu.skip = false
    * cu.mergeFlag = false
    * internally:
        * camProjMergeFlag = false
        * camProjMergeSkipFlag = false
2. Camera Projection merge mode
    * camProjFlag = true
    * cu.skip = false
    * cu.mergeFlag = true
    * internally:
        * camProjMergeFlag = true
        * camProjMergeSkipFlag = false
3. Camera Projection skip-merge mode
    * camProjFlag = true
    * cu.skip = true
    * cu.mergeFlag = true
    * internally:
        * camProjMergeFlag = true
        * camProjMergeSkipFlag = true

Key requirement:
camProjMergeFlag and camProjMergeSkipFlag should be derived from already parsed syntax state, not coded as separate syntax elements.

Decoder-side derivation rule:

After parsing camProjFlag, cu.skip, and cu.mergeFlag, derive internal flags like this:

if (cu.camProjFlag)
{
  if (cu.skip)
  {
    cu.mergeFlag = true;
    cu.camProjMergeFlag = true;
    cu.camProjMergeSkipFlag = true;
  }
  else if (cu.mergeFlag)
  {
    cu.camProjMergeFlag = true;
    cu.camProjMergeSkipFlag = false;
  }
  else
  {
    cu.camProjMergeFlag = false;
    cu.camProjMergeSkipFlag = false;
  }
}
else
{
  cu.camProjMergeFlag = false;
  cu.camProjMergeSkipFlag = false;
}

Encoder-side rule:
The encoder may still use camProjMergeFlag and camProjMergeSkipFlag internally for mode decision, prediction dispatch, debug, and RDO bookkeeping, but they must be consistent with the signaled state.

Encoder-side expected states:

Regular Camera Projection:

cu.camProjFlag = true;
cu.skip = false;
cu.mergeFlag = false;
cu.camProjMergeFlag = false;
cu.camProjMergeSkipFlag = false;

Camera Projection merge:

cu.camProjFlag = true;
cu.skip = false;
cu.mergeFlag = true;
cu.camProjMergeFlag = true;
cu.camProjMergeSkipFlag = false;

Camera Projection skip-merge:

cu.camProjFlag = true;
cu.skip = true;
cu.mergeFlag = true;
cu.camProjMergeFlag = true;
cu.camProjMergeSkipFlag = true;

Syntax design:

1. Do not signal camProjMergeFlag.
2. Do not signal camProjMergeSkipFlag.
3. Signal only camProjFlag in the relevant path.
4. camProjMergeIdx is still coded only for Camera Projection merge and Camera Projection skip-merge modes.
5. camProjMergeIdx must not be coded for regular Camera Projection mode.

Placement requirement:

Camera Projection should be checked and handled early in each relevant syntax/prediction path to avoid being misinterpreted as an existing merge/MMVD/GPM/affine/CIIP/AMVP tool.

In other words:
If cu.camProjFlag == true, the Camera Projection path should dispatch first and return/skip unrelated syntax or prediction paths as needed.

This is to prevent invalid combinations where existing tools only see cu.skip and cu.mergeFlag, then incorrectly try to execute normal merge/MMVD/GPM/affine/CIIP/AMVP behavior on a Camera Projection CU.

Required behavior by path:

A. Skip path:

* cu.skip == true
* If camProjFlag == true, derive:
    * cu.mergeFlag = true
    * cu.camProjMergeFlag = true
    * cu.camProjMergeSkipFlag = true
* Parse/code camProjMergeIdx if needed.
* Do not continue into unrelated normal skip/merge tool parsing after Camera Projection skip-merge is selected.

B. Merge path:

* cu.skip == false
* cu.mergeFlag == true
* If camProjFlag == true, derive:
    * cu.camProjMergeFlag = true
    * cu.camProjMergeSkipFlag = false
* Parse/code camProjMergeIdx.
* Do not continue into unrelated normal merge/MMVD/GPM/affine/CIIP syntax after Camera Projection merge is selected.

C. Regular inter / AMVP-like path:

* cu.skip == false
* cu.mergeFlag == false
* If camProjFlag == true, derive:
    * cu.camProjMergeFlag = false
    * cu.camProjMergeSkipFlag = false
* Do not parse normal ref_idx, mvd, mvp_idx, imv, affine_amvr, bcw, lic, obmc, or other AMVP-like syntax unless the Camera Projection regular mode explicitly requires it.

Files/functions to inspect and plan modifications for:

1. CABAC syntax:

* EncoderLib/CABACWriter.cpp
* DecoderLib/CABACReader.cpp
* EncoderLib/CABACWriter.h
* DecoderLib/CABACReader.h

Focus on:

* coding_unit()
* cu_pred_data()
* prediction_unit()
* merge_data()
* current Camera Projection flag coding functions
* current camProjMergeFlag / camProjMergeSkipFlag coding sites
* camProjMergeIdx coding site

2. CU state / flags:

* CodingUnit definition
* Any reset/init/copy functions for CU flags
* Any mode decision structures that store Camera Projection state

3. Encoder decision path:

* Camera Projection regular mode selection
* Camera Projection merge mode selection
* Camera Projection skip-merge mode selection
* RDO bit estimation for Camera Projection syntax
* Any temporary CU / best CU copying logic

4. Decoder prediction/reconstruction path:

* where Camera Projection prediction is dispatched
* where normal merge/MMVD/GPM/affine/CIIP/AMVP prediction is dispatched
* ensure Camera Projection is checked first when camProjFlag == true

5. Context definitions:

* Remove or stop using contexts for independently coded camProjMergeFlag and camProjMergeSkipFlag if they are no longer signaled.
* Keep only necessary context(s) for camProjFlag.
* Keep any context needed for camProjMergeIdx if it is CABAC-coded.

6. Debug/tracing:

* Update trace output so it clearly reports:
    * camProjFlag
    * derived camProjMergeFlag
    * derived camProjMergeSkipFlag
    * cu.skip
    * cu.mergeFlag
    * camProjMergeIdx

Planning requirements:

Please produce a staged implementation plan, not code yet.

The plan must include:

1. Current syntax flow summary.
2. Target syntax flow for:
    * skip path
    * merge path
    * regular inter / AMVP-like path
3. Which flags are still signaled and which are internal derived flags.
4. Exact files/functions to modify.
5. CABACWriter changes.
6. CABACReader changes.
7. Encoder-side CU flag setting changes.
8. Decoder-side internal flag derivation and prediction dispatch changes.
9. RDO bit-estimation changes.
10. Context cleanup changes.
11. GlobalMotion macro preservation points.
12. Potential decoder mismatch risks.
13. Validation checklist.

Validation checklist must include these expected final CU states:

Regular Camera Projection:

* camProjFlag == true
* skip == false
* mergeFlag == false
* camProjMergeFlag == false
* camProjMergeSkipFlag == false
* no camProjMergeIdx coded

Camera Projection merge:

* camProjFlag == true
* skip == false
* mergeFlag == true
* camProjMergeFlag == true
* camProjMergeSkipFlag == false
* camProjMergeIdx coded

Camera Projection skip-merge:

* camProjFlag == true
* skip == true
* mergeFlag == true
* camProjMergeFlag == true
* camProjMergeSkipFlag == true
* camProjMergeIdx coded

Also verify:

* Writer and Reader syntax order match exactly.
* No independent CABAC signaling remains for camProjMergeFlag.
* No independent CABAC signaling remains for camProjMergeSkipFlag.
* Existing merge/MMVD/GPM/affine/CIIP/AMVP tools do not run on Camera Projection CUs.
* No invalid CHECK is triggered by Camera Projection CUs having skip or mergeFlag set.
* No decoder mismatch occurs.
* All new or modified Camera Projection code is protected by #if GlobalMotion.
