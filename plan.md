Do not modify the code yet. First, write a concrete implementation plan for adding a picture-level enable flag for the Camera Projection / GlobalMotion tool.

Important macro rule:

The current Camera Projection tool is implemented under the #if GlobalMotion macro.

All new and modified code for this feature must also be guarded by the existing #if GlobalMotion / #endif macro.

Do not remove, rename, bypass, or weaken the GlobalMotion macro.

⸻

Goal

The Camera Projection tool currently can be tested/signaled at CU level for each picture.

However, when camera-induced motion is too small, the Camera Projection tool is likely not useful. Therefore, add a picture-level flag that enables or disables the Camera Projection tool for the current picture.

For this first implementation, do not change camera parameter signaling.

That means:

* Camera parameters are still transmitted for all pictures as before.
* The new flag controls only whether the Camera Projection tool is allowed for the current picture.
* If the picture-level enable flag is false, Camera Projection is completely disabled for that picture.
* In a disabled picture, CU-level Camera Projection CABAC syntax must not be written or read.
* In a disabled picture, Camera Projection RDO must not run.
* In a disabled picture, Camera Projection prediction must not run.

⸻

New picture-level flag

Use positive polarity.

Bitstream syntax name:

cam_proj_pic_enabled_flag

Semantic meaning:

camProjPicEnabledFlag == true

means:

Camera Projection tool is allowed for the current picture.

And:

camProjPicEnabledFlag == false

means:

Camera Projection tool is completely disabled for the current picture.

Required behavior:

camProjPicEnabledFlag == false:
  no CU-level camProjFlag coding
  no camProjMergeIdx coding
  no camProj RDO
  no camProj prediction
camProjPicEnabledFlag == true:
  existing Camera Projection tool behavior is available

⸻

Encoder-side gating calculation position

In the encoder, the current frame camera parameter and depth map are already loaded before CTU-level encoding starts.

At that point, before the CTU loop begins:

1. Current picture camera parameter and depth map are available.
2. Measure Camera Projection MV magnitude between the current picture and selected reference pictures.
3. If the measured value is below a hard-coded threshold, set:

camProjPicEnabledFlag = false;

4. Otherwise, set:

camProjPicEnabledFlag = true;

5. During CTU/CU RDO, this flag must be used to completely exclude Camera Projection when disabled.

⸻

Reference pictures used for MV measurement

Treat the current picture as the target picture.

Measure only against reference pictures that are actually available for inter prediction.

For this first implementation, do not check every active reference. Use only:

the closest reference picture in L0
the closest reference picture in L1

Selection criterion:

closest reference = reference picture with minimum abs(refPOC - currPOC)

If L0 or L1 is unavailable, use only the available list.

If the closest L0 reference and closest L1 reference have the same POC, deduplicate them.

The projection direction must match the existing Camera Projection prediction path:

current target picture -> selected reference picture

Examples:

For POC 16 with closest refs 0 and 32:
  measure 16 -> 0
  measure 16 -> 32
For POC 8 with closest refs 0 and 16:
  measure 8 -> 0
  measure 8 -> 16

⸻

5-point average MV metric

Use the following 5 target-picture sample points:

LT     = (0, 0)
RT     = (W - 1, 0)
LB     = (0, H - 1)
RB     = (W - 1, H - 1)
Center = ((W - 1) / 2, (H - 1) / 2)

For each selected reference picture, project these 5 target points from the current picture into the reference picture.

For each valid projected point:

mv_x = projected_ref_x - target_x
mv_y = projected_ref_y - target_y
mv_len = sqrt(mv_x * mv_x + mv_y * mv_y)

For each target-reference pair, compute:

avg_mv_px = average(mv_len over valid sample points)
norm_avg_mv_4096 = avg_mv_px / ((W + H) / 2) * 4096

For the selected closest L0/L1 references, compute norm_avg_mv_4096 separately, then use the minimum value conservatively:

minNormAvgMv4096 = min(normAvgMv4096 over selected closest references);

⸻

Hard-coded threshold

Do not add a cfg parameter in this task.

Do not add command-line options or encoder cfg parsing.

Hard-code the threshold for now.

Suggested constants:

#if GlobalMotion
static constexpr double CAM_PROJ_PIC_ENABLE_NORM_MV_TH = 3.0;
static constexpr int    CAM_PROJ_PIC_ENABLE_MIN_VALID_POINTS = 3;
#endif

These values must be defined under #if GlobalMotion, either as local constants or static constexpr constants near the new gating function.

Decision rule:

if (minNormAvgMv4096 >= CAM_PROJ_PIC_ENABLE_NORM_MV_TH)
{
  camProjPicEnabledFlag = true;
}
else
{
  camProjPicEnabledFlag = false;
}

Meaning:

Only enable Camera Projection for the current picture when even the closest selected reference has sufficiently large camera-projected motion.

⸻

Invalid sample handling

Some of the 5 projected points can be invalid.

Invalid conditions include:

invalid depth
depth <= 0
projection result is NaN or Inf
point is behind the reference camera

Do not mark a point invalid only because the projected coordinate is outside the reference picture boundary.

For this metric, outside-picture projection still indicates camera motion and the displacement can still be measured, as long as the projection is finite and in front of the reference camera.

For each selected reference, if the number of valid sample points is smaller than:

CAM_PROJ_PIC_ENABLE_MIN_VALID_POINTS

then ignore that reference for this metric.

If no selected reference has enough valid sample points, conservatively set:

camProjPicEnabledFlag = false;

⸻

Picture header bitstream syntax

Add cam_proj_pic_enabled_flag to the picture header bitstream.

Prefer Picture Header.

If the current codebase makes it difficult to add this custom tool flag to the Picture Header, find the closest high-level syntax location that is parsed before CU/CABAC syntax.

If Slice Header must be used instead, ensure the value is identical for all slices of the same picture.

The plan must include:

1. Where to store the flag:
    * PictureHeader class or equivalent structure
    * getter/setter locations
2. Encoder HLS writing location
3. Decoder HLS reading location
4. Trace/debug output location
5. Confirmation that the decoder can access this flag before CU/CABAC syntax parsing begins

⸻

CABAC / CU syntax requirement

If:

camProjPicEnabledFlag == false

then all CU-level Camera Projection syntax must be completely excluded.

The following syntax must not be written or read:

camProjFlag
camProjMergeFlag
camProjMergeSkipFlag
camProjMergeIdx

For disabled pictures, the reader must force:

cu.camProjFlag = false;
cu.camProjMergeFlag = false;
cu.camProjMergeSkipFlag = false;

For disabled pictures, the writer must CHECK that stale flags are not present:

CHECK(cu.camProjFlag, "camProjFlag must be false when picture-level camProj is disabled");
CHECK(cu.camProjMergeFlag, "camProjMergeFlag must be false when picture-level camProj is disabled");
CHECK(cu.camProjMergeSkipFlag, "camProjMergeSkipFlag must be false when picture-level camProj is disabled");

Use exact CHECK placement and wording consistent with the codebase style.

⸻

Encoder RDO requirement

If:

camProjPicEnabledFlag == false

then the encoder must not test any Camera Projection modes for that picture.

Disable all of the following:

regular Camera Projection RDO
Camera Projection merge RDO
Camera Projection skip-merge RDO
Camera Projection candidate generation
Camera Projection merge index search
Camera Projection bit-cost estimation
Camera Projection motion-info generation

The implementation plan must include how to prevent stale Camera Projection flags from surviving in temporary CU, best CU, split CU, or copied CU state.

⸻

Prediction requirement

If:

camProjPicEnabledFlag == false

then Camera Projection prediction must never execute.

Add a defensive CHECK in the prediction dispatch path.

Example:

#if GlobalMotion
if (cu.camProjFlag)
{
  CHECK(!picHeader->getCamProjPicEnabledFlag(),
        "Camera Projection prediction called while picture-level Camera Projection is disabled");
}
#endif

The exact location and style should match the codebase.

⸻

Things that must not be changed in this task

Do not do any of the following:

Do not optimize camera parameter signaling.
Do not add camParamPresentBitmap.
Do not modify RA reference dependency handling.
Do not unnecessarily change existing merge/MMVD/GPM/CIIP/affine syntax.
Do not add a cfg parameter.
Do not add a command-line option.
Do not modify encoder cfg parsing.
Do not remove or bypass the GlobalMotion macro.

⸻

Files / functions to inspect

Before writing code, inspect these areas and include them in the plan.

1. Picture header / high-level syntax
    * PictureHeader class or equivalent
    * HLS writer
    * HLS reader
    * syntax trace output
2. Encoder pre-CTU setup
    * current picture camera parameter load location
    * current picture depth map load location
    * location immediately before CTU encoding starts
    * per-picture tool availability initialization location
3. RPL / reference picture access
    * current slice RefPicList0
    * current slice RefPicList1
    * how to get ref POC
    * how to find closest ref by abs(refPOC - currPOC)
4. Camera Projection projection function
    * existing target-to-reference projection function
    * existing depth/camera convention
    * existing MV computation function, if any
    * reuse existing projection code as much as possible
5. CABAC syntax
    * EncoderLib/CABACWriter.cpp
    * DecoderLib/CABACReader.cpp
    * EncoderLib/CABACWriter.h
    * DecoderLib/CABACReader.h
    * current camProjFlag
    * current camProjMergeFlag
    * current camProjMergeSkipFlag
    * current camProjMergeIdx
6. Encoder RDO path
    * regular Camera Projection mode
    * Camera Projection merge mode
    * Camera Projection skip-merge mode
    * Camera Projection candidate generation
    * Camera Projection bit-cost estimation
7. Prediction path
    * where cu.camProjFlag dispatches Camera Projection prediction
    * merge/skip/inter prediction branching location
8. CU state management
    * CodingUnit flag definition
    * CU reset/init
    * CU copy
    * tempCU / bestCU copy
    * stale Camera Projection flag prevention

⸻

Implementation plan must include

Before coding, provide a plan with these sections:

1. Summary of current Camera Projection syntax / RDO / prediction flow
2. Where to store the new camProjPicEnabledFlag
3. Picture Header write/read locations
4. Encoder location for computing the 5-point normalized MV metric
5. Method for selecting closest L0/L1 references
6. Method for reusing existing projection code
7. Hard-coded threshold application
8. Invalid point handling
9. CABACWriter guard plan
10. CABACReader guard plan
11. RDO disable plan
12. Prediction disable plan
13. CU stale-flag prevention plan
14. Trace/debug log plan
15. #if GlobalMotion guard plan
16. Potential mismatch risks
17. Validation checklist

⸻

Validation checklist

The plan must end with the following validation checklist.

Bitstream syntax

* Encoder writes cam_proj_pic_enabled_flag.
* Decoder reads cam_proj_pic_enabled_flag from the same location.
* Writer/reader syntax order matches exactly.

Disabled picture

When:

camProjPicEnabledFlag == false

verify:

* no CU-level camProjFlag bin is written
* no CU-level camProjFlag bin is read
* no camProjMergeIdx is written/read
* no Camera Projection RDO is executed
* no Camera Projection prediction is executed
* all CU Camera Projection flags remain false
* normal inter/merge/intra coding still works

Enabled picture

When:

camProjPicEnabledFlag == true

verify:

* existing Camera Projection behavior remains available
* existing CU-level Camera Projection syntax works
* existing Camera Projection RDO works
* existing Camera Projection prediction works

Metric / threshold

Print per-picture logs containing:

POC
selected closest L0 ref POC
selected closest L1 ref POC
normAvgMv4096 for each selected ref
minNormAvgMv4096
threshold = 3.0
valid sample point count
final camProjPicEnabledFlag

Mismatch safety

Verify:

* no encoder/decoder reconstruction mismatch
* no CABAC read/write mismatch
* no stale camProjFlag, camProjMergeFlag, or camProjMergeSkipFlag in disabled pictures
* no CHECK is triggered during normal coding
* all new/modified Camera Projection code is under #if GlobalMotion
