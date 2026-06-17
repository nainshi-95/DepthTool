plan.md

Coding Plan: DepthCam parameter Slice Header bitstream integration

Goal

Implement DepthCam camera parameter loading, slice-header writing/parsing, and encoder/decoder parameter store synchronization.

DepthCam parameters are picture-level semantic data, but the bitstream syntax must be placed in the slice header, not the Picture Header, because the decoder can know the full reconstructed POC only inside parseSliceHeader().

Core rules:

Use display POC as the only parameter key.
Do not use gopId as a parameter index.
Do not store long-term params in PicHeader.
All long-term parameter state must be managed by DepthCamParam.
EncGOP must actually load DepthCam parameters before slice header writing.

Before coding, inspect the current codebase and produce an implementation-specific plan.

Do not assume ownership/access paths.

Before modifying code, report:

- Exact files to modify
- Exact functions to modify
- Exact insertion points
- Existing DepthCamParam API and missing API
- How EncGOP can access DepthCamParam
- How HLSWriter can access encoder-side DepthCamParam
- How HLSyntaxReader can access decoder-side DepthCamParam
- Where full POC is available
- Whether Slice needs additional fields
- Whether PicHeader changes are unnecessary
- Build risks
- Any mismatch between this plan and current code

Do not start coding until the implementation-specific plan is approved.

⸻

1. Final syntax location

Do not add DepthCam syntax to:

codePictureHeader()
parsePictureHeader()

Reason:

parsePictureHeader() only reads ph_pic_order_cnt_lsb.
Full reconstructed POC is available only after POC reconstruction in parseSliceHeader().
DepthCam syntax existence is inferred from full POC.
Therefore DepthCam syntax must be written/read in slice header.

Writer insertion point

In HLSWriter::codeSliceHeader(...), add DepthCam syntax after normal slice header syntax where pcSlice->m_poc is valid.

Recommended position:

codeClippingValues(pcSlice);
codeDepthCamSliceHeaderPayload(pcSlice);

Reader insertion point

In HLSyntaxReader::parseSliceHeader(...), add DepthCam parsing after full POC reconstruction and after the corresponding normal slice header syntax.

Recommended position:

parseClippingValues(pcSlice);
parseDepthCamSliceHeaderPayload(pcSlice);
std::vector<uint32_t> entryPointOffset;
...
m_pcBitstream->readByteAlignment();

Do not place DepthCam parsing after readByteAlignment().

⸻

2. DepthCam syntax existence rule

DepthCam slice-header payload has no explicit present flag.

Payload existence is inferred from:

pocCurr == 0
pocCurr > lastWrittenParamPOC
pocCurr > lastReadParamPOC

Encoder rule

POC 0:
  - Always write intrinsic/header.
  - Never write extrinsic.
  - POC 0 extrinsic is default all-zero.
POC > 0:
  - If pocCurr <= lastWrittenParamPOC:
      write nothing.
  - If pocCurr > lastWrittenParamPOC:
      write one bundle for POC lastWrittenParamPOC + 1 through pocCurr.
      update lastWrittenParamPOC = pocCurr.

Decoder rule

POC 0:
  - Always read intrinsic/header.
  - Do not read extrinsic.
  - Store POC 0 extrinsic as default all-zero.
POC > 0:
  - If pocCurr <= lastReadParamPOC:
      read nothing.
  - If pocCurr > lastReadParamPOC:
      read one bundle for POC lastReadParamPOC + 1 through pocCurr.
      store each parsed param by display POC.
      update lastReadParamPOC = pocCurr.

Multiple slices in one picture

The same rule prevents duplicate writing/parsing.

Example:

First slice of POC 32:
  lastReadParamPOC = 0
  pocCurr = 32
  read bundle POC 1~32
  set lastReadParamPOC = 32
Second slice of POC 32:
  lastReadParamPOC = 32
  pocCurr = 32
  pocCurr <= lastReadParamPOC
  read nothing

No extra first-slice flag is required for DepthCam duplication prevention.

⸻

3. DepthCamParam manager state

Add all persistent DepthCam state to DepthCamParam.

Suggested fields:

int m_lastLoadedParamPOC  = 0;
int m_lastWrittenParamPOC = 0;
int m_lastReadParamPOC    = 0;
bool m_intrinsicHeaderLoaded = false;
bool m_intrinsicHeaderParsed = false;
DepthCamIntrinsicHeader m_intrinsicHeader;
std::map<int, DepthCamExtrinsicParam> m_extrinsicParamStore;

Meaning:

m_lastLoadedParamPOC:
  Encoder-side extrinsic params from POC 1 to this POC are loaded sequentially with no holes.
m_lastWrittenParamPOC:
  Extrinsic params from POC 1 to this POC have already been written into the bitstream.
m_lastReadParamPOC:
  Extrinsic params from POC 1 to this POC have already been parsed from the bitstream.
m_extrinsicParamStore:
  Long-term parameter storage by display POC.

Required functions:

bool ensureIntrinsicHeaderLoaded();
bool ensureExtrinsicLoadedUpToPOC(int pocCurr);
const DepthCamIntrinsicHeader& getIntrinsicHeader() const;
void setIntrinsicHeader(const DepthCamIntrinsicHeader& header);
bool hasExtrinsicParam(int poc) const;
const DepthCamExtrinsicParam& getExtrinsicParam(int poc) const;
void setExtrinsicParam(int poc, const DepthCamExtrinsicParam& param);
DepthCamExtrinsicParam getDefaultZeroExtrinsicParam() const;
int getLastLoadedParamPOC() const;
int getLastWrittenParamPOC() const;
int getLastReadParamPOC() const;
void setLastLoadedParamPOC(int poc);
void setLastWrittenParamPOC(int poc);
void setLastReadParamPOC(int poc);

⸻

4. Encoder-side parameter loading

This part is mandatory.

The encoder must actually load DepthCam params before slice header writing.

In EncGOP::compressGOP, after pocCurr is computed and before slice header writing, call the DepthCamParam manager.

Do not write bits directly from EncGOP.

Required behavior:

DepthCamParam& depthCamParam = m_pcEncLib->getDepthCamParam();
if (pocCurr == 0)
{
  CHECK(!depthCamParam.ensureIntrinsicHeaderLoaded(),
        "Failed to load DepthCam intrinsic/header");
  // POC 0 extrinsic is default all-zero and must not be loaded/written as a normal extrinsic payload.
}
else
{
  CHECK(!depthCamParam.ensureExtrinsicLoadedUpToPOC(pocCurr),
        "Failed to load DepthCam extrinsic params up to current POC");
}

The manager must load sequentially by display POC.

Required invariant:

m_lastLoadedParamPOC == N means:
POC 1..N extrinsic params have been loaded with no holes.

Required sequential loading logic:

bool DepthCamParam::ensureExtrinsicLoadedUpToPOC(int pocCurr)
{
  if (pocCurr <= m_lastLoadedParamPOC)
  {
    return true;
  }
  for (int poc = m_lastLoadedParamPOC + 1; poc <= pocCurr; ++poc)
  {
    DepthCamExtrinsicParam param;
    if (!loadOneExtrinsicParamForPOC(poc, param))
    {
      // Do not advance m_lastLoadedParamPOC on failure.
      return false;
    }
    setExtrinsicParam(poc, param);
    m_lastLoadedParamPOC = poc;
  }
  return true;
}

Important RA behavior:

For RA GOP32:
  - When encoder reaches POC 32, EncGOP must call ensureExtrinsicLoadedUpToPOC(32).
  - This must load POC 1,2,3,...,32 sequentially.
  - When encoder later reaches POC 16/8/4/2/1, loading must be skipped because m_lastLoadedParamPOC is already 32.
  - When encoder reaches POC 64, it must load POC 33~64 sequentially.

Never use gopId as a DepthCam parameter index.

Do not silently reuse stale parameters.

If parameter loading fails for POC N:

- return false
- do not store invalid parameter
- do not advance m_lastLoadedParamPOC to N
- fail clearly with CHECK at the EncGOP call site

⸻

5. Slice status fields

Add optional status fields to Slice.

These are not long-term storage. They only record what happened for the current slice.

int m_depthCamLastWrittenParamPOCBefore = 0;
int m_depthCamLastWrittenParamPOCAfter  = 0;
int m_depthCamLastReadParamPOCBefore = 0;
int m_depthCamLastReadParamPOCAfter  = 0;
int m_depthCamBundleStartPOC = 0;
int m_depthCamBundleEndPOC   = 0;
int m_depthCamBundleNumPics  = 0;

Persistent state must remain in DepthCamParam.

⸻

6. HLSWriter helper

Add:

void HLSWriter::codeDepthCamSliceHeaderPayload(Slice* pcSlice);
void HLSWriter::codeDepthCamIntrinsicHeader(const DepthCamIntrinsicHeader& header);
void HLSWriter::codeDepthCamExtrinsicParam(const DepthCamExtrinsicParam& param);

Writer logic:

void HLSWriter::codeDepthCamSliceHeaderPayload(Slice* pcSlice)
{
  const int pocCurr = pcSlice->m_poc;
  DepthCamParam& depthCamParam = getEncoderDepthCamParam(); // Use actual current-code access path.
  if (pocCurr == 0)
  {
    const auto& header = depthCamParam.getIntrinsicHeader();
    codeDepthCamIntrinsicHeader(header);
    pcSlice->m_depthCamLastWrittenParamPOCBefore = depthCamParam.getLastWrittenParamPOC();
    pcSlice->m_depthCamLastWrittenParamPOCAfter  = depthCamParam.getLastWrittenParamPOC();
    // Never write POC 0 extrinsic.
    return;
  }
  const int lastWritten = depthCamParam.getLastWrittenParamPOC();
  pcSlice->m_depthCamLastWrittenParamPOCBefore = lastWritten;
  if (pocCurr <= lastWritten)
  {
    pcSlice->m_depthCamLastWrittenParamPOCAfter = lastWritten;
    return;
  }
  const int startPoc = lastWritten + 1;
  const int endPoc   = pocCurr;
  const int numPics  = endPoc - startPoc + 1;
  pcSlice->m_depthCamBundleStartPOC = startPoc;
  pcSlice->m_depthCamBundleEndPOC   = endPoc;
  pcSlice->m_depthCamBundleNumPics  = numPics;
  xWriteUvlc(startPoc, "depth_cam_param_bundle_start_poc");
  xWriteUvlc(numPics,  "depth_cam_param_bundle_num_pics");
  for (int poc = startPoc; poc <= endPoc; ++poc)
  {
    CHECK(!depthCamParam.hasExtrinsicParam(poc),
          "Missing DepthCam extrinsic param before writing");
    const auto& param = depthCamParam.getExtrinsicParam(poc);
    codeDepthCamExtrinsicParam(param);
  }
  depthCamParam.setLastWrittenParamPOC(endPoc);
  pcSlice->m_depthCamLastWrittenParamPOCAfter = endPoc;
}

Access path rule:

Use the actual existing access path to DepthCamParam.
If HLSWriter currently cannot access EncLib/DepthCamParam, add the minimum necessary API.
Do not create global state.
Do not invent ownership relationships.

⸻

7. HLSReader helper

Add:

void HLSyntaxReader::parseDepthCamSliceHeaderPayload(Slice* pcSlice);
void HLSyntaxReader::parseDepthCamIntrinsicHeader(DepthCamIntrinsicHeader& header);
void HLSyntaxReader::parseDepthCamExtrinsicParam(DepthCamExtrinsicParam& param);

Reader logic:

void HLSyntaxReader::parseDepthCamSliceHeaderPayload(Slice* pcSlice)
{
  const int pocCurr = pcSlice->m_poc;
  DepthCamParam& depthCamParam = getDecoderDepthCamParam(); // Use actual current-code access path.
  if (pocCurr == 0)
  {
    DepthCamIntrinsicHeader header;
    parseDepthCamIntrinsicHeader(header);
    depthCamParam.setIntrinsicHeader(header);
    depthCamParam.setExtrinsicParam(0, depthCamParam.getDefaultZeroExtrinsicParam());
    pcSlice->m_depthCamLastReadParamPOCBefore = depthCamParam.getLastReadParamPOC();
    pcSlice->m_depthCamLastReadParamPOCAfter  = depthCamParam.getLastReadParamPOC();
    // Never read POC 0 extrinsic.
    return;
  }
  const int lastRead = depthCamParam.getLastReadParamPOC();
  pcSlice->m_depthCamLastReadParamPOCBefore = lastRead;
  if (pocCurr <= lastRead)
  {
    pcSlice->m_depthCamLastReadParamPOCAfter = lastRead;
    return;
  }
  uint32_t startPoc = 0;
  uint32_t numPics  = 0;
  xReadUvlc(startPoc, "depth_cam_param_bundle_start_poc");
  xReadUvlc(numPics,  "depth_cam_param_bundle_num_pics");
  const int expectedStartPoc = lastRead + 1;
  const int expectedNumPics  = pocCurr - expectedStartPoc + 1;
  const int endPoc           = int(startPoc) + int(numPics) - 1;
  CHECK(int(startPoc) != expectedStartPoc,
        "Invalid DepthCam bundle start POC");
  CHECK(int(numPics) != expectedNumPics,
        "Invalid DepthCam bundle num pics");
  CHECK(endPoc != pocCurr,
        "Invalid DepthCam bundle end POC");
  pcSlice->m_depthCamBundleStartPOC = int(startPoc);
  pcSlice->m_depthCamBundleEndPOC   = endPoc;
  pcSlice->m_depthCamBundleNumPics  = int(numPics);
  for (int i = 0; i < int(numPics); ++i)
  {
    const int poc = int(startPoc) + i;
    DepthCamExtrinsicParam param;
    parseDepthCamExtrinsicParam(param);
    depthCamParam.setExtrinsicParam(poc, param);
  }
  depthCamParam.setLastReadParamPOC(pocCurr);
  pcSlice->m_depthCamLastReadParamPOCAfter = pocCurr;
}

Important:

Immediately copy parsed params into DepthCamParam manager.
Do not use PicHeader as long-term parameter storage.
Do not parse anything when pocCurr <= lastReadParamPOC.

⸻

8. Truncated signed Exp-Golomb coding

Use cameraParamQuantTest.py as the reference.

Inspect cameraParamQuantTest.py and port the same integer quantization and truncated signed Exp-Golomb behavior to C++.

Required helper functions:

void xWriteTruncatedSignedExpGolomb(int value, int minVal, int maxVal, const char* name);
void xReadTruncatedSignedExpGolomb(int& value, int minVal, int maxVal, const char* name);

Requirements:

- Signed camera parameters must use truncated signed Exp-Golomb.
- Unsigned counts/indices use UVLC.
- Quantization scale, clipping, offset, rounding, min/max range must match cameraParamQuantTest.py.
- Encoder and decoder must reconstruct identical integer camera params.
- Do not silently clamp decoder values unless the Python reference does.
- Add CHECK for decoded out-of-range values.

Suggested structure:

void HLSWriter::codeDepthCamIntrinsicHeader(const DepthCamIntrinsicHeader& header)
{
  // Write fields in deterministic order.
}
void HLSWriter::codeDepthCamExtrinsicParam(const DepthCamExtrinsicParam& param)
{
  // Write quantized signed fields using truncated signed Exp-Golomb.
}
void HLSyntaxReader::parseDepthCamIntrinsicHeader(DepthCamIntrinsicHeader& header)
{
  // Read fields in the exact same order.
}
void HLSyntaxReader::parseDepthCamExtrinsicParam(DepthCamExtrinsicParam& param)
{
  // Read quantized signed fields using truncated signed Exp-Golomb.
}

⸻

9. Slice header syntax order

POC 0

When pocCurr == 0, write/read:

depth_cam_intrinsic_<field_0>
depth_cam_intrinsic_<field_1>
...
depth_cam_header_<field_0>
depth_cam_header_<field_1>
...

Do not write/read:

depth_cam_param_bundle_start_poc
depth_cam_param_bundle_num_pics
depth_cam_extrinsic_*

POC 0 extrinsic must be initialized to all-zero/default in the manager.

POC > 0 and pocCurr > lastParamPOC

Write/read:

depth_cam_param_bundle_start_poc     uvlc
depth_cam_param_bundle_num_pics      uvlc
for poc = startPoc to pocCurr:
    depth_cam_extrinsic_<field_0>    truncated signed exp-golomb
    depth_cam_extrinsic_<field_1>    truncated signed exp-golomb
    ...

Where:

startPoc = lastWrittenParamPOC + 1;   // encoder
startPoc = lastReadParamPOC + 1;      // decoder
numPics  = pocCurr - startPoc + 1;

POC > 0 and pocCurr <= lastParamPOC

Write/read nothing.

⸻

10. Decoder parameter use

When decoder needs DepthCam params for any picture, use only DepthCamParam.

DepthCamParam& depthCamParam = getDecoderDepthCamParam();
if (pocCurr == 0)
{
  const auto param = depthCamParam.getDefaultZeroExtrinsicParam();
}
else
{
  CHECK(!depthCamParam.hasExtrinsicParam(pocCurr),
        "Missing DepthCam extrinsic param for current POC");
  const auto& param = depthCamParam.getExtrinsicParam(pocCurr);
}

Do not access DepthCam params through PicHeader.

⸻

11. Do not change unrelated behavior

Do not:

- Do not modify prediction logic.
- Do not modify RDO behavior.
- Do not modify RPL behavior.
- Do not use gopId as a parameter index.
- Do not write bits from EncGOP directly.
- Do not write POC 0 extrinsic.
- Do not store long-term params in PicHeader.
- Do not create global DepthCamParam state.
- Do not invent ownership/access paths.
- Do not refactor unrelated VTM code.

⸻

12. Implementation order

Recommended order:

1. Read current codebase and produce implementation-specific plan.
2. Add or update DepthCamParam manager state/functions.
3. Implement actual encoder-side parameter loading in EncGOP.
4. Add Slice status fields if needed.
5. Add HLSWriter access path to encoder-side DepthCamParam.
6. Add HLSWriter slice-header payload helper.
7. Add HLSReader access path to decoder-side DepthCamParam.
8. Add HLSReader slice-header payload helper.
9. Port truncated signed Exp-Golomb from cameraParamQuantTest.py.
10. Add intrinsic/extrinsic syntax helpers.
11. Add decoder-side parameter use through DepthCamParam manager.
12. Build.
13. Run tests according to test_skill.md.
14. Report final result.

⸻

13. Required final report

Report:

Modified files:
- ...
Modified functions:
- ...
New fields:
- ...
New helper functions:
- ...
DepthCamParam manager changes:
- ...
Encoder parameter loading path:
- ...
HLSWriter DepthCamParam access path:
- ...
HLSReader DepthCamParam access path:
- ...
New slice-header syntax order:
- ...
Build result:
- ...
Test result:
- ...
Final result:
- PASS or FAIL








































macro_guard_skill.md

Skill: Mandatory macro guard for DepthCam parameter integration

Goal

Ensure every DepthCam parameter bitstream/load/store change is fully compile-time guarded.

The macro guard must make macro-OFF behavior identical to original VTM.

Macro names

Use the project-selected compile-time macro:

<PUT_MACRO_NAME_HERE>

Optional debug/log macro:

DEBUG_DEPTH_CAM_PARAM

Required rule

Every DepthCam-related modification to existing VTM behavior must be guarded.

For existing function modifications:

#if <PUT_MACRO_NAME_HERE>
  // New DepthCam behavior
#else
  // Original existing VTM code exactly as before
#endif

For new fields, classes, functions, helper structs, and includes:

#if <PUT_MACRO_NAME_HERE>
  // DepthCam-only declaration or definition
#endif

For debug-only code:

#if <PUT_MACRO_NAME_HERE>
#if DEBUG_DEPTH_CAM_PARAM
  // debug code
#endif
#endif

Macro OFF requirements

When <PUT_MACRO_NAME_HERE> is undefined or set to 0:

- No DepthCam parameter loading.
- No DepthCam syntax writing.
- No DepthCam syntax parsing.
- No DepthCam store update.
- No DepthCam-specific fields used by original code.
- No DepthCam logs.
- No changed bitstream syntax.
- No changed encoder behavior.
- No changed decoder behavior.
- Original VTM bitstream compatibility must be preserved.

Existing function modification rule

If an existing function is modified, preserve the original code path.

Example:

void SomeExistingFunction(...)
{
#if <PUT_MACRO_NAME_HERE>
  // Modified version with DepthCam behavior.
#else
  // Original function body exactly as it existed before.
#endif
}

If only a small insertion is needed and the rest of the function is unchanged, this form is also acceptable:

original_code_before();
#if <PUT_MACRO_NAME_HERE>
  depthCamOnlyCode();
#endif
original_code_after();

Use the second form only when macro-OFF compiled output and behavior remain exactly original.

New API rule

Any new DepthCam-only API must be guarded.

Example:

class DepthCamParam
{
public:
#if <PUT_MACRO_NAME_HERE>
  bool ensureIntrinsicHeaderLoaded();
  bool ensureExtrinsicLoadedUpToPOC(int pocCurr);
#endif
};

New include rule

If a new include is needed only for DepthCam code, guard it.

Example:

#if <PUT_MACRO_NAME_HERE>
#include "DepthCamParam.h"
#include <map>
#endif

Syntax rule

DepthCam syntax must not exist when the macro is OFF.

When macro OFF:

HLSWriter must write the exact original slice header syntax.
HLSyntaxReader must read the exact original slice header syntax.
No DepthCam syntax element may be written or read.

Build requirement

Build both configurations:

1. Macro OFF
2. Macro ON

Macro OFF must compile and behave as original VTM.

Macro ON must compile and enable DepthCam behavior.

Do not use runtime flags as the primary guard

Runtime config flags may be added later, but they are not a replacement for the compile-time macro.

The compile-time macro is mandatory.

Report requirement

When implementation is complete, report:

Macro used:
- ...
Files with macro-guarded changes:
- ...
New macro-guarded fields:
- ...
New macro-guarded functions:
- ...
Existing functions modified:
- ...
Macro OFF build result:
- ...
Macro ON build result:
- ...























test_skill.md

Skill: DepthCam slice-header integration build and functional test

Goal

Verify that the DepthCam slice-header integration builds and runs correctly without relying on encoder/decoder parameter debug-log comparison.

This test skill validates:

- Macro OFF build
- Macro ON build
- Encoder run
- Decoder run
- Decoder MD5/checksum success
- No crash/assert during DepthCam parameter load/write/read/use
- Macro OFF behavior remains original-compatible

Do not add direct encoder-vs-decoder camera-parameter log comparison.

Do not require per-POC parameter checksum logs.

Required commands

The user will fill in actual commands.

Use this template:

ENCODER_CMD="<PUT_ENCODER_COMMAND_HERE>"
DECODER_CMD="<PUT_DECODER_COMMAND_HERE>"
ENC_LOG="depthcam_encoder.log"
DEC_LOG="depthcam_decoder.log"
CHECK_LOG="depthcam_test.log"
${ENCODER_CMD} > ${ENC_LOG} 2>&1
${DECODER_CMD} > ${DEC_LOG} 2>&1
python check_depthcam_run.py --enc-log ${ENC_LOG} --dec-log ${DEC_LOG} > ${CHECK_LOG} 2>&1

The following files must be created:

depthcam_encoder.log
depthcam_decoder.log
depthcam_test.log

Required checker script

Create:

check_depthcam_run.py

Required usage:

python check_depthcam_run.py --enc-log depthcam_encoder.log --dec-log depthcam_decoder.log

The script must print PASS/FAIL messages to stdout.

Checker requirements

The checker must inspect encoder and decoder logs for run-level success/failure only.

Encoder log checks

Pass if:

- Encoder log exists.
- Encoder log is non-empty.
- Encoder process completed without fatal error.

Fail if encoder log contains obvious fatal failure patterns such as:

- "ERROR"
- "Error:"
- "failed"
- "Failed"
- "assert"
- "CHECK"
- "Segmentation fault"
- "Aborted"
- "Exception"

Use case-insensitive matching where appropriate, but avoid false positives from harmless text if the codebase commonly prints benign “error” strings.

Decoder log checks

Pass if:

- Decoder log exists.
- Decoder log is non-empty.
- Decoder process completed without fatal error.
- Decoder MD5/checksum result is OK.

Fail if decoder log contains fatal failure patterns such as:

- "ERROR"
- "Error:"
- "failed"
- "Failed"
- "assert"
- "CHECK"
- "Segmentation fault"
- "Aborted"
- "Exception"

Decoder MD5/checksum check

Pass:

- Decoder log contains at least one MD5/checksum line with OK.
- Decoder log contains no MD5/checksum line with ERROR or mismatch.

Fail:

- No MD5/checksum OK line is found.
- Any MD5/checksum line contains ERROR.
- Any checksum mismatch line is found.

The checker should print:

[PASS] encoder log exists
[PASS] encoder completed without fatal errors
[PASS] decoder log exists
[PASS] decoder completed without fatal errors
[PASS] decoder MD5/checksum checks are OK
FINAL RESULT: PASS

On failure, print exact reasons:

[FAIL] encoder fatal error found: <full line>
[FAIL] decoder MD5 ERROR found: <full line>
FINAL RESULT: FAIL

Macro OFF validation

Build and run with <PUT_MACRO_NAME_HERE> OFF.

Expected:

- Build succeeds.
- No DepthCam syntax exists in slice header.
- No DepthCam parameter loading occurs.
- Encoder/decoder behavior remains original-compatible.
- Decoder MD5/checksum lines are OK.

If possible, compare macro-OFF output against an original baseline.

Fail if:

- Build fails.
- Encoder fails.
- Decoder fails.
- Decoder MD5/checksum mismatch occurs.
- Any DepthCam behavior appears in macro-OFF output.

Macro ON validation

Build and run with <PUT_MACRO_NAME_HERE> ON.

Expected:

- Build succeeds.
- Encoder runs.
- Decoder runs.
- DepthCam parameter loading/writing/parsing does not crash.
- Decoder MD5/checksum lines are OK.

Fail if:

- Build fails.
- Encoder fails.
- Decoder fails.
- Decoder MD5/checksum mismatch occurs.
- DepthCam parameter loading fails.
- DepthCam parameter parsing fails.
- Decoder cannot find required DepthCam param for current POC.

Required final report

Report:

Encoder command used:
- ...
Decoder command used:
- ...
Encoder log path:
- depthcam_encoder.log
Decoder log path:
- depthcam_decoder.log
Test log path:
- depthcam_test.log
Macro OFF build result:
- ...
Macro OFF run result:
- ...
Macro ON build result:
- ...
Macro ON run result:
- ...
Decoder MD5/checksum result:
- ...
Final result:
- PASS or FAIL















