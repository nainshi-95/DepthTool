Coding Plan: DepthCam parameter Slice Header bitstream integration

Goal

Implement DepthCam camera parameter loading, slice-header writing/parsing, and encoder/decoder parameter store synchronization.

DepthCam parameters are picture-level semantic data, but the bitstream syntax must be placed in the slice header, not the Picture Header, because the decoder can know the full reconstructed POC only inside parseSliceHeader().

Core rule:

Use display POC as the only key.
Do not use gopId as a parameter index.
Do not store long-term params in PicHeader.
All long-term state must be managed by DepthCamParam.

⸻

1. Mandatory macro guard

Use this compile-time macro for every DepthCam change:

<PUT_MACRO_NAME_HERE>

Use this debug macro for logs:

DEBUG_DEPTH_CAM_PARAM

Every modification to existing VTM behavior must be guarded.

For existing function modifications:

#if <PUT_MACRO_NAME_HERE>
  // New DepthCam behavior
#else
  // Original existing VTM code exactly as before
#endif

For new fields/classes/functions:

#if <PUT_MACRO_NAME_HERE>
  // DepthCam-only declarations or definitions
#endif

For debug logs:

#if <PUT_MACRO_NAME_HERE>
#if DEBUG_DEPTH_CAM_PARAM
  // DepthCam debug log
#endif
#endif

Macro OFF requirements:

- No DepthCam loading.
- No DepthCam writing.
- No DepthCam parsing.
- No DepthCam store update.
- No DepthCam logs.
- No bitstream syntax change.
- Encoder/decoder behavior identical to original VTM.

Build both:

1. Macro OFF
2. Macro ON

⸻

2. Final syntax location

Do not add DepthCam syntax to codePictureHeader() / parsePictureHeader().

Reason:

Decoder parsePictureHeader() only knows ph_pic_order_cnt_lsb.
Decoder parseSliceHeader() reconstructs full POC.
DepthCam syntax existence is inferred from full POC.
Therefore DepthCam syntax must be read/written from slice header.

Writer location

In HLSWriter::codeSliceHeader(...), add DepthCam syntax after normal slice header syntax where full POC is available.

Recommended position:

codeClippingValues(pcSlice);
#if <PUT_MACRO_NAME_HERE>
  codeDepthCamSliceHeaderPayload(pcSlice);
#endif

Reader location

In HLSyntaxReader::parseSliceHeader(...), add DepthCam parsing after full POC reconstruction and after corresponding normal syntax.

Recommended position:

parseClippingValues(pcSlice);
#if <PUT_MACRO_NAME_HERE>
  parseDepthCamSliceHeaderPayload(pcSlice);
#endif
std::vector<uint32_t> entryPointOffset;
...
m_pcBitstream->readByteAlignment();

Do not place DepthCam parsing after readByteAlignment().

⸻

3. DepthCam behavior rule

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

The same rule prevents duplicate syntax parsing/writing.

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

Therefore, no extra first-slice flag is required for DepthCam duplication prevention.

⸻

4. DepthCamParam manager state

Add all persistent DepthCam state to the DepthCamParam manager.

Suggested fields:

#if <PUT_MACRO_NAME_HERE>
int m_lastLoadedParamPOC  = 0;
int m_lastWrittenParamPOC = 0;
int m_lastReadParamPOC    = 0;
bool m_intrinsicHeaderLoaded = false;
bool m_intrinsicHeaderParsed = false;
DepthCamIntrinsicHeader m_intrinsicHeader;
std::map<int, DepthCamExtrinsicParam> m_extrinsicParamStore;
#endif

Meaning:

m_lastLoadedParamPOC:
  Encoder-side extrinsic params from POC 1 to this POC are loaded sequentially with no holes.
m_lastWrittenParamPOC:
  Extrinsic params from POC 1 to this POC have already been written into the bitstream.
m_lastReadParamPOC:
  Extrinsic params from POC 1 to this POC have already been parsed from the bitstream.
m_extrinsicParamStore:
  Long-term param storage by display POC.

Required functions:

#if <PUT_MACRO_NAME_HERE>
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
uint64_t calcIntrinsicChecksum(const DepthCamIntrinsicHeader& header) const;
uint64_t calcExtrinsicChecksum(const DepthCamExtrinsicParam& param) const;
#endif

Sequential loading rule:

#if <PUT_MACRO_NAME_HERE>
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
      // Do not advance last loaded POC on failure.
      return false;
    }
    setExtrinsicParam(poc, param);
    m_lastLoadedParamPOC = poc;
  }
  return true;
}
#endif

Do not silently reuse stale params from a previous POC.

⸻

5. Slice tracking fields

Add optional debug/status fields to Slice.

These are not long-term storage. They only record what happened for the current slice.

#if <PUT_MACRO_NAME_HERE>
int m_depthCamLastWrittenParamPOCBefore = 0;
int m_depthCamLastWrittenParamPOCAfter  = 0;
int m_depthCamLastReadParamPOCBefore = 0;
int m_depthCamLastReadParamPOCAfter  = 0;
int m_depthCamBundleStartPOC = 0;
int m_depthCamBundleEndPOC   = 0;
int m_depthCamBundleNumPics  = 0;
#endif

Persistent state must remain in DepthCamParam.

⸻

6. EncGOP integration

In EncGOP::compressGOP, after pocCurr is computed, ensure the required DepthCam params are loaded.

Do not write bits directly from EncGOP.

Pseudo behavior:

#if <PUT_MACRO_NAME_HERE>
auto& depthCamParam = m_pcEncLib->getDepthCamParam();
if (pocCurr == 0)
{
  CHECK(!depthCamParam.ensureIntrinsicHeaderLoaded(),
        "Failed to load DepthCam intrinsic/header");
  // POC 0 extrinsic is all-zero/default and is not loaded/written.
}
else
{
  CHECK(!depthCamParam.ensureExtrinsicLoadedUpToPOC(pocCurr),
        "Failed to load DepthCam extrinsic params");
}
#endif

Important:

- Access key is always pocCurr.
- Do not use gopId.
- For RA GOP32, when pocCurr == 32, load POC 1~32 sequentially.
- For later POC 16/8/4/2/1, do not load again if already loaded.

⸻

7. HLSWriter helper

Add:

#if <PUT_MACRO_NAME_HERE>
void HLSWriter::codeDepthCamSliceHeaderPayload(Slice* pcSlice);
void HLSWriter::codeDepthCamIntrinsicHeader(const DepthCamIntrinsicHeader& header);
void HLSWriter::codeDepthCamExtrinsicParam(const DepthCamExtrinsicParam& param);
#endif

Writer logic:

#if <PUT_MACRO_NAME_HERE>
void HLSWriter::codeDepthCamSliceHeaderPayload(Slice* pcSlice)
{
  const int pocCurr = pcSlice->m_poc;
  DepthCamParam& depthCamParam = getEncoderDepthCamParam(); // Use the actual existing access path.
  if (pocCurr == 0)
  {
    const auto& header = depthCamParam.getIntrinsicHeader();
    codeDepthCamIntrinsicHeader(header);
    pcSlice->m_depthCamLastWrittenParamPOCBefore = depthCamParam.getLastWrittenParamPOC();
    pcSlice->m_depthCamLastWrittenParamPOCAfter  = depthCamParam.getLastWrittenParamPOC();
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][ENC][PH_INTRINSIC_WRITE] poc=0 checksum=%llu\n",
            (unsigned long long)depthCamParam.calcIntrinsicChecksum(header));
#endif
    // Never write POC 0 extrinsic.
    return;
  }
  const int lastWritten = depthCamParam.getLastWrittenParamPOC();
  pcSlice->m_depthCamLastWrittenParamPOCBefore = lastWritten;
  if (pocCurr <= lastWritten)
  {
    pcSlice->m_depthCamLastWrittenParamPOCAfter = lastWritten;
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][ENC][SH_BUNDLE_SKIP] currPoc=%d lastWrittenPoc=%d\n",
            pocCurr,
            lastWritten);
#endif
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
#if DEBUG_DEPTH_CAM_PARAM
  fprintf(stderr,
          "[DepthCam][ENC][SH_BUNDLE_WRITE] currPoc=%d startPoc=%d numPics=%d endPoc=%d\n",
          pocCurr,
          startPoc,
          numPics,
          endPoc);
#endif
  for (int poc = startPoc; poc <= endPoc; ++poc)
  {
    CHECK(!depthCamParam.hasExtrinsicParam(poc),
          "Missing DepthCam extrinsic param before writing");
    const auto& param = depthCamParam.getExtrinsicParam(poc);
    codeDepthCamExtrinsicParam(param);
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][ENC][EXTRINSIC] poc=%d checksum=%llu\n",
            poc,
            (unsigned long long)depthCamParam.calcExtrinsicChecksum(param));
#endif
  }
  depthCamParam.setLastWrittenParamPOC(endPoc);
  pcSlice->m_depthCamLastWrittenParamPOCAfter = endPoc;
}
#endif

Notes:

- Use the actual existing access path to DepthCamParam.
- If HLSWriter cannot currently access EncLib/DepthCamParam, add a guarded pointer/reference setter.
- Do not create global state.

⸻

8. HLSReader helper

Add:

#if <PUT_MACRO_NAME_HERE>
void HLSyntaxReader::parseDepthCamSliceHeaderPayload(Slice* pcSlice);
void HLSyntaxReader::parseDepthCamIntrinsicHeader(DepthCamIntrinsicHeader& header);
void HLSyntaxReader::parseDepthCamExtrinsicParam(DepthCamExtrinsicParam& param);
#endif

Reader logic:

#if <PUT_MACRO_NAME_HERE>
void HLSyntaxReader::parseDepthCamSliceHeaderPayload(Slice* pcSlice)
{
  const int pocCurr = pcSlice->m_poc;
  DepthCamParam& depthCamParam = getDecoderDepthCamParam(); // Use the actual existing access path.
  if (pocCurr == 0)
  {
    DepthCamIntrinsicHeader header;
    parseDepthCamIntrinsicHeader(header);
    depthCamParam.setIntrinsicHeader(header);
    depthCamParam.setExtrinsicParam(0, depthCamParam.getDefaultZeroExtrinsicParam());
    pcSlice->m_depthCamLastReadParamPOCBefore = depthCamParam.getLastReadParamPOC();
    pcSlice->m_depthCamLastReadParamPOCAfter  = depthCamParam.getLastReadParamPOC();
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][DEC][PH_INTRINSIC_READ] poc=0 checksum=%llu\n",
            (unsigned long long)depthCamParam.calcIntrinsicChecksum(header));
#endif
    // Never read POC 0 extrinsic.
    return;
  }
  const int lastRead = depthCamParam.getLastReadParamPOC();
  pcSlice->m_depthCamLastReadParamPOCBefore = lastRead;
  if (pocCurr <= lastRead)
  {
    pcSlice->m_depthCamLastReadParamPOCAfter = lastRead;
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][DEC][SH_BUNDLE_SKIP] currPoc=%d lastReadPoc=%d\n",
            pocCurr,
            lastRead);
#endif
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
#if DEBUG_DEPTH_CAM_PARAM
  fprintf(stderr,
          "[DepthCam][DEC][SH_BUNDLE_READ] currPoc=%d startPoc=%d numPics=%d endPoc=%d\n",
          pocCurr,
          int(startPoc),
          int(numPics),
          endPoc);
#endif
  for (int i = 0; i < int(numPics); ++i)
  {
    const int poc = int(startPoc) + i;
    DepthCamExtrinsicParam param;
    parseDepthCamExtrinsicParam(param);
    depthCamParam.setExtrinsicParam(poc, param);
#if DEBUG_DEPTH_CAM_PARAM
    fprintf(stderr,
            "[DepthCam][DEC][EXTRINSIC] poc=%d checksum=%llu\n",
            poc,
            (unsigned long long)depthCamParam.calcExtrinsicChecksum(param));
#endif
  }
  depthCamParam.setLastReadParamPOC(pocCurr);
  pcSlice->m_depthCamLastReadParamPOCAfter = pocCurr;
}
#endif

Notes:

- Immediately copy parsed params into DepthCamParam manager.
- Do not use PicHeader as long-term storage.
- Do not parse anything when pocCurr <= lastReadParamPOC.

⸻

9. Truncated signed Exp-Golomb coding

Use cameraParamQuantTest.py as the reference.

Claude Code must inspect cameraParamQuantTest.py and port the same quantization and truncated signed Exp-Golomb behavior to C++.

Required helper functions:

#if <PUT_MACRO_NAME_HERE>
void xWriteTruncatedSignedExpGolomb(int value, int minVal, int maxVal, const char* name);
void xReadTruncatedSignedExpGolomb(int& value, int minVal, int maxVal, const char* name);
#endif

Requirements:

- Signed camera parameters must use truncated signed Exp-Golomb.
- Unsigned counts/indices use UVLC.
- Quantization scale, clipping, offset, rounding, min/max range must match cameraParamQuantTest.py.
- Encoder and decoder must reconstruct identical integer camera params.
- Do not silently clamp decoder values unless the Python reference does.
- Add CHECK for decoded out-of-range values.

Suggested syntax function structure:

#if <PUT_MACRO_NAME_HERE>
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
#endif

⸻

10. Slice header syntax order

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

11. Decoder parameter use

When decoder needs DepthCam params for any picture, use only DepthCamParam.

#if <PUT_MACRO_NAME_HERE>
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
#endif

Add debug use log:

#if <PUT_MACRO_NAME_HERE>
#if DEBUG_DEPTH_CAM_PARAM
const bool available = pocCurr == 0 || depthCamParam.hasExtrinsicParam(pocCurr);
uint64_t checksum = 0;
if (pocCurr == 0)
{
  checksum = depthCamParam.calcExtrinsicChecksum(depthCamParam.getDefaultZeroExtrinsicParam());
}
else if (available)
{
  checksum = depthCamParam.calcExtrinsicChecksum(depthCamParam.getExtrinsicParam(pocCurr));
}
fprintf(stderr,
        "[DepthCam][DEC][PARAM_USE] poc=%d available=%d checksum=%llu\n",
        pocCurr,
        available ? 1 : 0,
        (unsigned long long)checksum);
#endif
#endif

⸻

12. Debug logs

Encoder logs:

[DepthCam][ENC][PH_INTRINSIC_WRITE] poc=0 checksum=<checksum>
[DepthCam][ENC][SH_BUNDLE_WRITE] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][ENC][SH_BUNDLE_SKIP] currPoc=16 lastWrittenPoc=32
[DepthCam][ENC][EXTRINSIC] poc=1 checksum=<checksum>
[DepthCam][ENC][EXTRINSIC] poc=2 checksum=<checksum>
...

Decoder logs:

[DepthCam][DEC][PH_INTRINSIC_READ] poc=0 checksum=<checksum>
[DepthCam][DEC][SH_BUNDLE_READ] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][DEC][SH_BUNDLE_SKIP] currPoc=16 lastReadPoc=32
[DepthCam][DEC][EXTRINSIC] poc=1 checksum=<checksum>
[DepthCam][DEC][EXTRINSIC] poc=2 checksum=<checksum>
...
[DepthCam][DEC][PARAM_USE] poc=16 available=1 checksum=<checksum>

Checksum requirements:

- Deterministic.
- Same function on encoder and decoder sides.
- Use quantized integer fields.
- Do not hash raw float memory.
- Hash fields in explicit fixed order.

Suggested checksum style:

#if <PUT_MACRO_NAME_HERE>
static uint64_t depthCamHashUpdate(uint64_t h, int v)
{
  uint64_t x = uint64_t(int64_t(v)) ^ 0x9e3779b97f4a7c15ULL;
  return (h ^ x) * 1099511628211ULL;
}
#endif

⸻

13. Verification script

Create:

verify_depthcam_param_logs.py

Required usage:

python verify_depthcam_param_logs.py --enc-log depthcam_encoder.log --dec-log depthcam_decoder.log

The script must print all PASS/FAIL messages to stdout.

The caller will redirect stdout/stderr to:

depthcam_verify.log

Checks

Intrinsic/header check

Expected logs:

[DepthCam][ENC][PH_INTRINSIC_WRITE] poc=0 checksum=<checksum>
[DepthCam][DEC][PH_INTRINSIC_READ] poc=0 checksum=<checksum>

Pass:

- Encoder intrinsic exists exactly once.
- Decoder intrinsic exists exactly once.
- Checksums match.
- Encoder has no EXTRINSIC poc=0 log.

Bundle check

For RA GOP32:

[DepthCam][ENC][SH_BUNDLE_WRITE] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][DEC][SH_BUNDLE_READ] currPoc=32 startPoc=1 numPics=32 endPoc=32

Pass:

- Encoder and decoder bundle records match.
- POC 16/8/4/2/1 do not create duplicate bundle writes/reads after POC 1~32 are already covered.

Per-POC extrinsic checksum

For every encoder line:

[DepthCam][ENC][EXTRINSIC] poc=N checksum=X

Decoder must have:

[DepthCam][DEC][EXTRINSIC] poc=N checksum=X

Pass:

- Every encoder POC exists in decoder log.
- Every checksum matches.
- No missing POC in transmitted bundle.

Decoder parameter use

Expected:

[DepthCam][DEC][PARAM_USE] poc=N available=1 checksum=X

Pass:

- available=1 for all used POC.
- PARAM_USE checksum matches the decoded stored EXTRINSIC checksum for the same POC.
- No stale checksum reuse.

Decoder MD5/checksum

Pass:

- Decoder log contains at least one MD5/checksum OK.
- Decoder log contains no MD5/checksum ERROR.

Fail if any MD5/checksum line contains ERROR.

⸻

14. Validation command template

Claude Code must create all three logs.

ENCODER_CMD="<PUT_ENCODER_COMMAND_HERE>"
DECODER_CMD="<PUT_DECODER_COMMAND_HERE>"
ENC_LOG="depthcam_encoder.log"
DEC_LOG="depthcam_decoder.log"
CHECK_LOG="depthcam_verify.log"
${ENCODER_CMD} > ${ENC_LOG} 2>&1
${DECODER_CMD} > ${DEC_LOG} 2>&1
python verify_depthcam_param_logs.py --enc-log ${ENC_LOG} --dec-log ${DEC_LOG} > ${CHECK_LOG} 2>&1

depthcam_verify.log is mandatory.

Success output must include:

[PASS] intrinsic/header checksum match
[PASS] no POC0 extrinsic written
[PASS] slice-header bundle write/read check passed
[PASS] all extrinsic checksums match
[PASS] decoder parameter use checks passed
[PASS] decoder MD5 checks are OK
FINAL RESULT: PASS

Failure output must include exact reasons:

[FAIL] POC 16 extrinsic checksum mismatch: enc=1234 dec=5678
[FAIL] Decoder used unavailable param at POC 8
[FAIL] Decoder MD5 ERROR found: <full line>
FINAL RESULT: FAIL

⸻

15. Macro OFF validation

Build and run with <PUT_MACRO_NAME_HERE> OFF.

Expected:

- Build succeeds.
- No [DepthCam] logs appear.
- No DepthCam syntax exists in slice header.
- Bitstream remains compatible with original VTM.
- Encoder/decoder behavior is identical to original VTM.
- Decoder MD5/checksum lines are all OK.

Fail if:

- Any [DepthCam] log appears.
- Build fails.
- Decoder mismatch occurs.
- Bitstream syntax changes.

⸻

16. Macro ON validation

Build and run with <PUT_MACRO_NAME_HERE> ON.

Expected:

- Build succeeds.
- DepthCam logs appear.
- POC 0 intrinsic/header checksum matches.
- POC 0 extrinsic is not written.
- Slice-header bundle write/read records match.
- Every extrinsic POC checksum matches.
- Decoder PARAM_USE logs show available=1.
- Decoder MD5/checksum lines are all OK.

⸻

17. Required final report

Claude Code must report:

Modified files:
- ...
Modified functions:
- ...
New fields:
- ...
New helper functions:
- ...
New slice-header syntax order:
- ...
Macro OFF build result:
- ...
Macro ON build result:
- ...
Encoder command used:
- ...
Decoder command used:
- ...
Encoder log path:
- depthcam_encoder.log
Decoder log path:
- depthcam_decoder.log
Verification log path:
- depthcam_verify.log
Intrinsic/header checksum comparison:
- ...
Number of extrinsic POCs compared:
- ...
First 10 compared POC/checksum pairs:
- ...
Decoder MD5/checksum result:
- ...
Final result:
- PASS or FAIL

⸻

18. Do not change unrelated behavior

Do not:

- Do not modify prediction logic.
- Do not modify RDO behavior.
- Do not modify RPL behavior.
- Do not use gopId as a parameter index.
- Do not write bits from EncGOP directly.
- Do not write POC 0 extrinsic.
- Do not store long-term params in PicHeader.
- Do not add DepthCam behavior when macro is OFF.
- Do not rely on runtime flags as the primary guard.
- Do not refactor unrelated VTM code.

⸻

19. Implementation order

Recommended order:

1. Add macro definitions and guards.
2. Build macro OFF.
3. Add DepthCamParam manager state/functions.
4. Add Slice debug/status fields.
5. Add encoder-side sequential load in EncGOP.
6. Add HLSWriter slice-header payload helper.
7. Add HLSReader slice-header payload helper.
8. Port truncated signed Exp-Golomb from cameraParamQuantTest.py.
9. Add intrinsic/extrinsic syntax helpers.
10. Add deterministic checksum helpers.
11. Add debug logs.
12. Add decoder PARAM_USE log.
13. Create verify_depthcam_param_logs.py.
14. Build macro OFF.
15. Build macro ON.
16. Run encoder, decoder, verification script.
17. Report final results.
