# Coding Plan: DepthCam parameter bitstream write/read integration

Goal:
Implement DepthCam parameter loading, Picture Header writing, Picture Header parsing, and encoder/decoder store synchronization.

Current target behavior:

1. Encoder side:

   * EncGOP computes the actual `pocCurr`.
   * DepthCam params must be loaded sequentially by display POC.
   * For RA, when encoder reaches POC 32, it must load POC 1,2,3,...,32 sequentially and write them as one bundle.
   * Access/use key is always `pocCurr`, never `gopId`.

2. Bitstream side:

   * At POC 0, write the intrinsic/header information.
   * POC 0 extrinsic is all zero, so do not write POC 0 extrinsic.
   * For POC > 0, write a parameter bundle only when `pocCurr > lastWrittenParamPOC`.
   * Example for RA GOP32:

     * POC 32 writes extrinsic params for POC 1~32.
     * POC 16/8/4/2/1 do not write again because already written.
     * POC 64 writes extrinsic params for POC 33~64.

3. Decoder side:

   * Parse intrinsic/header from Picture Header.
   * Parse extrinsic bundle from Picture Header.
   * Store parsed params by POC.
   * When decoding each POC, get the same parameter that the encoder used for that POC.

Implementation tasks:

A. DepthCamParam / EncLib side

Add or update DepthCamParam state:

* `m_lastLoadedParamPOC`

  * Meaning: all params from POC 1 to `m_lastLoadedParamPOC` are loaded sequentially.
  * Initial value: 0 or -1 depending on existing convention.
  * If POC 0 intrinsic/header is loaded separately, keep extrinsic last-loaded starting from 0.

* `m_lastWrittenParamPOC`

  * Meaning: all extrinsic params from POC 1 to `m_lastWrittenParamPOC` have already been written into the bitstream.
  * Initial value: 0.
  * This is separate from `m_lastLoadedParamPOC`.

Add functions similar to:

```cpp
bool ensureIntrinsicHeaderLoaded();

bool ensureExtrinsicLoadedUpToPOC(int pocCurr);

const DepthCamIntrinsicHeader& getIntrinsicHeader() const;

const DepthCamExtrinsicParam& getExtrinsicParam(int poc) const;

int getLastWrittenParamPOC() const;
void setLastWrittenParamPOC(int poc);
```

Required invariant:

```text
m_lastLoadedParamPOC == N means:
POC 1..N extrinsic params are loaded with no holes.
```

Sequential load behavior:

```cpp
if (pocCurr <= m_lastLoadedParamPOC)
{
  return true;
}

for (int poc = m_lastLoadedParamPOC + 1; poc <= pocCurr; ++poc)
{
  if (!loadOneExtrinsicParamForPOC(poc))
  {
    return false;
  }

  storeExtrinsicParam(poc, loadedParam);
  m_lastLoadedParamPOC = poc;
}
```

Important:

* If loading fails at POC N, do not advance `m_lastLoadedParamPOC` to N.
* Do not use `gopId` as a parameter index.
* Do not allow stale params from a previous POC to be reused silently.

B. PicHeader syntax fields

Add fields to `PicHeader` or the appropriate Picture Header structure.

Suggested fields:

```cpp
bool m_depthCamIntrinsicHeaderPresentFlag = false;
DepthCamIntrinsicHeader m_depthCamIntrinsicHeader;

bool m_depthCamParamBundlePresentFlag = false;
int  m_depthCamParamBundleStartPoc = 0;
int  m_depthCamParamBundleNumPics = 0;
std::vector<DepthCamExtrinsicParam> m_depthCamExtrinsicParamBundle;
```

Also update:

* PicHeader reset/init code.
* PicHeader copy/assignment if needed.
* Any tracing or debug dump code if required.

C. EncGOP::compressGOP integration

In `EncGOP::compressGOP`, after `pocCurr` is computed and after `picHeader = pic->m_cs->picHeader` is available, prepare Picture Header fields.

Pseudo behavior:

```cpp
picHeader->m_depthCamIntrinsicHeaderPresentFlag = false;
picHeader->m_depthCamParamBundlePresentFlag = false;
picHeader->m_depthCamExtrinsicParamBundle.clear();

auto& depthCamParam = m_pcEncLib->getDepthCamParam();

if (pocCurr == 0)
{
  depthCamParam.ensureIntrinsicHeaderLoaded();

  picHeader->m_depthCamIntrinsicHeaderPresentFlag = true;
  picHeader->m_depthCamIntrinsicHeader = depthCamParam.getIntrinsicHeader();

  // POC 0 extrinsic is all zero, do not write it.
  picHeader->m_depthCamParamBundlePresentFlag = false;
}
else
{
  depthCamParam.ensureExtrinsicLoadedUpToPOC(pocCurr);

  const int lastWritten = depthCamParam.getLastWrittenParamPOC();

  if (pocCurr > lastWritten)
  {
    const int startPoc = lastWritten + 1;
    const int endPoc = pocCurr;

    picHeader->m_depthCamParamBundlePresentFlag = true;
    picHeader->m_depthCamParamBundleStartPoc = startPoc;
    picHeader->m_depthCamParamBundleNumPics = endPoc - startPoc + 1;

    for (int poc = startPoc; poc <= endPoc; ++poc)
    {
      picHeader->m_depthCamExtrinsicParamBundle.push_back(depthCamParam.getExtrinsicParam(poc));
    }

    depthCamParam.setLastWrittenParamPOC(endPoc);
  }
}
```

Make sure this is done before the Picture Header is written.

Current EncGOP path:

* Separate PH path uses `xWritePicHeader(accessUnit, pic->m_cs->picHeader)`.
* Embedded PH path uses `m_HLSWriter->codeSliceHeader(pcSlice)`, which can include PH.
* Therefore do not manually write bits in EncGOP.
* Only fill `PicHeader`.
* Actual syntax write must be in `HLSWriter::codePictureHeader()`.

D. HLSWriter / HLSReader syntax

Add symmetric syntax to Picture Header writing/parsing.

Writer side:

* Find `HLSWriter::codePictureHeader(...)`.
* Add syntax for:

  * `depth_cam_intrinsic_header_present_flag`
  * intrinsic/header payload if present
  * `depth_cam_param_bundle_present_flag`
  * bundle start POC
  * bundle num pics
  * extrinsic params for each POC in bundle

Parser side:

* Find matching Picture Header parser, likely `HLSReader::parsePictureHeader(...)` or equivalent.
* Read fields in the exact same order.
* Store parsed values in `PicHeader`.

Important:

* Writer and reader order must be exactly identical.
* Do not write POC 0 extrinsic.
* Use fixed, deterministic coding for values.
* Use signed syntax for signed extrinsic values.
* Use unsigned syntax for counts/indices.
* Keep names clear:

  * `depth_cam_intrinsic_header_present_flag`
  * `depth_cam_param_bundle_present_flag`
  * `depth_cam_param_bundle_start_poc`
  * `depth_cam_param_bundle_num_pics`

E. Decoder-side store synchronization

Add decoder-side DepthCamParam store if not already present.

Required behavior:

1. When Picture Header is parsed:

   * If intrinsic/header flag is present, store intrinsic/header in decoder DepthCamParam.
   * If bundle flag is present, store each extrinsic param by POC:

     * `poc = startPoc + i`
     * `storeExtrinsicParam(poc, parsedBundle[i])`

2. In decoder GOP/picture decoding path:

   * Before using DepthCam params for current picture, set or fetch the param for `pocCurr`.
   * POC 0 extrinsic should be treated as default all-zero.
   * For POC > 0, decoder must find the param in store by POC.

Suggested decoder flow:

```cpp
if (picHeader->m_depthCamIntrinsicHeaderPresentFlag)
{
  decLib.getDepthCamParam().setIntrinsicHeader(picHeader->m_depthCamIntrinsicHeader);
}

if (picHeader->m_depthCamParamBundlePresentFlag)
{
  int startPoc = picHeader->m_depthCamParamBundleStartPoc;
  int numPics = picHeader->m_depthCamParamBundleNumPics;

  for (int i = 0; i < numPics; ++i)
  {
    int poc = startPoc + i;
    decLib.getDepthCamParam().setExtrinsicParam(poc, picHeader->m_depthCamExtrinsicParamBundle[i]);
  }
}
```

Then, when decoding POC N:

```cpp
const auto& param = decLib.getDepthCamParam().getExtrinsicParam(pocCurr);
```

F. Debug logs

Add temporary logs guarded by `DEBUG_DEPTH_CAM_PARAM`.

Encoder logs:

```text
[DepthCam][ENC][PH_INTRINSIC_WRITE] poc=0 checksum=<checksum>
[DepthCam][ENC][PH_BUNDLE_WRITE] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][ENC][EXTRINSIC] poc=1 checksum=<checksum>
[DepthCam][ENC][EXTRINSIC] poc=2 checksum=<checksum>
...
```

Decoder logs:

```text
[DepthCam][DEC][PH_INTRINSIC_READ] poc=0 checksum=<checksum>
[DepthCam][DEC][PH_BUNDLE_READ] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][DEC][EXTRINSIC] poc=1 checksum=<checksum>
[DepthCam][DEC][EXTRINSIC] poc=2 checksum=<checksum>
...
[DepthCam][DEC][PARAM_USE] poc=16 available=1 checksum=<checksum>
```

Checksum rule:

* Implement a deterministic checksum/hash for intrinsic/header.
* Implement a deterministic checksum/hash for each extrinsic parameter.
* Use the same checksum function on encoder and decoder sides.

G. Do not change unrelated behavior

Do not:

* Change prediction logic yet.
* Change RDO behavior.
* Use `gopId` as parameter index.
* Write bits directly from EncGOP.
* Write POC 0 extrinsic.
* Refactor unrelated VTM code.

Report:

* Modified files.
* Modified functions.
* New fields.
* New syntax order.
* Build result.






































































# Mandatory Macro Guard Rule

Use a compile-time macro for every DepthCam parameter bitstream/load/store/log change.

Macro name:

```cpp
<PUT_MACRO_NAME_HERE>
```

Strict rule:
Every modification to existing VTM behavior must be guarded by:

```cpp
#if <PUT_MACRO_NAME_HERE>
  // New DepthCam parameter behavior
#else
  // Original existing VTM code, kept exactly as before
#endif
```

Requirements:

1. Macro OFF behavior

* When `<PUT_MACRO_NAME_HERE>` is 0 or undefined, encoder and decoder must behave exactly like the original code.
* No DepthCam parameter loading.
* No Picture Header DepthCam syntax writing.
* No Picture Header DepthCam syntax reading.
* No DepthCam store update.
* No DepthCam debug logs.
* No changed bitstream.
* No changed encoder/decoder behavior.

2. Existing function modifications
   For every existing function that is modified, preserve the original code path in the `#else` branch.

Example:

```cpp
#if <PUT_MACRO_NAME_HERE>
  // modified code with DepthCam support
#else
  // original code exactly as it existed before this change
#endif
```

3. New fields/classes/functions
   New DepthCam-only fields, functions, and helper structs must be guarded by:

```cpp
#if <PUT_MACRO_NAME_HERE>
  // new declaration or definition
#endif
```

4. Picture Header syntax
   DepthCam Picture Header syntax must only exist inside:

```cpp
#if <PUT_MACRO_NAME_HERE>
#endif
```

When the macro is OFF:

* `HLSWriter` must write the exact original Picture Header syntax.
* `HLSReader` must read the exact original Picture Header syntax.
* Encoder and decoder bitstreams must remain compatible with original VTM.

5. Debug logs
   All debug logs must be additionally guarded.

Use:

```cpp
#if <PUT_MACRO_NAME_HERE>
#if DEBUG_DEPTH_CAM_PARAM
  // log
#endif
#endif
```

or equivalent.

6. Build requirement
   Build both configurations:

* macro OFF
* macro ON

Macro OFF must compile and behave as the original code.
Macro ON must compile and enable the new DepthCam parameter behavior.

7. Do not use runtime flags as the primary guard
   Do not rely only on config/runtime flags.
   The compile-time macro guard is mandatory.

























































# Updated Validation Skill: DepthCam parameter verification with generated check log

Goal:
Verify that encoder-written DepthCam parameters are reconstructed identically by decoder, and that decoder MD5/checksum results are all OK.

I will fill in the actual encoder and decoder commands.

Commands:

```bash
ENCODER_CMD="<PUT_ENCODER_COMMAND_HERE>"
DECODER_CMD="<PUT_DECODER_COMMAND_HERE>"

ENC_LOG="depthcam_encoder.log"
DEC_LOG="depthcam_decoder.log"
CHECK_LOG="depthcam_verify.log"
```

Run commands:

```bash
${ENCODER_CMD} > ${ENC_LOG} 2>&1
${DECODER_CMD} > ${DEC_LOG} 2>&1
python verify_depthcam_param_logs.py --enc-log ${ENC_LOG} --dec-log ${DEC_LOG} > ${CHECK_LOG} 2>&1
```

Claude Code must automatically create all three logs:

```text
depthcam_encoder.log
depthcam_decoder.log
depthcam_verify.log
```

`depthcam_verify.log` is mandatory.

Required checker script:
Create:

```text
verify_depthcam_param_logs.py
```

The script must support:

```bash
python verify_depthcam_param_logs.py --enc-log depthcam_encoder.log --dec-log depthcam_decoder.log
```

The script must write all PASS/FAIL messages to stdout so that redirecting stdout/stderr creates `depthcam_verify.log`.

Validation checks:

1. Intrinsic/header check

Expected logs:

```text
[DepthCam][ENC][PH_INTRINSIC_WRITE] poc=0 checksum=<checksum>
[DepthCam][DEC][PH_INTRINSIC_READ] poc=0 checksum=<checksum>
```

Pass:

* Encoder intrinsic/header checksum exists.
* Decoder intrinsic/header checksum exists.
* Checksums are identical.
* Intrinsic/header is written once at POC 0.
* POC 0 extrinsic is not written.

Fail:

* Missing encoder intrinsic log.
* Missing decoder intrinsic log.
* Checksum mismatch.
* Multiple unexpected intrinsic writes.
* Any POC 0 extrinsic write.

2. RA bundle check

For RA GOP32, expected first GOP:

```text
[DepthCam][ENC][PH_BUNDLE_WRITE] currPoc=32 startPoc=1 numPics=32 endPoc=32
[DepthCam][DEC][PH_BUNDLE_READ] currPoc=32 startPoc=1 numPics=32 endPoc=32
```

Expected behavior:

* Encoder writes POC 1~32 bundle when it reaches POC 32.
* Decoder reads the same bundle.
* POC 16, 8, 4, 2, 1 must not require additional parameter writing because their params were already transmitted.

3. Per-POC extrinsic checksum check

For every encoder line:

```text
[DepthCam][ENC][EXTRINSIC] poc=N checksum=X
```

Decoder must have:

```text
[DepthCam][DEC][EXTRINSIC] poc=N checksum=X
```

Pass:

* Every encoder extrinsic POC exists in decoder log.
* Every matching POC checksum is identical.
* Decoder has no missing POC in the transmitted bundle.
* Decoder parameter use log is available and matches stored checksum:

```text
[DepthCam][DEC][PARAM_USE] poc=N available=1 checksum=X
```

Fail:

* Missing POC.
* Checksum mismatch.
* `available=0`.
* Stale checksum reused from another POC.

4. Decoder MD5/checksum result check

Inspect decoder log.

Pass:

* Decoder log contains at least one MD5/checksum `OK`.
* Decoder log contains no MD5/checksum `ERROR`.

Fail if decoder log contains `ERROR` in MD5/checksum lines.

5. Macro OFF check

Build and run with `<PUT_MACRO_NAME_HERE>` OFF.

Expected:

* Build succeeds.
* No DepthCam logs appear.
* No DepthCam syntax is written.
* Encoder/decoder behavior is identical to original VTM.
* Decoder MD5/checksum lines are all OK.

Fail:

* Any `[DepthCam]` log appears with macro OFF.
* Bitstream syntax changes with macro OFF.
* Build fails.
* Decoder mismatch occurs.

6. Macro ON check

Build and run with `<PUT_MACRO_NAME_HERE>` ON.

Expected:

* Build succeeds.
* DepthCam logs appear.
* Intrinsic/header checksum matches between encoder and decoder.
* Every extrinsic POC checksum matches.
* Decoder parameter use logs show `available=1`.
* Decoder MD5/checksum lines are all OK.

7. Required `depthcam_verify.log` output

On success, `depthcam_verify.log` must contain:

```text
[PASS] intrinsic/header checksum match
[PASS] no POC0 extrinsic written
[PASS] RA bundle write/read check passed
[PASS] all extrinsic checksums match
[PASS] decoder parameter use checks passed
[PASS] decoder MD5 checks are OK
FINAL RESULT: PASS
```

On failure, it must contain exact reasons, for example:

```text
[FAIL] POC 16 extrinsic checksum mismatch: enc=1234 dec=5678
[FAIL] Decoder used unavailable param at POC 8
[FAIL] Decoder MD5 ERROR found: <full line>
FINAL RESULT: FAIL
```

Final report must include:

* Encoder command used.
* Decoder command used.
* Encoder log path.
* Decoder log path.
* Verification log path.
* Macro OFF build result.
* Macro ON build result.
* Intrinsic/header checksum comparison result.
* Number of extrinsic POCs compared.
* First 10 compared POC/checksum pairs.
* Decoder MD5/checksum result.
* Final PASS/FAIL.




