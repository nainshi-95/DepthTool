I have modified CABACWriter.cpp and CABACReader.cpp.
My intended design is that the new camera projection tool has three types:
	1.	Regular camera projection mode
	●	camProjFlag = true
	●	mergeFlag = false
	2.	Camera projection merge mode
	●	camProjFlag = true
	●	camProjMergeFlag = true
	●	skip = false
	3.	Camera projection skip-merge mode
	●	camProjFlag = true
	●	camProjMergeFlag = true
	●	skip = true
Please verify whether the current implementation matches this intended behavior.
Specifically, check the following:
	1.	In CABACWriter.cpp, confirm that the flags are written in the correct syntax order and that the three modes are mutually exclusive.
	2.	In CABACReader.cpp, confirm that the flags are parsed in the exact same order as the writer and reconstruct the same CU state.
	3.	In the encoder-side tool decision path, confirm that:
	●	regular camera projection sets camProjFlag = true and mergeFlag = false
	●	camera projection merge sets camProjFlag = true, camProjMergeFlag = true, and skip = false
	●	camera projection skip-merge sets camProjFlag = true, camProjMergeFlag = true, and skip = true
	4.	In the decoder-side reconstruction path, confirm that these three cases trigger the correct camera projection tool behavior.
	5.	Check whether any existing merge, skip, MMVD, affine, GPM/Geo, CIIP, LIC, BCW, or other inter-tool syntax conflicts with these new flags.
	6.	Check whether any normal inter syntax such as ref_idx, mvd, mvp_idx, imv, affine_amvr, bcw, lic, or obmc is incorrectly coded or parsed when camera projection mode is selected.
	7.	Report any mismatch between encoder-side CU flags, CABAC bitstream syntax, decoder-side parsed flags, and actual tool execution.
Please do not modify the code first. Produce a verification report with PASS / FAIL / PARTIAL for each of the three tool types and list the exact files/functions where any mismatch occurs.
