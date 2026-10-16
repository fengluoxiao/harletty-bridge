# Format-scoped JOC input conditioning

The 48 kHz, five-fullband-input, downmix-configuration-3 path applies a ten-QMF-slot
front delay, a delayed -j surround rotation in bands 1..63, and the 21-tap complex
band-zero filter in src/eac3dec/joc_input.rs. Sparse channel indices and coefficient
words use cumulative reconstruction; unselected entries dequantize to zero. Both
changes are gated together. Other configurations retain the previous calculation.

This policy was tested by SDA with complete-scene local listening fixtures. It is
based on TheM14/JustOneCacophony revision 428772eb87275e3227efb3fc37911e7263234fda
(src/joc_qmf.py, src/joc_decode.py, native/src/eac3joc_core.cpp, data/tables.npz).
https://github.com/TheM14/JustOneCacophony/tree/428772eb87275e3227efb3fc37911e7263234fda
The MIT copyright/permission notice is retained in JOC_INPUT_CONDITIONING_LICENSE.
This research implementation is not an official Dolby conformance reference.

ObjectPcmFrame::processing_latency_samples() reports 1217 samples on that path and
577 elsewhere. Consumers must add the reported latency to OAMD event timestamps.
Core/LFE companions already include the same delay. Explicit reset, sequence gaps,
non-JOC frames and configuration changes discard the conditioning history.
No output makeup gain, content fingerprinting, file prescan or analysis cache is used.
