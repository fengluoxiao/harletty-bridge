use abi_stable::std_types::RVec;
use bridge_api::{RChannelLabel, RDecodedFrame, RMetadataFrame};
use eac3::{
    AccessUnitInfo, BedChannel, CorePcmFrame, FrameType, OamdPayload, ObjectPcmPushResult,
    ParsedEmdfPayloadData, inspect_access_unit,
};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "bridge-perf")]
use std::time::Instant;

use crate::bridge::AtmosBridge;
use crate::frame_builders::float_to_pcm_i32;
use crate::labels::bed_channel_to_r;
use crate::metadata::{build_eac3_metadata_frame, declare_object_channels};

const LEGACY_AC3_SAMPLE_COUNT: u32 = 1536;
const LEGACY_AC3_CHANNEL_COUNT: u32 = 6;
const AC3_CHANNELS: [u8; 8] = [2, 1, 2, 3, 3, 4, 4, 5];
const AC3_FRAME_SIZE_WORDS: [[usize; 3]; 19] = [
    [64, 69, 96],
    [80, 87, 120],
    [96, 104, 144],
    [112, 121, 168],
    [128, 139, 192],
    [160, 174, 240],
    [192, 208, 288],
    [224, 243, 336],
    [256, 278, 384],
    [320, 348, 480],
    [384, 417, 576],
    [448, 487, 672],
    [512, 557, 768],
    [640, 696, 960],
    [768, 835, 1152],
    [896, 975, 1344],
    [1024, 1114, 1536],
    [1152, 1253, 1728],
    [1280, 1393, 1920],
];
const EAC3_BLOCKS: [u32; 4] = [1, 2, 3, 6];
const EAC3_SAMPLE_RATES: [u32; 3] = [48_000, 44_100, 32_000];

/// Process a raw E-AC3 access unit (one complete syncframe).
///
/// Attempts object-level decode first (JOC + OAMD), then falls back to
/// core PCM decode.  Converts the result into one [`RDecodedFrame`].
pub(crate) fn process_eac3_frame(
    bridge: &mut AtmosBridge,
    frame: &[u8],
) -> Result<RDecodedFrame, String> {
    emit_eac3_frame_diagnostic(bridge, frame);

    match bridge.eac3_object_decoder.push_access_unit(frame) {
        Ok(Some(result)) => {
            update_eac3_dialogue_level(bridge, &result.info);
            let sample_count = result.pcm.samples_per_channel();
            let base_sample_pos = bridge.eac3_total_samples;
            bridge.eac3_total_samples += sample_count as u64;
            let rf = build_eac3_frame_from_object(result, base_sample_pos, bridge);
            bridge.perf.maybe_report(bridge.eac3_frame_count);
            maybe_dump_ok_frame(frame, "obj");
            return Ok(rf);
        }
        Ok(None) => {}
        Err(_) => {}
    }

    match bridge.eac3_pcm_decoder.push_access_unit(frame) {
        Ok(result) => {
            let info = result.info;
            update_eac3_dialogue_level(bridge, &info);
            let pcm = result.pcm;
            let decoded_samples = pcm.samples_per_channel();
            let sample_count = decoded_samples as u32;
            let base_sample_pos = bridge.eac3_total_samples;
            bridge.eac3_total_samples += sample_count as u64;
            let rf = build_eac3_frame_from_core(&pcm, &info, base_sample_pos, bridge);
            bridge.perf.maybe_report(bridge.eac3_frame_count);
            maybe_dump_ok_frame(frame, "pcm");
            Ok(rf)
        }
        Err(e) => {
            let diag = eac3_frame_reject_diag(frame);
            let err_str = format!("{e}");
            if err_str == "not-eac3" {
                if let Some(sample_rate) = legacy_ac3_sample_rate(frame) {
                    return Ok(build_silence_frame(
                        sample_rate,
                        LEGACY_AC3_SAMPLE_COUNT,
                        bridge,
                    ));
                }
            }
            if err_str == "unsupported-feature:non-independent-core-pcm" {
                if let Some((sample_rate, sample_count)) = eac3_frame_timing(frame) {
                    maybe_dump_reject_frame(frame, "unsupported");
                    return Ok(build_silence_frame(sample_rate, sample_count, bridge));
                }
            }
            if err_str == "short-packet" {
                maybe_dump_short_packet_frame(frame);
                if let Some((sample_rate, sample_count)) = eac3_frame_timing(frame) {
                    maybe_dump_reject_frame(frame, "shortpkt");
                    bridge.eac3_diag_stats.short_packet_silence_frames += 1;
                    return Ok(build_silence_frame(sample_rate, sample_count, bridge));
                }
            }
            maybe_dump_reject_frame(frame, "parseerr");
            Err(format!("E-AC3 decode error: {e} {diag}"))
        }
    }
}

/// Merge a decoded core (5.1) with its dependent E-AC3 substream — the discrete
/// surround/back channels of a 7.1 extension — into one bed, via the shared
/// [`eac3::merge_core_with_dependent`] helper (also used by the harletty CLI).
/// Returns `None` (caller falls back to the core alone) when the dependent
/// can't be decoded or doesn't line up.
fn merge_eac3_core_with_dependent(
    bridge: &mut AtmosBridge,
    core: &CorePcmFrame,
    dependent_frame: &[u8],
) -> Option<CorePcmFrame> {
    eac3::merge_core_with_dependent(
        &mut bridge.eac3_dependent_pcm_decoder,
        core,
        dependent_frame,
    )
}

pub(crate) fn process_eac3_dependent_frame_with_core(
    bridge: &mut AtmosBridge,
    frame: &[u8],
    core: CorePcmFrame,
) -> Result<Option<RDecodedFrame>, String> {
    emit_eac3_frame_diagnostic(bridge, frame);

    // Non-JOC dependent pair = a plain channel-extension bed (AC-3 core +
    // dependent E-AC3 for 7.1), not objects. Emit the channel bed instead of
    // dropping it (which left these streams silent), so the renderer's
    // channel-render modes spatialise it. The AC-3 core (5.1) is already decoded
    // and in hand; the dependent's 7.1 extension is merged on top below once
    // available. Skip the object decoder entirely for this case.
    let dep_has_joc = inspect_access_unit(frame)
        .map(|i| i.joc_payload_count() > 0)
        .unwrap_or(false);
    if !dep_has_joc {
        let dep_info = inspect_access_unit(frame).map_err(|e| format!("{e}"))?;
        update_eac3_dialogue_level(bridge, &dep_info);
        // Merge the dependent's discrete surround/back channels onto the core for
        // a full 7.1 bed; fall back to the 5.1 core alone if the dependent can't
        // be merged (never silent).
        let bed = merge_eac3_core_with_dependent(bridge, &core, frame).unwrap_or(core);
        let sample_count = bed.samples_per_channel();
        bridge.eac3_total_samples += sample_count as u64;
        bridge.eac3_diag_stats.dependent_pair_channel_beds += 1;
        bridge.perf.maybe_report(bridge.eac3_frame_count);
        return Ok(Some(build_eac3_channel_bed_frame(
            &bed,
            Some(&dep_info),
            bridge,
        )));
    }

    match bridge
        .eac3_object_decoder
        .push_access_unit_with_core(frame, core)
    {
        Ok(Some(result)) => {
            update_eac3_dialogue_level(bridge, &result.info);
            let sample_count = result.pcm.samples_per_channel();
            let base_sample_pos = bridge.eac3_total_samples;
            bridge.eac3_total_samples += sample_count as u64;
            bridge.eac3_diag_stats.paired_object_frames += 1;
            bridge.perf.maybe_report(bridge.eac3_frame_count);
            maybe_dump_ok_frame(frame, "depobj");
            Ok(Some(build_eac3_frame_from_object(
                result,
                base_sample_pos,
                bridge,
            )))
        }
        Ok(None) => Ok(None),
        Err(err) => {
            let diag = eac3_frame_reject_diag(frame);
            maybe_dump_reject_frame(frame, "depobj");
            Err(format!("E-AC3 dependent object decode error: {err} {diag}"))
        }
    }
}

pub(crate) fn is_legacy_ac3_frame(frame: &[u8]) -> bool {
    legacy_ac3_info(frame).is_some()
}

pub(crate) fn is_dependent_eac3_frame(frame: &[u8]) -> bool {
    inspect_access_unit(frame)
        .map(|info| info.frame_type == FrameType::Dependent)
        .unwrap_or(false)
}

pub(crate) fn diagnose_eac3_frame(bridge: &mut AtmosBridge, frame: &[u8]) {
    emit_eac3_frame_diagnostic(bridge, frame);
}

fn emit_eac3_frame_diagnostic(bridge: &mut AtmosBridge, frame: &[u8]) {
    bridge.eac3_diag_stats.total_frames += 1;

    match inspect_access_unit(frame) {
        Ok(info) => {
            match info.frame_type {
                FrameType::LegacyAc3 => bridge.eac3_diag_stats.legacy_ac3_frames += 1,
                FrameType::Independent => bridge.eac3_diag_stats.independent_frames += 1,
                FrameType::Dependent => bridge.eac3_diag_stats.dependent_frames += 1,
                FrameType::Ac3Convert => bridge.eac3_diag_stats.ac3_convert_frames += 1,
            }
            if info.joc_payload_count() > 0 {
                bridge.eac3_diag_stats.joc_frames += 1;
            }
            if info.oamd_payload_count() > 0 {
                bridge.eac3_diag_stats.oamd_frames += 1;
            }
        }
        Err(err) if format!("{err}") == "not-eac3" => {
            if legacy_ac3_info(frame).is_some() {
                bridge.eac3_diag_stats.legacy_ac3_frames += 1;
            }
        }
        Err(_) => {}
    }
}

pub(crate) fn is_temporary_eac3_silence_frame(frame: &RDecodedFrame) -> bool {
    frame.sampling_frequency != 0
        && frame.sample_count == LEGACY_AC3_SAMPLE_COUNT
        && frame.channel_count == LEGACY_AC3_CHANNEL_COUNT
        && frame.metadata.is_empty()
        && frame.pcm.iter().all(|sample| *sample == 0)
}

static SHORT_PACKET_DUMPED: AtomicBool = AtomicBool::new(false);

fn maybe_dump_short_packet_frame(frame: &[u8]) {
    if std::env::var_os("HARLETTY_DUMP_EAC3_SHORT_PACKET").is_none() {
        return;
    }
    if SHORT_PACKET_DUMPED.swap(true, Ordering::SeqCst) {
        return;
    }
    let path = "/tmp/eac3_short_packet.bin";
    if let Err(err) = std::fs::write(path, frame) {
        log::warn!("failed to dump short-packet frame to {path}: {err}");
    } else {
        log::warn!(
            "dumped first short-packet E-AC3 frame ({} bytes) to {path}",
            frame.len()
        );
    }
}

static REJECT_DUMP_CAP: OnceLock<usize> = OnceLock::new();
static REJECT_DUMP_INDEX: AtomicUsize = AtomicUsize::new(0);
static OK_DUMP_CAP: OnceLock<usize> = OnceLock::new();
static OK_DUMP_INDEX: AtomicUsize = AtomicUsize::new(0);

fn read_dump_cap_env(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

fn maybe_dump_reject_frame(frame: &[u8], reason: &str) {
    let cap = *REJECT_DUMP_CAP.get_or_init(|| read_dump_cap_env("HARLETTY_DUMP_EAC3_REJECTS"));
    if cap == 0 {
        return;
    }
    let idx = REJECT_DUMP_INDEX.fetch_add(1, Ordering::SeqCst);
    if idx >= cap {
        return;
    }
    let path = format!("/tmp/eac3_reject_{idx:04}_{reason}.bin");
    if let Err(err) = std::fs::write(&path, frame) {
        log::warn!("failed to dump rejected E-AC3 frame to {path}: {err}");
    } else {
        log::warn!(
            "dumped rejected E-AC3 frame [{reason}] ({} bytes) to {path}",
            frame.len()
        );
    }
}

fn maybe_dump_ok_frame(frame: &[u8], path_kind: &str) {
    let cap = *OK_DUMP_CAP.get_or_init(|| read_dump_cap_env("HARLETTY_DUMP_EAC3_OK"));
    if cap == 0 {
        return;
    }
    let idx = OK_DUMP_INDEX.fetch_add(1, Ordering::SeqCst);
    if idx >= cap {
        return;
    }
    let path = format!("/tmp/eac3_ok_{idx:04}_{path_kind}.bin");
    if let Err(err) = std::fs::write(&path, frame) {
        log::warn!("failed to dump accepted E-AC3 frame to {path}: {err}");
    } else {
        log::warn!(
            "dumped accepted E-AC3 frame [{path_kind}] ({} bytes) to {path}",
            frame.len()
        );
    }
}

fn eac3_frame_reject_diag(frame: &[u8]) -> String {
    let sync = frame
        .get(0..2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]));
    let bsid = frame_bsid(frame);
    let first8 = frame
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02X}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "sync={:?} bsid={:?} frame_bytes={} first8=[{}]",
        sync,
        bsid,
        frame.len(),
        first8
    )
}

fn frame_bsid(frame: &[u8]) -> Option<u8> {
    if frame.len() < 6 {
        return None;
    }
    let bit_offset = 40usize;
    let byte_index = bit_offset / 8;
    let bit_shift = 8 - (bit_offset % 8) - 5;
    frame.get(byte_index).map(|byte| (byte >> bit_shift) & 0x1F)
}

fn legacy_ac3_sample_rate(frame: &[u8]) -> Option<u32> {
    legacy_ac3_info(frame).map(|info| info.sample_rate)
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
struct LegacyAc3Info {
    bsid: u8,
    sample_rate: u32,
    frame_size: usize,
    channel_mode: u8,
    channels: u8,
    lfe_on: bool,
}

fn legacy_ac3_info(frame: &[u8]) -> Option<LegacyAc3Info> {
    let bsid = frame_bsid(frame)?;
    if bsid > 10 || frame.len() < 7 {
        return None;
    }

    let fscod = frame[4] >> 6;
    let frmsizecod = frame[4] & 0x3F;
    let sample_rate = match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => return None,
    };
    let frame_size = ac3_frame_size_from_fscod_frmsizecod(fscod, frmsizecod)?;
    let channel_mode = read_bits(frame, 48, 3)? as u8;
    let mut bit_offset = 51usize;
    if (channel_mode & 0x01) != 0 && channel_mode != 1 {
        bit_offset += 2;
    }
    if (channel_mode & 0x04) != 0 {
        bit_offset += 2;
    }
    if channel_mode == 2 {
        bit_offset += 2;
    }
    let lfe_on = read_bits(frame, bit_offset, 1)? != 0;
    let channels = AC3_CHANNELS[usize::from(channel_mode)] + u8::from(lfe_on);

    Some(LegacyAc3Info {
        bsid,
        sample_rate,
        frame_size,
        channel_mode,
        channels,
        lfe_on,
    })
}

fn ac3_frame_size_from_fscod_frmsizecod(fscod: u8, frmsizecod: u8) -> Option<usize> {
    let bitrate_index = usize::from(frmsizecod >> 1);
    if fscod > 2 || bitrate_index >= AC3_FRAME_SIZE_WORDS.len() {
        return None;
    }
    Some(AC3_FRAME_SIZE_WORDS[bitrate_index][usize::from(fscod)] * 2)
}

fn read_bits(data: &[u8], bit_offset: usize, bit_count: usize) -> Option<u32> {
    if bit_count > 32 || bit_offset + bit_count > data.len() * 8 {
        return None;
    }
    let mut value = 0u32;
    for bit in bit_offset..bit_offset + bit_count {
        let byte = *data.get(bit / 8)?;
        let bit_value = (byte >> (7 - (bit % 8))) & 1;
        value = (value << 1) | u32::from(bit_value);
    }
    Some(value)
}

fn eac3_frame_timing(frame: &[u8]) -> Option<(u32, u32)> {
    if frame.len() < 5 || frame_bsid(frame)? < 11 {
        return None;
    }

    let sr_code = (frame[4] >> 6) & 0x03;
    let (sample_rate, blocks) = if sr_code == 3 {
        let sr_code2 = (frame[4] >> 4) & 0x03;
        if sr_code2 == 3 {
            return None;
        }
        (EAC3_SAMPLE_RATES[usize::from(sr_code2)] / 2, 6)
    } else {
        let num_blocks_code = (frame[4] >> 4) & 0x03;
        (
            EAC3_SAMPLE_RATES[usize::from(sr_code)],
            EAC3_BLOCKS[usize::from(num_blocks_code)],
        )
    };

    Some((sample_rate, blocks * 256))
}

fn build_silence_frame(
    sample_rate: u32,
    sample_count: u32,
    bridge: &mut AtmosBridge,
) -> RDecodedFrame {
    let sample_count_usize = sample_count as usize;
    let channel_count = LEGACY_AC3_CHANNEL_COUNT as usize;
    bridge.eac3_total_samples += sample_count as u64;

    RDecodedFrame {
        sampling_frequency: sample_rate,
        sample_count,
        channel_count: LEGACY_AC3_CHANNEL_COUNT,
        pcm: vec![0; sample_count_usize * channel_count].into(),
        channel_labels: vec![
            RChannelLabel::L,
            RChannelLabel::R,
            RChannelLabel::C,
            RChannelLabel::LFE,
            RChannelLabel::Ls,
            RChannelLabel::Rs,
        ]
        .into(),
        metadata: RVec::new(),
        drc_gain: 1.0,
        drc_ramp_duration: 0,
        dialogue_level: bridge.current_dialogue_level.into(),
        is_new_segment: false,
    }
}

/// Build an [`RDecodedFrame`] from an E-AC3 object decode result.
fn build_eac3_frame_from_object(
    result: ObjectPcmPushResult,
    base_sample_pos: u64,
    bridge: &mut AtmosBridge,
) -> RDecodedFrame {
    let pcm_frame = &result.pcm;
    let info = &result.info;
    let core = &pcm_frame.core;
    let sample_count = pcm_frame.samples_per_channel();
    let sampling_frequency = core.sample_rate;

    // JOC reconstructs every OAMD essence except the ordinary LFE carried by
    // CorePcmFrame::lfe_channel. Keep non-LFE bed members (including LFE2) as
    // fixed labelled channels and expose only the dynamic suffix as objects.
    // Reject inconsistent multi-payload topology instead of binding a bed/dialogue
    // slot to another payload's moving trajectory.
    let Some(slot_layout) = eac3_joc_slot_layout(pcm_frame) else {
        let clear_metadata = clear_eac3_object_metadata(base_sample_pos, bridge);
        let mut frame = build_eac3_channel_bed_frame(core, Some(info), bridge);
        if let Some(metadata) = clear_metadata {
            frame.metadata.push(metadata);
        }
        return frame;
    };
    let dynamic_objects = slot_layout
        .slots
        .iter()
        .filter(|slot| matches!(slot, Eac3JocSlot::Dynamic { .. }))
        .count();
    let fixed_channels = usize::from(core.lfe_channel.is_some())
        + slot_layout.slots.len().saturating_sub(dynamic_objects);
    let (pcm, channel_labels) =
        build_eac3_object_output_pcm_and_labels(pcm_frame, &slot_layout.slots);
    let total_channel_count = channel_labels.len();

    #[cfg(feature = "bridge-perf")]
    let metadata_started = Instant::now();
    let mut metadata: RVec<RMetadataFrame> = RVec::new();
    #[cfg(feature = "bridge-perf")]
    let mut metadata_events = 0usize;
    if dynamic_objects == 0 {
        if let Some(clear_metadata) = clear_eac3_object_metadata(base_sample_pos, bridge) {
            metadata.push(clear_metadata);
        }
    } else {
        for (oamd, sample_offset) in &pcm_frame.oamd_payloads {
            let evo_base = base_sample_pos
                + sample_offset.unwrap_or(0) as u64
                + eac3::JOC_QMF_LATENCY_SAMPLES as u64;
            let meta = build_eac3_metadata_frame(
                oamd,
                evo_base,
                base_sample_pos,
                fixed_channels,
                dynamic_objects,
                bridge,
            );
            #[cfg(feature = "bridge-perf")]
            {
                metadata_events += meta.events.len();
            }
            metadata.push(meta);
        }
    }
    #[cfg(feature = "bridge-perf")]
    {
        let elapsed = metadata_started.elapsed();
        bridge.perf.record_build_metadata(elapsed);
        bridge
            .perf
            .note_built_frame(metadata.len(), metadata_events);
    }

    let (drc_gain, drc_ramp_duration) = eac3_drc_params(bridge, info, sample_count as u32);

    RDecodedFrame {
        sampling_frequency,
        sample_count: sample_count as u32,
        channel_count: total_channel_count as u32,
        pcm,
        channel_labels,
        metadata,
        drc_gain,
        drc_ramp_duration,
        dialogue_level: bridge.current_dialogue_level.into(),
        is_new_segment: false,
    }
}

fn clear_eac3_object_metadata(sample_pos: u64, bridge: &mut AtmosBridge) -> Option<RMetadataFrame> {
    let had_objects = bridge
        .declared_object_channels
        .as_ref()
        .is_some_and(|channels| !channels.is_empty());
    let object_channels =
        declare_object_channels(&mut bridge.declared_object_channels, RVec::new());
    had_objects.then_some(RMetadataFrame {
        events: RVec::new(),
        object_channels,
        channel_gains: RVec::new(),
        name_updates: RVec::new(),
        sample_pos,
        ramp_duration: 0,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Eac3JocPresentationLayout {
    has_lfe: bool,
    slots: Vec<Eac3JocSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Eac3JocSlot {
    Bed(BedChannel),
    Dynamic { id: u32 },
}

fn eac3_oamd_slot_layout(
    oamd: &OamdPayload,
    joc_slot_count: usize,
) -> Option<Eac3JocPresentationLayout> {
    if oamd.isf_in_use
        || oamd.bed_or_isf_objects > oamd.object_count
        || oamd.dynamic_objects != oamd.object_count - oamd.bed_or_isf_objects
    {
        return None;
    }
    let bed_labels: Vec<BedChannel> = oamd.bed_assignment.iter().flatten().copied().collect();
    if bed_labels.len() != oamd.beds || oamd.beds != oamd.bed_or_isf_objects {
        return None;
    }

    let lfe_count = bed_labels
        .iter()
        .filter(|&&label| label == BedChannel::LowFrequencyEffects)
        .count();
    if lfe_count > 1 {
        return None;
    }
    let has_lfe = lfe_count == 1;
    let mut slots = Vec::with_capacity(joc_slot_count);
    for label in bed_labels {
        if label != BedChannel::LowFrequencyEffects {
            slots.push(Eac3JocSlot::Bed(label));
        }
    }
    for dynamic_index in 0..oamd.dynamic_objects {
        slots.push(Eac3JocSlot::Dynamic {
            id: (10 + dynamic_index) as u32,
        });
    }
    (slots.len() == joc_slot_count).then_some(Eac3JocPresentationLayout { has_lfe, slots })
}

fn eac3_joc_slot_layout(pcm_frame: &eac3::ObjectPcmFrame) -> Option<Eac3JocPresentationLayout> {
    let (first, _) = pcm_frame.oamd_payloads.first()?;
    let layout = eac3_oamd_slot_layout(first, pcm_frame.object_channels.len())?;
    (layout.has_lfe == pcm_frame.core.lfe_channel.is_some()
        && pcm_frame.oamd_payloads.iter().skip(1).all(|(oamd, _)| {
            eac3_oamd_slot_layout(oamd, pcm_frame.object_channels.len()).as_ref() == Some(&layout)
        }))
    .then_some(layout)
}

fn build_eac3_object_output_pcm_and_labels(
    pcm_frame: &eac3::ObjectPcmFrame,
    slot_layout: &[Eac3JocSlot],
) -> (RVec<i32>, RVec<RChannelLabel>) {
    let core = &pcm_frame.core;
    let sample_count = pcm_frame.samples_per_channel();
    let total_channel_count = usize::from(core.lfe_channel.is_some()) + slot_layout.len();
    let mut pcm: RVec<i32> = RVec::with_capacity(sample_count * total_channel_count);

    for sample in 0..sample_count {
        if let Some(lfe) = &core.lfe_channel {
            pcm.push(float_to_pcm_i32(lfe[sample]));
        }
        for channel in &pcm_frame.object_channels {
            pcm.push(float_to_pcm_i32(channel[sample]));
        }
    }

    let mut channel_labels: RVec<RChannelLabel> = RVec::with_capacity(total_channel_count);
    if core.lfe_channel.is_some() {
        channel_labels.push(RChannelLabel::LFE);
    }
    for slot in slot_layout {
        channel_labels.push(match slot {
            Eac3JocSlot::Bed(label) => bed_channel_to_r(*label),
            Eac3JocSlot::Dynamic { .. } => RChannelLabel::Object,
        });
    }
    (pcm, channel_labels)
}

/// Build an [`RDecodedFrame`] from a core-PCM-only E-AC3 decode result.
fn build_eac3_frame_from_core(
    core: &CorePcmFrame,
    info: &AccessUnitInfo,
    base_sample_pos: u64,
    bridge: &mut AtmosBridge,
) -> RDecodedFrame {
    let sampling_frequency = core.sample_rate;
    let sample_count = core.samples_per_channel();
    let total_channel_count = core.total_channels();

    // Build interleaved PCM.
    let pcm_capacity = sample_count * total_channel_count;
    let mut pcm: RVec<i32> = RVec::with_capacity(pcm_capacity);

    for s in 0..sample_count {
        for ch in &core.fullband_channels {
            pcm.push(float_to_pcm_i32(ch[s]));
        }
        if let Some(lfe) = &core.lfe_channel {
            pcm.push(float_to_pcm_i32(lfe[s]));
        }
    }

    // Channel labels.
    let mut channel_labels: RVec<RChannelLabel> = RVec::with_capacity(total_channel_count);
    for bed in &core.fullband_channel_order {
        channel_labels.push(bed_channel_to_r(*bed));
    }
    if core.lfe_channel.is_some() {
        channel_labels.push(RChannelLabel::LFE);
    }

    // Extract metadata from OAMD payloads in the access unit info.
    let mut metadata: RVec<RMetadataFrame> = RVec::new();
    for payload in info.payloads() {
        if let ParsedEmdfPayloadData::Oamd(oamd) = &payload.parsed {
            let evo_base = base_sample_pos + payload.info.sample_offset.unwrap_or(0) as u64;
            let meta = build_eac3_metadata_frame(oamd, evo_base, base_sample_pos, 0, 0, bridge);
            metadata.push(meta);
        }
    }

    let (drc_gain, drc_ramp_duration) = eac3_drc_params(bridge, info, sample_count as u32);

    RDecodedFrame {
        sampling_frequency,
        sample_count: sample_count as u32,
        channel_count: total_channel_count as u32,
        pcm,
        channel_labels,
        metadata,
        drc_gain,
        drc_ramp_duration,
        dialogue_level: bridge.current_dialogue_level.into(),
        is_new_segment: false,
    }
}

/// Build a pure channel-bed frame (PCM + labels, **empty metadata**) from a
/// decoded core. Used for non-object content (a plain 5.1 / 7.1 bed) so the
/// renderer's channel-render modes (virtual / direct / host) spatialise it,
/// rather than the object path. Unlike [`build_eac3_frame_from_core`] this never
/// derives OAMD/object metadata. DRC and dialogue level are still carried.
/// Emit a standalone AC-3 core (no E-AC3 dependent followed it) as a plain 5.1
/// channel bed, so the renderer's channel-render modes can spatialise it. Plain
/// AC-3 streams never deliver a dependent, so without this the buffered core
/// would sit unpaired forever and the track would be silent. DRC/dialnorm come
/// from the AC-3 frame itself (`frame`).
pub(crate) fn build_standalone_ac3_core_frame(
    bridge: &mut AtmosBridge,
    core: &CorePcmFrame,
    frame: &[u8],
) -> RDecodedFrame {
    let info = inspect_access_unit(frame).ok();
    if let Some(info) = info.as_ref() {
        update_eac3_dialogue_level(bridge, info);
    }
    let sample_count = core.samples_per_channel();
    bridge.eac3_total_samples += sample_count as u64;
    bridge.eac3_diag_stats.standalone_ac3_core_beds += 1;
    bridge.perf.maybe_report(bridge.eac3_frame_count);
    build_eac3_channel_bed_frame(core, info.as_ref(), bridge)
}

fn build_eac3_channel_bed_frame(
    core: &CorePcmFrame,
    info: Option<&AccessUnitInfo>,
    bridge: &mut AtmosBridge,
) -> RDecodedFrame {
    let sample_count = core.samples_per_channel();
    let total_channel_count = core.total_channels();

    let mut pcm: RVec<i32> = RVec::with_capacity(sample_count * total_channel_count);
    for s in 0..sample_count {
        for ch in &core.fullband_channels {
            pcm.push(float_to_pcm_i32(ch[s]));
        }
        if let Some(lfe) = &core.lfe_channel {
            pcm.push(float_to_pcm_i32(lfe[s]));
        }
    }

    let mut channel_labels: RVec<RChannelLabel> = RVec::with_capacity(total_channel_count);
    for bed in &core.fullband_channel_order {
        channel_labels.push(bed_channel_to_r(*bed));
    }
    if core.lfe_channel.is_some() {
        channel_labels.push(RChannelLabel::LFE);
    }

    let (drc_gain, drc_ramp_duration) = info
        .map(|info| eac3_drc_params(bridge, info, sample_count as u32))
        .unwrap_or((1.0, 0));

    RDecodedFrame {
        sampling_frequency: core.sample_rate,
        sample_count: sample_count as u32,
        channel_count: total_channel_count as u32,
        pcm,
        channel_labels,
        metadata: RVec::new(),
        drc_gain,
        drc_ramp_duration,
        dialogue_level: bridge.current_dialogue_level.into(),
        is_new_segment: false,
    }
}

fn update_eac3_dialogue_level(bridge: &mut AtmosBridge, info: &AccessUnitInfo) {
    bridge.current_dialogue_level = Some(info.dialogue_normalization[0]);
}

fn eac3_drc_params(bridge: &AtmosBridge, info: &AccessUnitInfo, sample_count: u32) -> (f32, u32) {
    match bridge.drc_mode {
        crate::bridge::DrcMode::Off => (1.0, 0),
        crate::bridge::DrcMode::Standard => {
            let gain = info
                .block_drc
                .iter()
                .rev()
                .find_map(|block| {
                    block.dynamic_range_exists[0]
                        .then(|| decode_ac3_dynamic_range_word(block.dynamic_range_word[0]))
                })
                .unwrap_or(1.0);
            let ramp = if gain != 1.0 { sample_count } else { 0 };
            (gain, ramp)
        }
        crate::bridge::DrcMode::Heavy => {
            let gain = if info.heavy_compression_exists[0] {
                decode_ac3_heavy_range_word(info.heavy_compression_word[0])
            } else {
                info.block_drc
                    .iter()
                    .rev()
                    .find_map(|block| {
                        block.dynamic_range_exists[0]
                            .then(|| decode_ac3_dynamic_range_word(block.dynamic_range_word[0]))
                    })
                    .unwrap_or(1.0)
            };
            let ramp = if gain != 1.0 { sample_count } else { 0 };
            (gain, ramp)
        }
    }
}

fn decode_ac3_dynamic_range_word(word: u8) -> f32 {
    let v = ((word >> 5) as i32) - (((word >> 7) as i32) << 3) - 5;
    2.0f32.powi(v) * (((word & 0x1F) | 0x20) as f32)
}

fn decode_ac3_heavy_range_word(word: u8) -> f32 {
    let v = ((word >> 4) as i32) - (((word >> 7) as i32) << 4) - 4;
    2.0f32.powi(v) * (((word & 0x0F) | 0x10) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eac3::{
        AccessUnitInfo, AudioFrameInfo, AuxParseStatus, BedChannel, BlockDrcInfo, CorePcmFrame,
        EmdfSource, FrameType, JocPayload, OamdPayload, ObjectPcmFrame,
    };

    /// Local-only end-to-end check: decode a real non-JOC E-AC3 7.1 stream
    /// (AC-3 core + dependent) through the full bridge and confirm it now emits a
    /// non-silent multichannel bed (it used to drop the pair → silence). Skips
    /// gracefully when the local capture is absent. Writes the decoded PCM next
    /// to the fixture as `<fixture>.harletty_bridge.f32` for `compare_pcm`
    /// against ffmpeg. The fixture path can be overridden with
    /// `HARLETTY_EAC3_PAIR_FIXTURE` — keep such captures on disk, not tmpfs.
    #[test]
    fn aladdin_eac3_pair_emits_nonsilent_bed() {
        use crate::bridge::AtmosBridge;
        use abi_stable::std_types::RSlice;
        use bridge_api::{FormatBridge, RInputTransport};

        let path = std::env::var("HARLETTY_EAC3_PAIR_FIXTURE")
            .unwrap_or_else(|_| "/tmp/aladdin/fr.eac3".to_string());
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skip: {path} not present");
            return;
        };

        let mut bridge = AtmosBridge::new(false);
        let mut pcm_f32: Vec<f32> = Vec::new();
        let mut channels = 0u32;
        let mut frames = 0u64;
        let mut labels = String::new();
        // Feed in mpv-sized chunks so the raw extractor frames it like the host.
        for chunk in bytes.chunks(4096) {
            let r = bridge.push_packet(RSlice::from_slice(chunk), RInputTransport::Raw, 0);
            for f in r.frames.iter() {
                frames += 1;
                channels = f.channel_count;
                if labels.is_empty() {
                    labels = format!("{:?}", f.channel_labels);
                }
                // Bridge PCM is i32 scaled to 24-bit (see `float_to_pcm_i32`).
                for &s in f.pcm.iter() {
                    pcm_f32.push(s as f32 / 8_388_608.0);
                }
            }
        }
        eprintln!("labels: {labels}");

        let peak = pcm_f32.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        eprintln!("decoded {frames} frames, {channels} ch, peak={peak:.4}");
        // Dump before asserting so an ad-hoc fixture (other layouts fed via
        // HARLETTY_EAC3_PAIR_FIXTURE) still leaves PCM to compare against ffmpeg.
        let _ = std::fs::write(
            format!("{path}.harletty_bridge.f32"),
            pcm_f32
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        assert!(frames > 0, "no frames decoded from the E-AC3 pair stream");
        assert!(channels >= 6, "expected >= 5.1, got {channels} channels");
        assert!(peak > 1e-4, "decoded bed is silent (peak={peak})");
    }

    /// Local-only end-to-end check: decode a real DTS (DTS-HD MA / DTS:X) stream
    /// through the bridge's raw path and confirm it emits a non-silent
    /// multichannel bed (the 2D core/HD path; DTS:X objects are ignored). Skips
    /// when the local capture is absent; dumps PCM for `compare_pcm` vs ffmpeg.
    #[test]
    fn dts_decodes_nonsilent_bed() {
        use crate::bridge::AtmosBridge;
        use abi_stable::std_types::RSlice;
        use bridge_api::{FormatBridge, RInputTransport};

        let path = "/tmp/dts/sample.dts";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skip: {path} not present");
            return;
        };

        let mut bridge = AtmosBridge::new(false);
        let mut pcm_f32: Vec<f32> = Vec::new();
        let mut channels = 0u32;
        let mut frames = 0u64;
        let mut labels = String::new();
        for chunk in bytes.chunks(4096) {
            let r = bridge.push_packet(RSlice::from_slice(chunk), RInputTransport::Raw, 0);
            for f in r.frames.iter() {
                frames += 1;
                channels = f.channel_count;
                if labels.is_empty() {
                    labels = format!("{:?}", f.channel_labels);
                }
                for &s in f.pcm.iter() {
                    pcm_f32.push(s as f32 / 8_388_608.0);
                }
            }
        }
        eprintln!("dts labels: {labels}");

        assert!(frames > 0, "no frames decoded from the DTS stream");
        assert!(channels >= 6, "expected >= 5.1, got {channels} channels");
        let peak = pcm_f32.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 1e-4, "decoded DTS bed is silent (peak={peak})");
        eprintln!("dts decoded {frames} frames, {channels} ch, peak={peak:.4}");
        let _ = std::fs::write(
            "/tmp/dts/harletty_bridge.f32",
            pcm_f32
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
    }

    fn object_pcm_frame(core: CorePcmFrame, object_channels: Vec<Vec<f32>>) -> ObjectPcmFrame {
        ObjectPcmFrame {
            core,
            object_active: vec![true; object_channels.len()],
            object_channels,
            joc: JocPayload {
                downmix_config: 0,
                channel_count: 0,
                object_count: 0,
                gain: 1.0,
                sequence_counter: 0,
                objects: Vec::new(),
            },
            oamd_payloads: Vec::new(),
        }
    }

    fn minimal_access_unit_info() -> AccessUnitInfo {
        AccessUnitInfo {
            frame_size: 0,
            bitstream_id: 16,
            frame_type: FrameType::Independent,
            substreamid: 0,
            sample_rate: 48_000,
            num_blocks: 6,
            channel_mode: 2,
            channels: 2,
            fullband_channels: 2,
            lfe_on: false,
            dependent_channel_map: None,
            dialogue_normalization: [-31, -31],
            heavy_compression_exists: [false, false],
            heavy_compression_word: [0, 0],
            addbsi_present: false,
            extension_type_a: false,
            complexity_index_type_a: 0,
            mixing_metadata_present: false,
            informational_metadata_present: false,
            addbsi_bytes: Vec::new(),
            body_start_bit_offset: 0,
            audio_frame: AudioFrameInfo {
                exponent_strategies_embedded: true,
                adaptive_hybrid_transform_enabled: false,
                coupling_uses_aht: false,
                channel_uses_aht: Vec::new(),
                lfe_uses_aht: false,
                snr_offset_strategy: 0,
                transient_processing_enabled: false,
                block_switching_enabled: false,
                dithering_enabled: false,
                bit_allocation_mode_enabled: false,
                frame_gain_syntax_enabled: false,
                delta_bit_allocation_enabled: false,
                skip_field_syntax_enabled: false,
                spectral_extension_attenuation_enabled: false,
                coupling_strategy_updates: Vec::new(),
                coupling_in_use: Vec::new(),
                coupling_exponent_strategy: Vec::new(),
                channel_exponent_strategy: Vec::new(),
                lfe_exponent_strategy: Vec::new(),
                converter_exponent_strategy_present: false,
                converter_exponent_strategy: Vec::new(),
                frame_csnr_offset: None,
                frame_fsnr_offset: None,
                transient_processors: Vec::new(),
                spectral_extension_attenuation: Vec::new(),
                block_start_info_present: false,
                block_start_info_bit_len: 0,
                block_payload_start_bit_offset: 0,
            },
            block_drc: Vec::new(),
            skip_fields: Vec::new(),
            trailing_aux_data: Vec::new(),
            aux_data: Vec::new(),
            aux_parse_status: AuxParseStatus::Disabled,
            emdf_source: EmdfSource::None,
            emdf_blocks: Vec::new(),
            emdf_block_count: 0,
            first_emdf_sync_offset: None,
        }
    }

    fn oamd_with_slots(bed: Vec<BedChannel>, dynamic_objects: usize) -> OamdPayload {
        OamdPayload {
            version: 0,
            object_count: bed.len() + dynamic_objects,
            alternate_object_present: false,
            element_count: 0,
            beds: bed.len(),
            bed_instances: usize::from(!bed.is_empty()),
            bed_or_isf_objects: bed.len(),
            dynamic_objects,
            isf_in_use: false,
            isf_index: None,
            bed_assignment: (!bed.is_empty()).then_some(bed).into_iter().collect(),
            elements: Vec::new(),
        }
    }

    #[test]
    fn lfe_only_program_maps_joc_slots_to_dynamic_objects() {
        let oamd = oamd_with_slots(vec![BedChannel::LowFrequencyEffects], 2);
        assert_eq!(
            eac3_oamd_slot_layout(&oamd, 2),
            Some(Eac3JocPresentationLayout {
                has_lfe: true,
                slots: vec![
                    Eac3JocSlot::Dynamic { id: 10 },
                    Eac3JocSlot::Dynamic { id: 11 },
                ],
            })
        );
    }

    #[test]
    fn fixed_bed_and_lfe2_remain_leading_joc_slots() {
        let oamd = oamd_with_slots(
            vec![
                BedChannel::Center,
                BedChannel::LowFrequencyEffects,
                BedChannel::LowFrequencyEffects2,
            ],
            1,
        );
        assert_eq!(
            eac3_oamd_slot_layout(&oamd, 3),
            Some(Eac3JocPresentationLayout {
                has_lfe: true,
                slots: vec![
                    Eac3JocSlot::Bed(BedChannel::Center),
                    Eac3JocSlot::Bed(BedChannel::LowFrequencyEffects2),
                    Eac3JocSlot::Dynamic { id: 10 },
                ],
            })
        );
    }

    #[test]
    fn eac3_object_output_keeps_only_lfe_bed_before_dynamic_objects() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![
                BedChannel::FrontLeft,
                BedChannel::FrontRight,
                BedChannel::Center,
                BedChannel::SurroundLeft,
                BedChannel::SurroundRight,
            ],
            fullband_channels: vec![
                vec![0.11, 0.12],
                vec![0.21, 0.22],
                vec![0.31, 0.32],
                vec![0.41, 0.42],
                vec![0.51, 0.52],
            ],
            lfe_channel: Some(vec![0.61, 0.62]),
        };
        let frame = object_pcm_frame(core.clone(), vec![vec![0.71, 0.72], vec![0.81, 0.82]]);
        let slots = vec![
            Eac3JocSlot::Dynamic { id: 10 },
            Eac3JocSlot::Dynamic { id: 11 },
        ];

        let (pcm, labels) = build_eac3_object_output_pcm_and_labels(&frame, &slots);

        assert_eq!(
            labels.as_slice(),
            &[
                RChannelLabel::LFE,
                RChannelLabel::Object,
                RChannelLabel::Object
            ]
        );
        assert_eq!(
            pcm.as_slice(),
            &[
                float_to_pcm_i32(0.61),
                float_to_pcm_i32(0.71),
                float_to_pcm_i32(0.81),
                float_to_pcm_i32(0.62),
                float_to_pcm_i32(0.72),
                float_to_pcm_i32(0.82),
            ]
        );
    }

    #[test]
    fn eac3_object_output_labels_fixed_joc_slot_before_dynamic_object() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft, BedChannel::FrontRight],
            fullband_channels: vec![vec![0.11], vec![0.21]],
            lfe_channel: Some(vec![0.31]),
        };
        let frame = object_pcm_frame(core, vec![vec![0.41], vec![0.51]]);
        let slots = vec![
            Eac3JocSlot::Bed(BedChannel::Center),
            Eac3JocSlot::Dynamic { id: 10 },
        ];

        let (pcm, labels) = build_eac3_object_output_pcm_and_labels(&frame, &slots);

        assert_eq!(
            labels.as_slice(),
            &[RChannelLabel::LFE, RChannelLabel::C, RChannelLabel::Object]
        );
        assert_eq!(
            pcm.as_slice(),
            &[
                float_to_pcm_i32(0.31),
                float_to_pcm_i32(0.41),
                float_to_pcm_i32(0.51),
            ]
        );
    }

    #[test]
    fn eac3_object_output_omits_bed_when_lfe_is_absent() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft, BedChannel::FrontRight],
            fullband_channels: vec![vec![0.11], vec![0.21]],
            lfe_channel: None,
        };
        let frame = object_pcm_frame(core.clone(), vec![vec![0.31], vec![0.41]]);
        let slots = vec![
            Eac3JocSlot::Dynamic { id: 10 },
            Eac3JocSlot::Dynamic { id: 11 },
        ];

        let (pcm, labels) = build_eac3_object_output_pcm_and_labels(&frame, &slots);

        assert_eq!(
            labels.as_slice(),
            &[RChannelLabel::Object, RChannelLabel::Object]
        );
        assert_eq!(
            pcm.as_slice(),
            &[float_to_pcm_i32(0.31), float_to_pcm_i32(0.41)]
        );
    }

    /// A 7.1.4 bed-only OAMD payload: 12 bed channels, no dynamic objects.
    fn bed_only_oamd(bed: Vec<BedChannel>) -> OamdPayload {
        let object_count = bed.len();
        OamdPayload {
            version: 0,
            object_count,
            alternate_object_present: false,
            element_count: 0,
            beds: object_count,
            bed_instances: 1,
            bed_or_isf_objects: object_count,
            dynamic_objects: 0,
            isf_in_use: false,
            isf_index: None,
            bed_assignment: vec![bed],
            elements: Vec::new(),
        }
    }

    fn full_714_bed() -> Vec<BedChannel> {
        vec![
            BedChannel::FrontLeft,
            BedChannel::FrontRight,
            BedChannel::Center,
            BedChannel::LowFrequencyEffects,
            BedChannel::SurroundLeft,
            BedChannel::SurroundRight,
            BedChannel::RearLeft,
            BedChannel::RearRight,
            BedChannel::TopFrontLeft,
            BedChannel::TopFrontRight,
            BedChannel::TopRearLeft,
            BedChannel::TopRearRight,
        ]
    }

    #[test]
    fn pure_bed_labels_skip_lfe_and_match_reconstructed_objects() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft, BedChannel::FrontRight],
            fullband_channels: vec![vec![0.0], vec![0.0]],
            lfe_channel: Some(vec![0.0]),
        };
        // 11 reconstructed channels = the 12-channel 7.1.4 bed minus the LFE
        // (carried by the core, not reconstructed by JOC).
        let mut frame = object_pcm_frame(core, vec![vec![0.0]; 11]);
        frame.oamd_payloads = vec![(bed_only_oamd(full_714_bed()), None)];

        let slots = eac3_joc_slot_layout(&frame).expect("pure bed slot layout");
        assert!(slots.has_lfe);
        assert_eq!(
            slots.slots,
            vec![
                Eac3JocSlot::Bed(BedChannel::FrontLeft),
                Eac3JocSlot::Bed(BedChannel::FrontRight),
                Eac3JocSlot::Bed(BedChannel::Center),
                Eac3JocSlot::Bed(BedChannel::SurroundLeft),
                Eac3JocSlot::Bed(BedChannel::SurroundRight),
                Eac3JocSlot::Bed(BedChannel::RearLeft),
                Eac3JocSlot::Bed(BedChannel::RearRight),
                Eac3JocSlot::Bed(BedChannel::TopFrontLeft),
                Eac3JocSlot::Bed(BedChannel::TopFrontRight),
                Eac3JocSlot::Bed(BedChannel::TopRearLeft),
                Eac3JocSlot::Bed(BedChannel::TopRearRight),
            ]
        );

        // The output channels carry their real bed labels (LFE first, then the
        // reconstructed bed channels) so the engine spatialises them.
        let (_pcm, channel_labels) = build_eac3_object_output_pcm_and_labels(&frame, &slots.slots);
        assert_eq!(
            channel_labels.as_slice(),
            &[
                RChannelLabel::LFE,
                RChannelLabel::L,
                RChannelLabel::R,
                RChannelLabel::C,
                RChannelLabel::Ls,
                RChannelLabel::Rs,
                RChannelLabel::Lb,
                RChannelLabel::Rb,
                RChannelLabel::Tfl,
                RChannelLabel::Tfr,
                RChannelLabel::Tbl,
                RChannelLabel::Tbr,
            ]
        );
    }

    #[test]
    fn pure_bed_detection_declines_dynamic_objects_and_count_mismatch() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft, BedChannel::FrontRight],
            fullband_channels: vec![vec![0.0], vec![0.0]],
            lfe_channel: Some(vec![0.0]),
        };

        // Has a dynamic object → not a pure bed.
        let mut dynamic = object_pcm_frame(core.clone(), vec![vec![0.0]; 11]);
        let mut oamd = bed_only_oamd(full_714_bed());
        oamd.dynamic_objects = 1;
        dynamic.oamd_payloads = vec![(oamd, None)];
        assert!(eac3_joc_slot_layout(&dynamic).is_none());

        // Bed assignment does not line up with the reconstructed channel count.
        let mut mismatch = object_pcm_frame(core, vec![vec![0.0]; 5]);
        mismatch.oamd_payloads = vec![(bed_only_oamd(full_714_bed()), None)];
        assert!(eac3_joc_slot_layout(&mismatch).is_none());
    }

    #[test]
    fn legacy_ac3_not_eac3_frame_can_advance_as_silence() {
        let mut bridge = AtmosBridge::new(false);
        let mut frame = vec![0u8; 1234];
        frame[..8].copy_from_slice(&[0x0B, 0x77, 0x2A, 0x68, 0x22, 0x30, 0xE1, 0xFF]);

        let decoded = process_eac3_frame(&mut bridge, &frame).expect("legacy AC-3 silence frame");

        assert_eq!(bridge.eac3_diag_stats.total_frames, 1);
        assert_eq!(bridge.eac3_diag_stats.legacy_ac3_frames, 1);
        assert_eq!(decoded.sampling_frequency, 48_000);
        assert_eq!(decoded.sample_count, LEGACY_AC3_SAMPLE_COUNT);
        assert_eq!(decoded.channel_count, LEGACY_AC3_CHANNEL_COUNT);
        assert_eq!(
            decoded.channel_labels.as_slice(),
            &[
                RChannelLabel::L,
                RChannelLabel::R,
                RChannelLabel::C,
                RChannelLabel::LFE,
                RChannelLabel::Ls,
                RChannelLabel::Rs,
            ]
        );
        assert_eq!(
            decoded.pcm.len(),
            LEGACY_AC3_SAMPLE_COUNT as usize * LEGACY_AC3_CHANNEL_COUNT as usize
        );
        assert!(decoded.pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn legacy_ac3_diag_extracts_header_shape() {
        let mut frame = vec![0u8; 2304];
        frame[..8].copy_from_slice(&[0x0B, 0x77, 0x2A, 0x68, 0x22, 0x30, 0xE1, 0xFF]);

        let info = legacy_ac3_info(&frame).expect("legacy AC-3 info");

        assert_eq!(info.bsid, 6);
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.frame_size, 2304);
        assert_eq!(info.channel_mode, 7);
        assert_eq!(info.channels, 6);
        assert!(info.lfe_on);
    }

    #[test]
    fn dependent_eac3_timing_uses_header_blocks() {
        let mut frame = vec![0u8; 2304];
        frame[..8].copy_from_slice(&[0x0B, 0x77, 0x44, 0x7F, 0x3A, 0x87, 0xFF, 0x31]);

        assert_eq!(frame_bsid(&frame), Some(16));
        assert_eq!(eac3_frame_timing(&frame), Some((48_000, 1536)));
    }

    #[test]
    fn standard_drc_uses_last_block_dynamic_range_word() {
        let mut bridge = AtmosBridge::new(false);
        bridge.drc_mode = crate::bridge::DrcMode::Standard;
        let mut info = minimal_access_unit_info();
        info.block_drc = vec![
            BlockDrcInfo {
                dynamic_range_exists: [true, false],
                dynamic_range_word: [0x00, 0x00],
            },
            BlockDrcInfo {
                dynamic_range_exists: [true, false],
                dynamic_range_word: [0x20, 0x00],
            },
        ];

        let (gain, ramp) = eac3_drc_params(&bridge, &info, 1536);

        assert!((gain - decode_ac3_dynamic_range_word(0x20)).abs() < 1e-6);
        assert_eq!(ramp, 1536);
    }

    #[test]
    fn heavy_drc_prefers_heavy_word_and_falls_back_to_standard_word() {
        let mut bridge = AtmosBridge::new(false);
        bridge.drc_mode = crate::bridge::DrcMode::Heavy;

        let mut heavy_info = minimal_access_unit_info();
        heavy_info.heavy_compression_exists = [true, false];
        heavy_info.heavy_compression_word = [0x20, 0x00];
        heavy_info.block_drc = vec![BlockDrcInfo {
            dynamic_range_exists: [true, false],
            dynamic_range_word: [0x40, 0x00],
        }];
        let (heavy_gain, _) = eac3_drc_params(&bridge, &heavy_info, 1536);
        assert!((heavy_gain - decode_ac3_heavy_range_word(0x20)).abs() < 1e-6);

        let mut fallback_info = minimal_access_unit_info();
        fallback_info.block_drc = vec![BlockDrcInfo {
            dynamic_range_exists: [true, false],
            dynamic_range_word: [0x40, 0x00],
        }];
        let (fallback_gain, _) = eac3_drc_params(&bridge, &fallback_info, 1536);
        assert!((fallback_gain - decode_ac3_dynamic_range_word(0x40)).abs() < 1e-6);
    }
}
