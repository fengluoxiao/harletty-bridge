// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::metadata::{JocObject, JocObjectData, JocPayload};
use super::pcm::CorePcmFrame;
use super::qmf::{QMF_SUBBANDS, QmfSubbands, QuadratureMirrorFilterBank};
use super::syncframe::ParseError;
use crate::BedChannel;

const JOC_INPUT_5_X: [BedChannel; 5] = [
    BedChannel::FrontLeft,
    BedChannel::FrontRight,
    BedChannel::Center,
    BedChannel::SurroundLeft,
    BedChannel::SurroundRight,
];

const JOC_INPUT_7_X: [BedChannel; 7] = [
    BedChannel::FrontLeft,
    BedChannel::FrontRight,
    BedChannel::Center,
    BedChannel::SurroundLeft,
    BedChannel::SurroundRight,
    BedChannel::RearLeft,
    BedChannel::RearRight,
];

const JOC_INPUT_5_X_2: [BedChannel; 7] = [
    BedChannel::FrontLeft,
    BedChannel::FrontRight,
    BedChannel::Center,
    BedChannel::SurroundLeft,
    BedChannel::SurroundRight,
    BedChannel::TopFrontLeft,
    BedChannel::TopFrontRight,
];

/// Observable analysis+synthesis latency of the normative 64-band complex QMF.
/// Object PCM and its core/LFE companions must share this delay.
pub const JOC_QMF_LATENCY_SAMPLES: usize = 577;

const JOC_PARAMETER_BAND_BOUNDARIES: [&[u8]; 8] = [
    &[0],
    &[0, 3, 14],
    &[0, 1, 3, 9, 23],
    &[0, 1, 2, 4, 8, 14, 23],
    &[0, 1, 2, 3, 5, 7, 9, 14, 23],
    &[0, 1, 2, 3, 4, 6, 8, 11, 14, 18, 23, 35],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 23, 35],
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 23, 26, 30, 35, 41, 48,
    ],
];

pub type JocSubbandMatrix = Vec<[f32; 64]>;
pub type JocTimeslotMatrices = Vec<JocSubbandMatrix>;
pub type JocObjectMatrices = Vec<JocTimeslotMatrices>;

type SubbandMatrix = JocSubbandMatrix;
type TimeslotMatrices = JocTimeslotMatrices;

#[derive(Debug)]
struct SampleDelay {
    history: [f32; JOC_QMF_LATENCY_SAMPLES],
    cursor: usize,
}

impl Default for SampleDelay {
    fn default() -> Self {
        Self {
            history: [0.0; JOC_QMF_LATENCY_SAMPLES],
            cursor: 0,
        }
    }
}

impl SampleDelay {
    fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            std::mem::swap(sample, &mut self.history[self.cursor]);
            self.cursor += 1;
            if self.cursor == JOC_QMF_LATENCY_SAMPLES {
                self.cursor = 0;
            }
        }
    }
}

#[derive(Debug, Default)]
struct CoreDelayState {
    sample_rate: Option<u32>,
    layout: Vec<BedChannel>,
    fullband: Vec<SampleDelay>,
    lfe: Option<SampleDelay>,
}

impl CoreDelayState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn process(&mut self, core: &mut CorePcmFrame) {
        let lfe_present = core.lfe_channel.is_some();
        if self.sample_rate != Some(core.sample_rate)
            || self.layout != core.fullband_channel_order
            || self.fullband.len() != core.fullband_channels.len()
            || self.lfe.is_some() != lfe_present
        {
            self.sample_rate = Some(core.sample_rate);
            self.layout.clone_from(&core.fullband_channel_order);
            self.fullband = (0..core.fullband_channels.len())
                .map(|_| SampleDelay::default())
                .collect();
            self.lfe = lfe_present.then(SampleDelay::default);
        }

        for (delay, channel) in self.fullband.iter_mut().zip(&mut core.fullband_channels) {
            delay.process(channel);
        }
        if let (Some(delay), Some(channel)) = (&mut self.lfe, &mut core.lfe_channel) {
            delay.process(channel);
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct JocObjectDecoderState {
    prev_matrix: Vec<SubbandMatrix>,
    mix_matrix: Vec<[SubbandMatrix; 2]>,
    timeslot_offsets: Vec<[u8; 2]>,
    forward_qmf: Vec<QuadratureMirrorFilterBank>,
    inverse_qmf: Vec<QuadratureMirrorFilterBank>,
    inverse_history: Vec<bool>,
    analysis: Vec<QmfSubbands>,
    core_delay: CoreDelayState,
    last_frame_matrices: JocObjectMatrices,
    last_sequence_counter: Option<u16>,
    last_configuration: Option<(u8, usize, usize, u32, usize)>,
}

impl JocObjectDecoderState {
    pub fn reset(&mut self) {
        self.clear_reconstruction_state();
        self.last_sequence_counter = None;
        self.last_configuration = None;
    }

    fn clear_reconstruction_state(&mut self) {
        self.prev_matrix.clear();
        self.mix_matrix.clear();
        self.timeslot_offsets.clear();
        self.forward_qmf.clear();
        self.inverse_qmf.clear();
        self.inverse_history.clear();
        self.analysis.clear();
        self.core_delay.reset();
        self.last_frame_matrices.clear();
    }

    pub fn decode_frame(
        &mut self,
        core: &CorePcmFrame,
        joc: &JocPayload,
    ) -> Result<Vec<Vec<f32>>, ParseError> {
        let mut objects = Vec::new();
        self.decode_frame_into(core, joc, &mut objects)?;
        Ok(objects)
    }

    pub fn decode_frame_into(
        &mut self,
        core: &CorePcmFrame,
        joc: &JocPayload,
        objects: &mut Vec<Vec<f32>>,
    ) -> Result<(), ParseError> {
        let input_order = joc_input_order(joc.downmix_config)?;
        if joc.channel_count != input_order.len() {
            return Err(ParseError::InvalidHeader("joc-channel-count"));
        }
        let samples = core.samples_per_channel();
        if samples == 0 || !samples.is_multiple_of(QMF_SUBBANDS) {
            return Err(ParseError::InvalidHeader("joc-frame-samples"));
        }

        let configuration = (
            joc.downmix_config,
            joc.channel_count,
            joc.object_count,
            core.sample_rate,
            samples,
        );
        let sequence_continues = self
            .last_sequence_counter
            .is_some_and(|previous| joc_sequence_is_contiguous(previous, joc.sequence_counter));
        if !sequence_continues || self.last_configuration != Some(configuration) {
            // A zero/discontinuous sequence counter marks the first frame after a
            // splice. Matrix interpolation and both QMF banks must start without
            // history from the preceding segment (ETSI TS 103 420 §6.3.3.3).
            self.clear_reconstruction_state();
        }
        self.last_sequence_counter = Some(joc.sequence_counter);
        self.last_configuration = Some(configuration);

        let input_indices = map_input_channel_indices(core, input_order)?;
        let timeslots = samples / QMF_SUBBANDS;
        self.reconfigure(joc.channel_count, joc.object_count);
        self.build_frame_matrices(joc, timeslots)?;
        prepare_object_output_buffers(objects, joc.object_count, samples);
        self.analysis.resize(joc.channel_count, QmfSubbands::zero());
        let analysis = &mut self.analysis[..joc.channel_count];

        for timeslot in 0..timeslots {
            let sample_offset = timeslot * QMF_SUBBANDS;
            for (joc_channel, core_channel) in input_indices.iter().enumerate() {
                analysis[joc_channel] = self.forward_qmf[joc_channel].process_forward(
                    &core.fullband_channels[*core_channel]
                        [sample_offset..sample_offset + QMF_SUBBANDS],
                );
            }

            for object_index in 0..joc.object_count {
                let active = joc.objects[object_index].active;
                let matrix = &self.last_frame_matrices[object_index][timeslot];
                let output =
                    &mut objects[object_index][sample_offset..sample_offset + QMF_SUBBANDS];
                if !active && !self.inverse_history[object_index] {
                    output.fill(0.0);
                    continue;
                }

                let mut mixed = QmfSubbands::zero();
                for (channel_index, gains) in matrix.iter().enumerate() {
                    for subband in 0..QMF_SUBBANDS {
                        mixed.real[subband] +=
                            analysis[channel_index].real[subband] * gains[subband];
                        mixed.imaginary[subband] +=
                            analysis[channel_index].imaginary[subband] * gains[subband];
                    }
                }

                self.inverse_qmf[object_index].process_inverse(&mixed, output);
                if joc.gain != 1.0 {
                    for sample in output.iter_mut() {
                        *sample *= joc.gain;
                    }
                }
                if active {
                    self.inverse_history[object_index] = true;
                }
            }
        }

        Ok(())
    }

    pub fn delay_core_frame(&mut self, core: &mut CorePcmFrame) {
        self.core_delay.process(core);
    }

    pub fn last_frame_matrices(&self) -> &JocObjectMatrices {
        &self.last_frame_matrices
    }

    fn reconfigure(&mut self, channel_count: usize, object_count: usize) {
        self.forward_qmf
            .resize_with(channel_count, QuadratureMirrorFilterBank::new);
        if self.forward_qmf.len() > channel_count {
            self.forward_qmf.truncate(channel_count);
        }

        self.inverse_qmf
            .resize_with(object_count, QuadratureMirrorFilterBank::new);
        if self.inverse_qmf.len() > object_count {
            self.inverse_qmf.truncate(object_count);
        }

        self.inverse_history.resize(object_count, false);
        if self.inverse_history.len() > object_count {
            self.inverse_history.truncate(object_count);
        }
        self.analysis.resize(channel_count, QmfSubbands::zero());
        if self.analysis.len() > channel_count {
            self.analysis.truncate(channel_count);
        }

        self.prev_matrix
            .resize_with(object_count, || vec![[0.0; QMF_SUBBANDS]; channel_count]);
        if self.prev_matrix.len() > object_count {
            self.prev_matrix.truncate(object_count);
        }
        for object in &mut self.prev_matrix {
            object.resize(channel_count, [0.0; QMF_SUBBANDS]);
            if object.len() > channel_count {
                object.truncate(channel_count);
            }
        }

        self.mix_matrix.resize_with(object_count, || {
            [
                vec![[0.0; QMF_SUBBANDS]; channel_count],
                vec![[0.0; QMF_SUBBANDS]; channel_count],
            ]
        });
        if self.mix_matrix.len() > object_count {
            self.mix_matrix.truncate(object_count);
        }
        for object in &mut self.mix_matrix {
            for slot in object {
                slot.resize(channel_count, [0.0; QMF_SUBBANDS]);
                if slot.len() > channel_count {
                    slot.truncate(channel_count);
                }
            }
        }

        self.timeslot_offsets.resize(object_count, [0; 2]);
        if self.timeslot_offsets.len() > object_count {
            self.timeslot_offsets.truncate(object_count);
        }
    }

    fn build_frame_matrices(
        &mut self,
        joc: &JocPayload,
        timeslots: usize,
    ) -> Result<(), ParseError> {
        ensure_object_matrix_storage(
            &mut self.last_frame_matrices,
            joc.object_count,
            timeslots,
            joc.channel_count,
        );
        for object_index in 0..joc.object_count {
            let object = joc
                .objects
                .get(object_index)
                .ok_or(ParseError::InvalidHeader("joc-object"))?;
            let prev_matrix = &mut self.prev_matrix[object_index];
            let mix_matrix = &mut self.mix_matrix[object_index];
            let timeslot_offsets = &mut self.timeslot_offsets[object_index];
            decode_parameter_points(mix_matrix, timeslot_offsets, object, joc.channel_count)?;
            build_object_timeslots(
                prev_matrix,
                mix_matrix,
                *timeslot_offsets,
                object,
                timeslots,
                &mut self.last_frame_matrices[object_index],
            )?;
        }
        Ok(())
    }
}

fn joc_sequence_is_contiguous(previous: u16, current: u16) -> bool {
    if current == 0 {
        return false;
    }
    let expected = if previous == 1023 { 1 } else { previous + 1 };
    current == expected
}

fn prepare_object_output_buffers(output: &mut Vec<Vec<f32>>, object_count: usize, samples: usize) {
    output.resize_with(object_count, Vec::new);
    if output.len() > object_count {
        output.truncate(object_count);
    }
    for channel in output.iter_mut() {
        if channel.len() != samples {
            channel.resize(samples, 0.0);
        }
    }
}

fn ensure_object_matrix_storage(
    storage: &mut JocObjectMatrices,
    object_count: usize,
    timeslots: usize,
    channel_count: usize,
) {
    storage.resize_with(object_count, Vec::new);
    if storage.len() > object_count {
        storage.truncate(object_count);
    }
    for object in storage.iter_mut() {
        object.resize_with(timeslots, Vec::new);
        if object.len() > timeslots {
            object.truncate(timeslots);
        }
        for timeslot in object.iter_mut() {
            timeslot.resize(channel_count, [0.0; QMF_SUBBANDS]);
            if timeslot.len() > channel_count {
                timeslot.truncate(channel_count);
            }
        }
    }
}

fn build_object_timeslots(
    prev_matrix: &mut SubbandMatrix,
    mix_matrix: &[SubbandMatrix; 2],
    timeslot_offsets: [u8; 2],
    object: &JocObject,
    timeslots: usize,
    output: &mut TimeslotMatrices,
) -> Result<(), ParseError> {
    if !object.active {
        zero_timeslot_matrices(output);
        return Ok(());
    }

    let bands_index = object
        .bands_index
        .ok_or(ParseError::InvalidHeader("joc_num_bands_idx"))? as usize;
    let mapping = expanded_parameter_band_mapping(bands_index)?;
    if object.data_points == 1 {
        if object.steep_slope {
            let split = usize::from(timeslot_offsets[0]).min(timeslots);
            for (timeslot, matrix) in output.iter_mut().enumerate() {
                if timeslot < split {
                    copy_matrix(matrix, prev_matrix);
                } else {
                    copy_matrix_mapped(matrix, &mix_matrix[0], mapping);
                }
            }
        } else {
            for (timeslot, matrix) in output.iter_mut().enumerate() {
                let lerp = (timeslot + 1) as f32 / timeslots as f32;
                lerp_matrix_to_mapped(matrix, prev_matrix, &mix_matrix[0], mapping, lerp);
            }
        }
    } else if object.steep_slope {
        let first_split = usize::from(timeslot_offsets[0]).min(timeslots);
        let second_split = usize::from(timeslot_offsets[1]).min(timeslots);
        for (timeslot, matrix) in output.iter_mut().enumerate() {
            if timeslot < first_split {
                copy_matrix(matrix, prev_matrix);
            } else if timeslot < second_split {
                copy_matrix_mapped(matrix, &mix_matrix[0], mapping);
            } else {
                copy_matrix_mapped(matrix, &mix_matrix[1], mapping);
            }
        }
    } else {
        let first_half = (timeslots >> 1).max(1);
        for (timeslot, matrix) in output.iter_mut().enumerate() {
            let timeslot_index = timeslot + 1;
            if timeslot_index <= first_half {
                let lerp = timeslot_index as f32 / first_half as f32;
                lerp_matrix_to_mapped(matrix, prev_matrix, &mix_matrix[0], mapping, lerp);
            } else {
                let second_len = (timeslots - first_half).max(1);
                let lerp = (timeslot_index - first_half) as f32 / second_len as f32;
                lerp_matrix_mapped(matrix, &mix_matrix[0], &mix_matrix[1], mapping, lerp);
            }
        }
    }

    update_prev_matrix(prev_matrix, &mix_matrix[object.data_points - 1], mapping);
    Ok(())
}

fn zero_timeslot_matrices(output: &mut TimeslotMatrices) {
    for timeslot in output.iter_mut() {
        for channel in timeslot.iter_mut() {
            *channel = [0.0; QMF_SUBBANDS];
        }
    }
}

fn joc_input_order(downmix_config: u8) -> Result<&'static [BedChannel], ParseError> {
    match downmix_config {
        0 | 3 => Ok(&JOC_INPUT_5_X),
        1 => Ok(&JOC_INPUT_7_X),
        2 | 4 => Ok(&JOC_INPUT_5_X_2),
        _ => Err(ParseError::UnsupportedFeature("joc_dmx_config_idx")),
    }
}

fn map_input_channel_indices(
    core: &CorePcmFrame,
    input_order: &[BedChannel],
) -> Result<Vec<usize>, ParseError> {
    input_order
        .iter()
        .map(|channel| {
            core.fullband_channel_order
                .iter()
                .position(|candidate| candidate == channel)
                .ok_or(ParseError::UnsupportedFeature("joc-input-layout"))
        })
        .collect()
}

fn decode_parameter_points(
    mix_matrix: &mut [SubbandMatrix; 2],
    timeslot_offsets: &mut [u8; 2],
    object: &JocObject,
    channel_count: usize,
) -> Result<(), ParseError> {
    // `mix_matrix` and offsets are persistent decoder scratch. Clear both data
    // point slots every frame so a one-point object can never consume the stale
    // second point from a previous frame (ETSI TS 103 420 §6.6.5).
    *timeslot_offsets = [0; 2];
    for data_point in mix_matrix.iter_mut() {
        for channel in data_point.iter_mut() {
            channel.fill(0.0);
        }
    }
    if !object.active {
        return Ok(());
    }

    let quantization_table = object
        .quantization_table
        .ok_or(ParseError::InvalidHeader("joc_num_quant_idx"))?
        as usize;
    let bands = object.bands;
    let data_points = object.data_points;
    let data = object
        .data
        .as_ref()
        .ok_or(ParseError::InvalidHeader("joc-object-data"))?;

    if object.steep_slope {
        for (slot, offset) in object.timeslot_offsets.iter().copied().enumerate() {
            timeslot_offsets[slot] = offset;
        }
    }

    match data {
        JocObjectData::Dense { matrices } => {
            if matrices.len() != data_points {
                return Err(ParseError::InvalidHeader("joc_dense_points"));
            }
            let nquant = if quantization_table == 0 { 96 } else { 192 };
            let center = nquant / 2;
            let gain_step = 820.0f32 / (4096.0 * (1 + quantization_table) as f32);
            for (data_point, source) in matrices.iter().enumerate() {
                if source.len() != channel_count {
                    return Err(ParseError::InvalidHeader("joc_dense_channels"));
                }
                for (channel_index, source_channel) in source.iter().enumerate() {
                    if source_channel.len() != bands {
                        return Err(ParseError::InvalidHeader("joc_dense_bands"));
                    }
                    let channel = &mut mix_matrix[data_point][channel_index];
                    let mut current = 0usize;
                    for (band_index, value) in source_channel.iter().enumerate() {
                        current = if band_index == 0 {
                            (center + usize::from(*value)) % nquant
                        } else {
                            (current + usize::from(*value)) % nquant
                        };
                        channel[band_index] = (current as f32 - center as f32) * gain_step;
                    }
                }
            }
        }
        JocObjectData::Sparse { .. } => {
            // The available ETSI commentary identifies the differential
            // domains but does not unambiguously specify the reconstruction
            // order. Returning success with an all-zero matrix silently drops
            // active objects, so reject sparse payloads until the algorithm can
            // be verified against a normative/known vector.
            return Err(ParseError::UnsupportedFeature("joc-sparse"));
        }
    }
    Ok(())
}

fn expanded_parameter_band_mapping(
    bands_index: usize,
) -> Result<&'static [u8; QMF_SUBBANDS], ParseError> {
    static CACHE: OnceLock<[[u8; QMF_SUBBANDS]; 8]> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let mut expanded = [[0u8; QMF_SUBBANDS]; 8];
        for (index, boundaries) in JOC_PARAMETER_BAND_BOUNDARIES.iter().enumerate() {
            for subband in 0..QMF_SUBBANDS {
                let parameter_band = match boundaries.binary_search(&(subband as u8)) {
                    Ok(found) => found,
                    Err(insert) => insert.saturating_sub(1),
                };
                expanded[index][subband] = parameter_band as u8;
            }
        }
        expanded
    });
    cache
        .get(bands_index)
        .ok_or(ParseError::InvalidHeader("joc_num_bands_idx"))
}

fn copy_matrix(target: &mut SubbandMatrix, source: &[[f32; QMF_SUBBANDS]]) {
    for (dst, src) in target.iter_mut().zip(source.iter()) {
        *dst = *src;
    }
}

fn copy_matrix_mapped(
    target: &mut SubbandMatrix,
    source: &[[f32; QMF_SUBBANDS]],
    mapping: &[u8; QMF_SUBBANDS],
) {
    for (dst, src) in target.iter_mut().zip(source.iter()) {
        for subband in 0..QMF_SUBBANDS {
            dst[subband] = src[mapping[subband] as usize];
        }
    }
}

fn lerp_matrix_to_mapped(
    target: &mut SubbandMatrix,
    from: &[[f32; QMF_SUBBANDS]],
    to: &[[f32; QMF_SUBBANDS]],
    mapping: &[u8; QMF_SUBBANDS],
    lerp: f32,
) {
    for ((dst, src_from), src_to) in target.iter_mut().zip(from.iter()).zip(to.iter()) {
        for subband in 0..QMF_SUBBANDS {
            let parameter_band = mapping[subband] as usize;
            let target_value = src_to[parameter_band];
            dst[subband] = src_from[subband] + (target_value - src_from[subband]) * lerp;
        }
    }
}

fn lerp_matrix_mapped(
    target: &mut SubbandMatrix,
    from: &[[f32; QMF_SUBBANDS]],
    to: &[[f32; QMF_SUBBANDS]],
    mapping: &[u8; QMF_SUBBANDS],
    lerp: f32,
) {
    for ((dst, src_from), src_to) in target.iter_mut().zip(from.iter()).zip(to.iter()) {
        for subband in 0..QMF_SUBBANDS {
            let parameter_band = mapping[subband] as usize;
            let from_value = src_from[parameter_band];
            let to_value = src_to[parameter_band];
            dst[subband] = from_value + (to_value - from_value) * lerp;
        }
    }
}

fn update_prev_matrix(
    prev_matrix: &mut SubbandMatrix,
    source: &[[f32; QMF_SUBBANDS]],
    mapping: &[u8; QMF_SUBBANDS],
) {
    for (dst, src) in prev_matrix.iter_mut().zip(source.iter()) {
        for subband in 0..QMF_SUBBANDS {
            dst[subband] = src[mapping[subband] as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoreDelayState, JOC_QMF_LATENCY_SAMPLES, JocObjectDecoderState, SampleDelay,
        build_object_timeslots, decode_parameter_points, expanded_parameter_band_mapping,
        joc_input_order, joc_sequence_is_contiguous, map_input_channel_indices,
    };
    use crate::{BedChannel, CorePcmFrame, JocObject, JocObjectData, JocPayload};

    #[test]
    fn sample_delay_preserves_exact_latency_across_frames() {
        let mut delay = SampleDelay::default();
        let mut first: Vec<f32> = (1..=400).map(|sample| sample as f32).collect();
        let mut second: Vec<f32> = (401..=800).map(|sample| sample as f32).collect();

        delay.process(&mut first);
        delay.process(&mut second);

        let mut output = first;
        output.extend(second);
        assert!(
            output[..JOC_QMF_LATENCY_SAMPLES]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert_eq!(output[JOC_QMF_LATENCY_SAMPLES], 1.0);
        assert_eq!(output[799], (800 - JOC_QMF_LATENCY_SAMPLES) as f32);
    }

    #[test]
    fn core_delay_discards_history_when_sample_rate_changes() {
        let mut delay = CoreDelayState::default();
        let mut first = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft],
            fullband_channels: vec![vec![1.0; JOC_QMF_LATENCY_SAMPLES]],
            lfe_channel: None,
        };
        delay.process(&mut first);
        assert!(
            first.fullband_channels[0]
                .iter()
                .all(|sample| *sample == 0.0)
        );

        let mut switched = CorePcmFrame {
            sample_rate: 44_100,
            fullband_channel_order: vec![BedChannel::FrontLeft],
            fullband_channels: vec![vec![2.0; 64]],
            lfe_channel: None,
        };
        delay.process(&mut switched);

        assert!(
            switched.fullband_channels[0]
                .iter()
                .all(|sample| *sample == 0.0),
            "samples delayed at the old rate must not leak into the new configuration"
        );
    }

    #[test]
    fn joc_configuration_includes_sample_rate_and_frame_samples() {
        fn core(sample_rate: u32, samples: usize) -> CorePcmFrame {
            CorePcmFrame {
                sample_rate,
                fullband_channel_order: vec![
                    BedChannel::FrontLeft,
                    BedChannel::FrontRight,
                    BedChannel::Center,
                    BedChannel::SurroundLeft,
                    BedChannel::SurroundRight,
                ],
                fullband_channels: vec![vec![0.0; samples]; 5],
                lfe_channel: None,
            }
        }

        let mut joc = JocPayload {
            downmix_config: 0,
            channel_count: 5,
            object_count: 1,
            gain: 1.0,
            sequence_counter: 1,
            objects: vec![JocObject {
                active: false,
                bands_index: None,
                bands: 0,
                sparse_coded: false,
                quantization_table: None,
                steep_slope: false,
                data_points: 0,
                timeslot_offsets: Vec::new(),
                data: None,
            }],
        };
        let mut state = JocObjectDecoderState::default();
        state
            .decode_frame(&core(48_000, 64), &joc)
            .expect("initial frame");
        state.prev_matrix[0][0][0] = 9.0;

        joc.sequence_counter = 2;
        state
            .decode_frame(&core(44_100, 64), &joc)
            .expect("sample-rate switch");
        assert_eq!(state.last_configuration, Some((0, 5, 1, 44_100, 64)));
        assert_eq!(state.prev_matrix[0][0][0], 0.0);

        state.prev_matrix[0][0][0] = 7.0;
        joc.sequence_counter = 3;
        state
            .decode_frame(&core(44_100, 128), &joc)
            .expect("sample-count switch");
        assert_eq!(state.last_configuration, Some((0, 5, 1, 44_100, 128)));
        assert_eq!(state.prev_matrix[0][0][0], 0.0);
    }

    #[test]
    fn parameter_band_mapping_expands_last_band() {
        let mapping = expanded_parameter_band_mapping(7).expect("mapping");
        assert_eq!(mapping[0], 0);
        assert_eq!(mapping[48], 22);
        assert_eq!(mapping[63], 22);
    }

    #[test]
    fn acmod7_core_layout_reorders_to_joc_input() {
        let frame = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![
                BedChannel::FrontLeft,
                BedChannel::Center,
                BedChannel::FrontRight,
                BedChannel::SurroundLeft,
                BedChannel::SurroundRight,
            ],
            fullband_channels: vec![vec![0.0; 64]; 5],
            lfe_channel: Some(vec![0.0; 64]),
        };

        let indices =
            map_input_channel_indices(&frame, joc_input_order(3).unwrap()).expect("indices");
        assert_eq!(indices, vec![0, 2, 1, 3, 4]);
    }

    #[test]
    fn seven_channel_joc_rejects_missing_rear_inputs() {
        let frame = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![
                BedChannel::FrontLeft,
                BedChannel::FrontRight,
                BedChannel::Center,
                BedChannel::SurroundLeft,
                BedChannel::SurroundRight,
            ],
            fullband_channels: vec![vec![0.0; 64]; 5],
            lfe_channel: Some(vec![0.0; 64]),
        };

        assert!(map_input_channel_indices(&frame, joc_input_order(1).unwrap()).is_err());
    }

    #[test]
    fn five_x_plus_two_uses_top_front_inputs() {
        let frame = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![
                BedChannel::FrontLeft,
                BedChannel::FrontRight,
                BedChannel::Center,
                BedChannel::SurroundLeft,
                BedChannel::SurroundRight,
                BedChannel::TopFrontLeft,
                BedChannel::TopFrontRight,
            ],
            fullband_channels: vec![vec![0.0; 64]; 7],
            lfe_channel: Some(vec![0.0; 64]),
        };

        let indices =
            map_input_channel_indices(&frame, joc_input_order(4).unwrap()).expect("indices");
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn sparse_joc_is_rejected_instead_of_silently_zeroed() {
        let object = JocObject {
            active: true,
            bands_index: Some(1),
            bands: 3,
            sparse_coded: true,
            quantization_table: Some(0),
            steep_slope: false,
            data_points: 2,
            timeslot_offsets: Vec::new(),
            data: Some(JocObjectData::Sparse {
                channel_indices: vec![vec![0, 1, 0], vec![1, 0, 1]],
                vectors: vec![vec![4, 5, 6], vec![7, 8, 9]],
            }),
        };

        let mut mix_matrix = [vec![[1.0; 64]; 5], vec![[2.0; 64]; 5]];
        let mut timeslot_offsets = [0; 2];
        let error = decode_parameter_points(&mut mix_matrix, &mut timeslot_offsets, &object, 5)
            .expect_err("unverified sparse coding must not decode as silence");

        assert_eq!(
            error,
            crate::AccessUnitParseError::UnsupportedFeature("joc-sparse")
        );
    }

    #[test]
    fn steep_two_points_use_prev_then_both_current_points() {
        let object = JocObject {
            active: true,
            bands_index: Some(0),
            bands: 1,
            sparse_coded: false,
            quantization_table: Some(0),
            steep_slope: true,
            data_points: 2,
            timeslot_offsets: vec![1, 3],
            data: Some(JocObjectData::Dense {
                matrices: vec![vec![vec![0]], vec![vec![1]]],
            }),
        };

        let mut prev_matrix = vec![[1.0; 64]];
        let mut mix_matrix = [vec![[9.0; 64]], vec![[9.0; 64]]];
        let mut timeslot_offsets = [9; 2];
        decode_parameter_points(&mut mix_matrix, &mut timeslot_offsets, &object, 1)
            .expect("points");
        let mut output = vec![vec![[0.0; 64]]; 4];
        build_object_timeslots(
            &mut prev_matrix,
            &mix_matrix,
            timeslot_offsets,
            &object,
            4,
            &mut output,
        )
        .expect("timeslots");

        assert!(output[0][0].iter().all(|value| *value == 1.0));
        assert!(output[1][0].iter().all(|value| *value == 0.0));
        assert!(output[2][0].iter().all(|value| *value == 0.0));
        let gain_step = 820.0 / 4096.0;
        assert!(
            output[3][0]
                .iter()
                .all(|value| (*value - gain_step).abs() < 1e-6)
        );
        assert!(
            prev_matrix[0]
                .iter()
                .all(|value| (*value - gain_step).abs() < 1e-6)
        );
    }

    #[test]
    fn steep_single_point_clears_and_ignores_stale_second_slot() {
        let current = JocObject {
            active: true,
            bands_index: Some(0),
            bands: 1,
            sparse_coded: false,
            quantization_table: Some(0),
            steep_slope: true,
            data_points: 1,
            timeslot_offsets: vec![1],
            data: Some(JocObjectData::Dense {
                matrices: vec![vec![vec![0]]],
            }),
        };

        let mut mix_matrix = [vec![[7.0; 64]], vec![[5.0; 64]]];
        let mut timeslot_offsets = [8, 3];
        decode_parameter_points(&mut mix_matrix, &mut timeslot_offsets, &current, 1)
            .expect("current");

        assert_eq!(timeslot_offsets, [1, 0]);
        assert!(mix_matrix[1][0].iter().all(|value| *value == 0.0));

        let mut prev_matrix = vec![[1.0; 64]];
        let mut output = vec![vec![[0.0; 64]]; 4];
        build_object_timeslots(
            &mut prev_matrix,
            &mix_matrix,
            timeslot_offsets,
            &current,
            4,
            &mut output,
        )
        .expect("timeslots");

        assert!(output[0][0].iter().all(|value| *value == 1.0));
        assert!(
            output[1..]
                .iter()
                .all(|timeslot| timeslot[0].iter().all(|value| *value == 0.0))
        );
        assert!(prev_matrix[0].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn steep_parameter_bands_expand_to_all_qmf_subbands() {
        let object = JocObject {
            active: true,
            bands_index: Some(5),
            bands: 12,
            sparse_coded: false,
            quantization_table: Some(0),
            steep_slope: true,
            data_points: 1,
            timeslot_offsets: vec![0],
            data: Some(JocObjectData::Dense {
                matrices: vec![vec![vec![5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
            }),
        };

        let mut mix_matrix = [vec![[0.0; 64]], vec![[0.0; 64]]];
        let mut timeslot_offsets = [0; 2];
        decode_parameter_points(&mut mix_matrix, &mut timeslot_offsets, &object, 1)
            .expect("points");
        let mut prev_matrix = vec![[0.0; 64]];
        let mut output = vec![vec![[0.0; 64]]; 1];
        build_object_timeslots(
            &mut prev_matrix,
            &mix_matrix,
            timeslot_offsets,
            &object,
            1,
            &mut output,
        )
        .expect("timeslots");

        let expected = 5.0 * 820.0 / 4096.0;
        assert!(
            output[0][0]
                .iter()
                .all(|value| (*value - expected).abs() < 1e-6)
        );
    }

    #[test]
    fn dense_dequantization_uses_normative_steps() {
        let object = JocObject {
            active: true,
            bands_index: Some(0),
            bands: 1,
            sparse_coded: false,
            quantization_table: Some(1),
            steep_slope: false,
            data_points: 1,
            timeslot_offsets: Vec::new(),
            data: Some(JocObjectData::Dense {
                matrices: vec![vec![vec![1]]],
            }),
        };

        let mut mix_matrix = [vec![[0.0; 64]], vec![[0.0; 64]]];
        let mut timeslot_offsets = [0; 2];
        decode_parameter_points(&mut mix_matrix, &mut timeslot_offsets, &object, 1)
            .expect("points");

        let expected = 820.0 / (4096.0 * 2.0);
        assert!((mix_matrix[0][0][0] - expected).abs() < 1e-6);
    }

    #[test]
    fn joc_sequence_detects_splices_and_wraps() {
        assert!(joc_sequence_is_contiguous(1, 2));
        assert!(joc_sequence_is_contiguous(1023, 1));
        assert!(!joc_sequence_is_contiguous(12, 0));
        assert!(!joc_sequence_is_contiguous(12, 14));
        assert!(!joc_sequence_is_contiguous(1023, 2));
    }
}
