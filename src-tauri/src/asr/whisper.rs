//! Whisper 引擎：whisper.cpp（通过 whisper-rs 绑定）。

use super::{AsrEngine, Segment};
use anyhow::{anyhow, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl AsrEngine for WhisperEngine {
    fn load(model_path: &Path) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| anyhow!("模型路径无效"))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| anyhow!("加载 Whisper 模型失败: {e:?}"))?;
        Ok(Self { ctx })
    }

    fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
        threads: i32,
        time_offset_sec: f64,
        vad_model_path: Option<&str>,
    ) -> Result<Vec<Segment>> {
        let mut state =
            self.ctx.create_state().map_err(|e| anyhow!("创建 Whisper state 失败: {e:?}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_n_threads(threads);
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some(language.unwrap_or("auto")));
        // 中文：用简体提示词偏置，减少繁体输出、轻微帮助同音字判断
        if language == Some("zh") {
            params.set_initial_prompt("以下是普通话的简体中文。");
        }
        // 通用抗幻觉参数：禁止用上一段文本做条件（避免幻觉传播），
        // 降低 no_speech 阈值（减少把歌声/轻语音判为“无语音”而跳过）。
        params.set_no_context(true); // 等价于 condition_on_previous_text=false
        params.set_no_speech_thold(0.4);
        // 仅在已经过人声分离时启用 VAD：
        // 分离后是纯人声，VAD 可安全过滤静音段幻觉；
        // 未分离的原始视频可能含音乐/歌声，VAD 会把唱歌当“非语音”跳过。
        if let Some(vad_path) = vad_model_path {
            params.set_vad_model_path(Some(vad_path));
            let mut vad_params = whisper_rs::WhisperVadParams::default();
            vad_params.set_threshold(0.25);
            vad_params.set_min_silence_duration(500);
            vad_params.set_min_speech_duration(100);
            vad_params.set_speech_pad(100);
            params.set_vad_params(vad_params);
            params.enable_vad(true);
        }

        state.full(params, audio).map_err(|e| anyhow!("Whisper 转写失败: {e:?}"))?;

        let mut segments = Vec::new();
        let mut idx = 0usize;
        for segment in state.as_iter() {
            let text = segment.to_string().trim().to_string();
            let text = super::clean_fillers(&text);
            if text.is_empty() {
                continue;
            }
            idx += 1;
            let t0 = segment.start_timestamp() as f64 / 100.0 + time_offset_sec;
            let t1 = segment.end_timestamp() as f64 / 100.0 + time_offset_sec;
            segments.push(Segment { index: idx, start: t0, end: t1, text });
        }
        Ok(segments)
    }
}
