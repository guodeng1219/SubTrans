//! 高精度人声分离的分片计划：全部边界使用整数采样帧运算，
//! 杜绝浮点累计误差。总帧数来自规范 PCM WAV 头（Task 7 传入），
//! 不对压缩容器做任何时长推算。

/// 分离/识别人声轨的固定采样率。
pub const SEPARATION_SAMPLE_RATE: u32 = 44_100;
/// 识别基础分片秒数（沿用现有 120 秒识别配置）。
pub const RECOGNITION_CHUNK_SEC: u32 = 120;
/// 分离核心长度：两个识别分片，保证分离边界与识别主窗口边界对齐。
pub const DEFAULT_CORE_SEC: u32 = RECOGNITION_CHUNK_SEC * 2;
/// 内存超限后的一次性重试核心长度。
pub const RETRY_CORE_SEC: u32 = RECOGNITION_CHUNK_SEC;
/// 核心左右两侧的上下文保护区秒数。
pub const GUARD_SEC: u32 = 5;

/// 单个分离片段的采样帧范围。
///
/// - `core_*`：进入最终输出的范围（首尾相接、无重叠无空洞）。
/// - `extract_*`：送入模型的窗口（含保护区，相邻窗口允许重叠）。
/// - `trim_start_frame`：分离结果中核心范围起点的本地采样偏移
///   （`core_start_frame - extract_start_frame`）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocalChunk {
    pub index: usize,
    pub core_start_frame: u64,
    pub core_frames: u64,
    pub extract_start_frame: u64,
    pub extract_frames: u64,
    pub trim_start_frame: u64,
}

impl VocalChunk {
    pub fn core_end_frame(&self) -> u64 {
        self.core_start_frame + self.core_frames
    }
}

/// 一次分离尝试的完整计划。
#[derive(Clone, Debug)]
pub struct VocalChunkPlan {
    pub total_frames: u64,
    pub core_sec: u32,
    pub chunks: Vec<VocalChunk>,
}

/// 仅测试/常量换算用：生产总帧数来自规范 WAV 头，整数 atrim 不需要秒→帧换算。
/// `#[cfg(test)]` 使 lib 目标（不含 cfg(test)）不编译它，避免 `clippy --all-targets`
/// 的死代码门禁误报。
#[cfg(test)]
pub fn frames(seconds: f64) -> u64 {
    (seconds * SEPARATION_SAMPLE_RATE as f64).round() as u64
}

/// 校验并生成覆盖 `[0, total_frames)` 的完整分片计划。
///
/// 校验：总帧数非零、核心长度非零、保护区不大于核心长度。
/// 核心范围首尾相接无重叠；仅提取窗口（含保护区）可以重叠。
pub fn build_chunk_plan(
    total_frames: u64,
    core_sec: u32,
    guard_sec: u32,
) -> Result<VocalChunkPlan, String> {
    if total_frames == 0 {
        return Err("总帧数必须大于 0".into());
    }
    if core_sec == 0 {
        return Err("核心长度必须大于 0".into());
    }
    if guard_sec > core_sec {
        return Err("保护区长度不得大于核心长度".into());
    }
    let core_frames = frames_of(core_sec);
    let guard_frames = frames_of(guard_sec);
    Ok(build_range_plan(total_frames, core_sec, 0, core_frames, guard_frames))
}

/// 生成覆盖 `[completed_until_frame, total_frames)` 的重试计划。
/// 已完成的前缀不被重复，未完成范围无空洞。
pub fn build_remaining_plan(
    total_frames: u64,
    completed_until_frame: u64,
    core_sec: u32,
    guard_sec: u32,
) -> Result<VocalChunkPlan, String> {
    if total_frames == 0 {
        return Err("总帧数必须大于 0".into());
    }
    if core_sec == 0 {
        return Err("核心长度必须大于 0".into());
    }
    if guard_sec > core_sec {
        return Err("保护区长度不得大于核心长度".into());
    }
    if completed_until_frame > total_frames {
        return Err("已完成位置超出源范围".into());
    }
    let core_frames = frames_of(core_sec);
    let guard_frames = frames_of(guard_sec);
    Ok(build_range_plan(total_frames, core_sec, completed_until_frame, core_frames, guard_frames))
}

/// 通用范围计划构造：核心从 `start_frame` 起按 `core_frames` 步进直至覆盖 `total_frames`。
fn build_range_plan(
    total_frames: u64,
    core_sec: u32,
    start_frame: u64,
    core_frames: u64,
    guard_frames: u64,
) -> VocalChunkPlan {
    let mut chunks = Vec::new();
    let mut core_start = start_frame;
    let mut index = 0usize;
    while core_start < total_frames {
        let core_end = total_frames.min(core_start + core_frames);
        let extract_start = core_start.saturating_sub(guard_frames);
        let extract_end = total_frames.min(core_end + guard_frames);
        chunks.push(VocalChunk {
            index,
            core_start_frame: core_start,
            core_frames: core_end - core_start,
            extract_start_frame: extract_start,
            extract_frames: extract_end - extract_start,
            trim_start_frame: core_start - extract_start,
        });
        index += 1;
        core_start = core_end;
    }
    VocalChunkPlan { total_frames, core_sec, chunks }
}

/// 秒 → 采样帧（checked 整数运算，避免溢出与浮点）。
fn frames_of(sec: u32) -> u64 {
    u64::from(sec) * u64::from(SEPARATION_SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_chunk_has_two_five_second_guards() {
        let plan = build_chunk_plan(frames(5657.258685), DEFAULT_CORE_SEC, GUARD_SEC).unwrap();
        let c = &plan.chunks[1];
        assert_eq!(c.core_start_frame, frames(240.0));
        assert_eq!(c.core_frames, frames(240.0));
        assert_eq!(c.extract_start_frame, frames(235.0));
        assert_eq!(c.extract_frames, frames(250.0));
        assert_eq!(c.trim_start_frame, frames(5.0));
    }

    #[test]
    fn first_and_last_chunks_clip_guards_to_source() {
        let plan = build_chunk_plan(frames(481.0), DEFAULT_CORE_SEC, GUARD_SEC).unwrap();
        assert_eq!(plan.chunks[0].extract_start_frame, 0);
        assert_eq!(plan.chunks[0].trim_start_frame, 0);
        let last = plan.chunks.last().unwrap();
        assert_eq!(last.core_start_frame, frames(480.0));
        assert_eq!(last.core_frames, frames(1.0));
        assert_eq!(last.extract_frames, frames(6.0));
    }

    #[test]
    fn retry_plan_preserves_completed_prefix_without_gap() {
        let retry =
            build_remaining_plan(frames(5657.258685), frames(480.0), RETRY_CORE_SEC, GUARD_SEC)
                .unwrap();
        assert_eq!(retry.chunks[0].core_start_frame, frames(480.0));
        assert_eq!(retry.chunks[0].core_frames, frames(120.0));
        assert_eq!(retry.chunks.last().unwrap().core_end_frame(), retry.total_frames);
        for pair in retry.chunks.windows(2) {
            assert_eq!(pair[0].core_end_frame(), pair[1].core_start_frame);
        }
    }

    #[test]
    fn invalid_frame_ranges_are_rejected() {
        assert!(build_chunk_plan(0, DEFAULT_CORE_SEC, GUARD_SEC).is_err());
        assert!(build_chunk_plan(frames(10.0), 0, GUARD_SEC).is_err());
        assert!(build_chunk_plan(frames(10.0), GUARD_SEC, DEFAULT_CORE_SEC).is_err());
        assert!(
            build_remaining_plan(frames(10.0), frames(11.0), RETRY_CORE_SEC, GUARD_SEC).is_err()
        );
    }

    #[test]
    fn cores_are_contiguous_and_cover_the_whole_source() {
        let plan = build_chunk_plan(frames(5657.258685), DEFAULT_CORE_SEC, GUARD_SEC).unwrap();
        assert!(!plan.chunks.is_empty());
        assert_eq!(plan.chunks[0].core_start_frame, 0);
        assert_eq!(plan.chunks.last().unwrap().core_end_frame(), plan.total_frames);
        for pair in plan.chunks.windows(2) {
            assert_eq!(pair[0].core_end_frame(), pair[1].core_start_frame);
        }
    }
}
