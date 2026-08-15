//! 滚动结构化任务日志：每个高精度任务用 `task_id` 关联事件，
//! 路径脱敏（文件名 + 稳定哈希），沿用 5 MiB 轮转策略。

use std::path::PathBuf;

const LOG_ROTATE_BYTES: u64 = 5 * 1024 * 1024;

/// 生成随机任务 ID：`vocal-<16 hex>`，由 PID、纳秒时间戳与会话代次经
/// `RandomState` 哈希得到（每次任务独立随机，无需额外依赖）。
pub fn new_task_id(session_generation: u64) -> String {
    use std::hash::{BuildHasher, Hash, Hasher};
    let random = std::collections::hash_map::RandomState::new();
    let mut hasher = random.build_hasher();
    std::process::id().hash(&mut hasher);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .hash(&mut hasher);
    session_generation.hash(&mut hasher);
    format!("vocal-{:016x}", hasher.finish())
}

/// 滚动结构化任务日志：向 `<app_data_dir>/logs/subtrans.log` 追加 JSON 行。
pub struct TaskLogger {
    log_file: PathBuf,
    task_id: String,
}

impl TaskLogger {
    pub fn new(app_data_dir: PathBuf, task_id: impl Into<String>) -> Self {
        Self { log_file: app_data_dir.join("logs").join("subtrans.log"), task_id: task_id.into() }
    }

    /// 写入一行 `{ts, task_id, event, fields}`（fields 内路径先脱敏）。
    pub fn write(&self, event: &str, fields: &serde_json::Value) -> Result<(), String> {
        let dir = self.log_file.parent().ok_or_else(|| "日志路径无效".to_string())?;
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        // 5 MiB 轮转：旧日志保留一份 subtrans.log.1，排障时不丢历史现场
        if self.log_file.metadata().map(|m| m.len() > LOG_ROTATE_BYTES).unwrap_or(false) {
            let old = self.log_file.with_extension("log.1");
            let _ = std::fs::remove_file(&old);
            let _ = std::fs::rename(&self.log_file, &old);
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let record = serde_json::json!({
            "ts": ts,
            "task_id": self.task_id,
            "event": event,
            "fields": redact_paths(fields),
        });
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)
            .map_err(|e| e.to_string())?;
        writeln!(file, "{record}").map_err(|e| e.to_string())
    }
}

/// 递归脱敏：含路径分隔符的字符串 → `文件名#<16 hex 稳定哈希>`；
/// 不记录完整用户目录、环境变量、API Key 或字幕文本。
fn redact_paths(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => {
            if s.contains('\\') || s.contains('/') {
                let base = s.rsplit(['\\', '/']).next().unwrap_or(s.as_str());
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                s.hash(&mut hasher);
                serde_json::Value::String(format!("{base}#{:016x}", hasher.finish()))
            } else {
                serde_json::Value::String(s.clone())
            }
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(redact_paths).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter().map(|(k, val)| (k.clone(), redact_paths(val))).collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_test_dir(label: &str) -> std::path::PathBuf {
        let stamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("subtrans-{label}-{}-{stamp}", std::process::id()))
    }

    #[test]
    fn task_logger_rotates_and_redacts_user_directories() {
        let dir = unique_test_dir("task-log");
        let logger = TaskLogger::new(dir.clone(), "task-123");
        logger
            .write(
                "chunk_start",
                &serde_json::json!({
                    "video": r#"C:\Users\Alice\Movies\film.mp4"#,
                    "chunk_index": 2
                }),
            )
            .unwrap();
        let text = std::fs::read_to_string(dir.join("logs/subtrans.log")).unwrap();
        assert!(text.contains("film.mp4"));
        assert!(!text.contains(r#"C:\Users\Alice"#));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn task_id_is_random_and_prefixed() {
        let a = new_task_id(1);
        let b = new_task_id(1);
        assert!(a.starts_with("vocal-"));
        assert_eq!(a.len(), "vocal-".len() + 16);
        assert_ne!(a, b, "同代次两次生成的 task_id 不应相同");
    }
}
