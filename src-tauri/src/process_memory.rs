//! 人声分离 worker 进程树的内存守卫。
//!
//! - 预算按物理内存分档（4/8/16 GiB）。
//! - 每 2 秒采样进程树（Windows 用 Private Usage 求和，其余平台用 RSS 求和）。
//! - 软限：85% 警告一次、100% 或系统可用 <1 GiB 终止。
//! - Windows Job Object 硬限（软预算的 115%）只作突发兜底；
//!   关闭 Job handle 前可查询 `PeakJobMemoryUsed` 参与失败分类。

pub const GIB: u64 = 1024 * 1024 * 1024;

/// 一次进程树内存采样。
#[derive(Clone, Copy, Debug)]
pub struct MemorySnapshot {
    pub working_set_bytes: u64,
    pub private_bytes: u64,
    pub available_system_bytes: u64,
}

/// 软限决策。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDecision {
    Continue,
    Warn,
    Exceeded,
}

/// worker 失败后的分类：是否走一次性 120 秒降片重试。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerFailureClass {
    MemoryLimit,
    Other,
}

/// 单个进程的内存行（进程树求和与测试用）。
#[derive(Clone, Copy, Debug)]
pub struct ProcessUsageRow {
    pub pid: u32,
    pub parent: Option<u32>,
    pub private_bytes: u64,
    pub working_set_bytes: u64,
}

/// 按物理内存分档返回预算（纯函数，便于测试）。
pub fn memory_budget_bytes(total_physical_bytes: u64) -> u64 {
    if total_physical_bytes < 16 * GIB {
        4 * GIB
    } else if total_physical_bytes < 32 * GIB {
        8 * GIB
    } else {
        16 * GIB
    }
}

/// 生产预算入口：debug 构建允许 `SUBTRANS_VOCAL_MEMORY_BUDGET_BYTES` 覆盖
/// （Task 9 的重试测试用）；release 构建不读取该环境变量。
pub fn effective_memory_budget_bytes(total_physical_bytes: u64) -> u64 {
    #[cfg(debug_assertions)]
    {
        if let Ok(v) = std::env::var("SUBTRANS_VOCAL_MEMORY_BUDGET_BYTES") {
            if let Ok(n) = v.trim().parse::<u64>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    memory_budget_bytes(total_physical_bytes)
}

/// worker 失败三证据分类（纯函数）：
/// 最后有效采样峰值达到预算、Job 峰值达到预算、或 Python 明确报告 OOM → 内存超限。
pub fn classify_worker_failure(
    last_sample_peak_bytes: u64,
    job_peak_bytes: Option<u64>,
    budget_bytes: u64,
    explicit_oom: bool,
) -> WorkerFailureClass {
    if explicit_oom
        || last_sample_peak_bytes >= budget_bytes
        || job_peak_bytes.is_some_and(|p| p >= budget_bytes)
    {
        WorkerFailureClass::MemoryLimit
    } else {
        WorkerFailureClass::Other
    }
}

/// 求和 root 及其递归后代的 private_bytes（进程树内、不含无关进程）。
pub fn sum_process_tree_bytes(root_pid: u32, rows: &[ProcessUsageRow]) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root_pid];
    let mut visited = std::collections::HashSet::new();
    while let Some(pid) = stack.pop() {
        if !visited.insert(pid) {
            continue;
        }
        if let Some(row) = rows.iter().find(|r| r.pid == pid) {
            total = total.saturating_add(row.private_bytes);
            for child in rows.iter().filter(|r| r.parent == Some(pid)) {
                stack.push(child.pid);
            }
        }
    }
    total
}

/// 跨平台进程树采样器。
pub struct ProcessMemoryProbe {
    system: sysinfo::System,
    root_pid: u32,
}

impl ProcessMemoryProbe {
    pub fn new(root_pid: u32) -> Self {
        Self { system: sysinfo::System::new(), root_pid }
    }

    /// 物理内存总量（预算分档依据）。
    pub fn total_physical_bytes(&mut self) -> u64 {
        self.system.refresh_memory();
        self.system.total_memory()
    }

    /// 采样当前进程树：root 已消失时返回 Err（调用方视作 worker 退出，
    /// 不得推断为零内存采样）。
    pub fn sample(&mut self) -> Result<MemorySnapshot, String> {
        self.system.refresh_memory();
        let available = self.system.available_memory();
        let rows = self.collect_rows()?;
        if rows.is_empty() {
            return Err("worker 进程已消失".into());
        }
        let working_set_bytes = rows.iter().map(|r| r.working_set_bytes).sum();
        // 用树求和辅助函数（生产消费方，避免其成为测试专用死代码）
        let private_bytes = sum_process_tree_bytes(self.root_pid, &rows);
        Ok(MemorySnapshot { working_set_bytes, private_bytes, available_system_bytes: available })
    }

    /// 从 root 出发递归收集后代进程行（仅后代，不含祖先）。
    fn collect_rows(&mut self) -> Result<Vec<ProcessUsageRow>, String> {
        self.system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        let processes = self.system.processes();
        let root = sysinfo::Pid::from_u32(self.root_pid);
        if !processes.contains_key(&root) {
            return Err("worker 进程已消失".into());
        }
        let mut rows = Vec::new();
        let mut stack = vec![self.root_pid];
        let mut visited = std::collections::HashSet::new();
        while let Some(pid) = stack.pop() {
            if !visited.insert(pid) {
                continue;
            }
            let pid_obj = sysinfo::Pid::from_u32(pid);
            let Some(proc) = processes.get(&pid_obj) else { continue };
            let working_set_bytes = proc.memory();
            let private_bytes = windows_private_bytes(pid).unwrap_or(working_set_bytes);
            rows.push(ProcessUsageRow {
                pid,
                parent: proc.parent().map(|p| p.as_u32()),
                private_bytes,
                working_set_bytes,
            });
            for child in processes.values() {
                if child.parent() == Some(pid_obj) {
                    stack.push(child.pid().as_u32());
                }
            }
        }
        Ok(rows)
    }
}

/// Windows：查询单个进程的 Private Usage（提交内存）；失败返回 None，调用方回退 RSS。
#[cfg(target_os = "windows")]
fn windows_private_bytes(pid: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        let ok = GetProcessMemoryInfo(handle, &mut counters as *mut _ as *mut _, size);
        CloseHandle(handle);
        if ok != 0 {
            Some(counters.PrivateUsage as u64)
        } else {
            None
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn windows_private_bytes(_pid: u32) -> Option<u64> {
    None
}

/// 软限守卫：记录峰值、85% 只警告一次、100% 或系统可用 <1 GiB 触发超限。
pub struct MemoryGuard {
    budget_bytes: u64,
    warned: bool,
    peak_private_bytes: u64,
}

impl MemoryGuard {
    pub fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes, warned: false, peak_private_bytes: 0 }
    }

    pub fn observe(&mut self, sample: MemorySnapshot) -> MemoryDecision {
        self.peak_private_bytes = self.peak_private_bytes.max(sample.private_bytes);
        if sample.private_bytes >= self.budget_bytes || sample.available_system_bytes < GIB {
            return MemoryDecision::Exceeded;
        }
        let warn_threshold = self.budget_bytes.saturating_mul(85) / 100;
        if sample.private_bytes >= warn_threshold && !self.warned {
            self.warned = true;
            return MemoryDecision::Warn;
        }
        MemoryDecision::Continue
    }

    pub fn peak_private_bytes(&self) -> u64 {
        self.peak_private_bytes
    }
}

// ── Windows Job Object 硬限 ──

#[cfg(target_os = "windows")]
pub struct JobGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

// HANDLE 是 *mut c_void（原始指针默认 !Send/!Sync）。JobGuard 仅被创建它的任务
// 持有与销毁，跨 await 存活需要 Send 才能让 Tauri 命令 future 可迁移。
#[cfg(target_os = "windows")]
unsafe impl Send for JobGuard {}
#[cfg(target_os = "windows")]
unsafe impl Sync for JobGuard {}

#[cfg(target_os = "windows")]
impl JobGuard {
    /// 把 Python PID 绑进 Job：内存硬限 = 软预算 × 115%（只作突发兜底），
    /// 并启用 KILL_ON_JOB_CLOSE 保证句柄关闭时整树清理。
    pub fn assign(pid: u32, soft_budget_bytes: u64) -> Result<Self, String> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
        };
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err("CreateJobObjectW 失败".into());
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            info.JobMemoryLimit =
                soft_budget_bytes.saturating_mul(115).saturating_div(100) as usize;
            let info_size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
            let set_ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                info_size,
            );
            if set_ok == 0 {
                CloseHandle(job);
                return Err("SetInformationJobObject 失败".into());
            }
            let proc = OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                pid,
            );
            if proc.is_null() {
                CloseHandle(job);
                return Err(format!("OpenProcess 失败（pid {pid}）"));
            }
            let assign_ok = AssignProcessToJobObject(job, proc);
            CloseHandle(proc); // 进程句柄立即释放，Job 句柄保留
            if assign_ok == 0 {
                CloseHandle(job);
                return Err(format!("AssignProcessToJobObject 失败（pid {pid}）"));
            }
            Ok(Self { handle: job })
        }
    }

    /// 关闭 handle 前查询 `PeakJobMemoryUsed`（此时 Job 记账仍有效）。
    pub fn peak_job_memory_bytes(&self) -> Result<u64, String> {
        use windows_sys::Win32::System::JobObjects::{
            JobObjectExtendedLimitInformation, QueryInformationJobObject,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        };
        unsafe {
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            let info_size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
            let ok = QueryInformationJobObject(
                self.handle,
                JobObjectExtendedLimitInformation,
                &mut info as *mut _ as *mut _,
                info_size,
                std::ptr::null_mut(),
            );
            if ok == 0 {
                return Err("QueryInformationJobObject 失败".into());
            }
            Ok(info.PeakJobMemoryUsed as u64)
        }
    }

    /// 终止 Job 内的整个进程树。
    pub fn terminate(&self, exit_code: u32) {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;
        unsafe {
            let _ = TerminateJobObject(self.handle, exit_code);
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for JobGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub struct JobGuard;

#[cfg(not(target_os = "windows"))]
impl JobGuard {
    /// 非 Windows：无硬限，仅保留周期采样；接口保持一致。
    pub fn assign(_pid: u32, _soft_budget_bytes: u64) -> Result<Self, String> {
        Ok(Self)
    }
    pub fn peak_job_memory_bytes(&self) -> Result<u64, String> {
        Ok(0)
    }
    pub fn terminate(&self, _exit_code: u32) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(private_bytes: u64, available_system_bytes: u64) -> MemorySnapshot {
        MemorySnapshot { working_set_bytes: private_bytes, private_bytes, available_system_bytes }
    }

    #[test]
    fn budget_uses_four_eight_sixteen_gib_tiers() {
        assert_eq!(memory_budget_bytes(8 * GIB), 4 * GIB);
        assert_eq!(memory_budget_bytes(16 * GIB), 8 * GIB);
        assert_eq!(memory_budget_bytes(31 * GIB), 8 * GIB);
        assert_eq!(memory_budget_bytes(32 * GIB), 16 * GIB);
        assert_eq!(memory_budget_bytes(64 * GIB), 16 * GIB);
    }

    #[test]
    fn guard_warns_once_then_exceeds_at_budget() {
        let mut guard = MemoryGuard::new(8 * GIB);
        assert_eq!(guard.observe(snapshot(6 * GIB, 3 * GIB)), MemoryDecision::Continue);
        assert_eq!(guard.observe(snapshot(7 * GIB, 3 * GIB)), MemoryDecision::Warn);
        assert_eq!(guard.observe(snapshot(7 * GIB, 3 * GIB)), MemoryDecision::Continue);
        assert_eq!(guard.observe(snapshot(8 * GIB, 3 * GIB)), MemoryDecision::Exceeded);
    }

    #[test]
    fn one_gib_system_floor_also_exceeds() {
        let mut guard = MemoryGuard::new(16 * GIB);
        assert_eq!(guard.observe(snapshot(2 * GIB, GIB - 1)), MemoryDecision::Exceeded);
    }

    #[test]
    fn process_tree_sum_includes_descendants_only() {
        let rows = vec![
            ProcessUsageRow { pid: 10, parent: None, private_bytes: 1, working_set_bytes: 1 },
            ProcessUsageRow { pid: 11, parent: Some(10), private_bytes: 2, working_set_bytes: 2 },
            ProcessUsageRow { pid: 12, parent: Some(11), private_bytes: 3, working_set_bytes: 3 },
            ProcessUsageRow { pid: 99, parent: None, private_bytes: 9, working_set_bytes: 9 },
        ];
        assert_eq!(sum_process_tree_bytes(10, &rows), 6);
    }

    #[test]
    fn worker_failure_uses_last_sample_job_peak_or_explicit_oom() {
        let budget = 8 * GIB;
        assert_eq!(
            classify_worker_failure(7 * GIB, Some(9 * GIB), budget, false),
            WorkerFailureClass::MemoryLimit
        );
        assert_eq!(
            classify_worker_failure(7 * GIB, None, budget, true),
            WorkerFailureClass::MemoryLimit
        );
        assert_eq!(
            classify_worker_failure(7 * GIB, None, budget, false),
            WorkerFailureClass::Other
        );
    }

    #[test]
    fn guard_tracks_peak_private_bytes() {
        let mut guard = MemoryGuard::new(8 * GIB);
        guard.observe(snapshot(5 * GIB, 16 * GIB));
        guard.observe(snapshot(3 * GIB, 16 * GIB));
        assert_eq!(guard.peak_private_bytes(), 5 * GIB);
    }
}
