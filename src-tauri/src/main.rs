// 阻止 Windows release 模式下弹出额外的命令行窗口
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    subtrans_lib::run()
}
