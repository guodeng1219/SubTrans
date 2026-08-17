// 启动流程的会话归属守卫（纯函数，供 main.js 与 Node 测试共用）。
//
// 关键不变量：捕获必须在**任何异步等待之前**完成——startProcess 里存在
// cancel 等待与最长 15 秒的元数据等待，若等待之后才读取 state.session，
// 等待期间打开新视频/打开项目/再次点击开始都会让旧调用恢复后读到新会话号，
// 把新会话误当成自己的会话（快速双击时两个流程甚至会捕获同一会话号）。
// 恢复后的每一次过期检查都必须对照这份捕获值，而不是重读当前 state。

/**
 * 在任何异步等待之前捕获启动流程的归属：
 * 会话令牌立即 +1（新一轮识别使旧分片过期）并记录视频路径。
 * @param {{session: number, videoPath: string}} state
 * @returns {{session: number, videoPath: string}}
 */
export function captureStartSession(state) {
  return {
    session: ++state.session,
    videoPath: state.videoPath,
  };
}

/**
 * 启动流程是否已过期：会话或视频路径与捕获时不一致即放弃。
 * 过期流程不得再调用分离、修改 stream 或启动流水线。
 * @param {{session: number, videoPath: string}} captured 捕获值（await 之前取得）
 * @param {{session: number, videoPath: string}} state 当前状态
 * @returns {boolean}
 */
export function startFlowExpired(captured, state) {
  return state.session !== captured.session || state.videoPath !== captured.videoPath;
}
