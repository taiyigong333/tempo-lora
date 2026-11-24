#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-default}"
RESTART="${2:-}"
if [[ -n "$RESTART" && "$RESTART" != --restart ]]; then
    echo "第二个参数支持 --restart" >&2
    exit 2
fi
case "$PROFILE" in
    default) CONFIG="$ROOT/trained_model/deployment.json" ;;
    rtc_paper) CONFIG="$ROOT/trained_model/deployment_rtc_paper.json" ;;
    step4000) CONFIG="$ROOT/trained_model/deployment_step4000.json" ;;
    hist90_direct_step2000) CONFIG="$ROOT/trained_model/deployment_hist90_direct_step2000.json" ;;
    hist90_direct_step5000) CONFIG="$ROOT/trained_model/deployment_hist90_direct_step5000.json" ;;
    *) echo "支持的配置为 default、rtc_paper、step4000 、hist90_direct_step2000 或 hist90_direct_step5000" >&2; exit 2 ;;
esac
SESSION=tempo_full_inference
# 锁覆盖检查、退出和新进程创建，避免两个客户端并发切换模型。
mkdir -p "$ROOT/trained_model/runtime" "$ROOT/trained_model/logs"
exec 9>"$ROOT/trained_model/runtime/service_switch.lock"
flock -w 5 9
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    PANE_PID="$(tmux display-message -p -t "$SESSION:0.0" '#{pane_pid}')"
    RUNNING_COMMAND="$(ps -p "$PANE_PID" -o args=)"
    if [[ "$RUNNING_COMMAND" != *"openpi.tempo_full.inference"* || "$RUNNING_COMMAND" != *"$ROOT/"* ]]; then
        echo "会话中的进程与本项目入口不匹配，请核查 $SESSION" >&2
        exit 1
    fi
    if [[ "$RUNNING_COMMAND" == *"--config $CONFIG"* && "$RESTART" != --restart ]]; then
        echo "配置已运行 $PROFILE"
        exit 0
    fi
    if [[ -n "$(ss -Htn state established '( sport = :8002 )')" ]]; then
        echo "8002端口仍有客户端连接，请结束当前推理后切换模型" >&2
        exit 1
    fi
    tmux send-keys -t "$SESSION" C-c
    for attempt in {1..40}; do
        if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
            break
        fi
        sleep 0.2
    done
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "原服务仍在退出，请查看日志后重试" >&2
        exit 1
    fi
fi
# 将完整命令作为单个tmux参数传入，路径按bash规则引用。
printf -v LAUNCH 'exec bash %q --config %q >> %q 2>&1' \
    "$ROOT/trained_model/scripts/serve_wired.sh" "$CONFIG" "$ROOT/trained_model/logs/serve_$PROFILE.log"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$LAUNCH" 9>&-
echo "已启动配置 $PROFILE，等待日志中的预热完成与8002端口就绪"
