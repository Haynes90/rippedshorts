# Repository coding-agent review guidance


## Architecture and maintenance review: realtime updates and single canonical implementation

When changing data flows or existing code, enforce both checks during design, code review, and testing:

1. **Realtime before polling:** Before adding `setInterval`, recursive `setTimeout`, or any fixed-interval refresh against an application-owned data source, inspect whether the backend offers events, subscriptions, WebSockets, or server-sent events. Prefer an authenticated push/subscription implementation when supported, with lifecycle cleanup, reconnection/backoff, and initial-state synchronization. Polling is acceptable for externally imposed REST-only interfaces, scheduled reconciliation, reliability fallback, and other cases without a practical push mechanism; document the reason, bounded interval, and shutdown conditions. Do not replace intentional watchdog/reconciliation jobs just because they poll.
2. **One canonical implementation:** Update an existing function/module in place instead of leaving `v2`, `_new`, `-copy`, `_updated`, or similar duplicate implementations beside the original. Use version control for rollback. On necessary migrations, identify every caller, migrate references, remove obsolete code, and test compatibility. Keep deliberately versioned public API contracts and documented staged migrations.
3. **Review gate:** In every relevant PR, search changed files for new polling loops and version-suffixed duplicate files/functions. Record whether the new behavior is justified, and verify no duplicate live implementation or unintended concurrent refresh remains. Where feasible add a focused regression test proving the subscription or single canonical path. Do not claim automated enforcement unless CI actually performs the checks.
4. **Safety:** Changes to this guidance do not authorize merges, deployment, publishing, scheduler recovery, or changing existing approval/hold states. Keep production gates, data preservation, and current project-specific instructions intact.
