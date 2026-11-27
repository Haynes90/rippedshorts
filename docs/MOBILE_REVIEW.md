# Mobile review: focused refresh correction

Fresh YouTube intake passes through `schedule_route_gate`, installed by both web
entry points. That wrapper must persist `review_interface=website` too; changing
only the underlying Telegram intake does not affect its intercepted requests.
Explicit owner brand context is resolved against active Show Config rows, either
from the authenticated intake's `show_id` or a named `TCB:`/`TDOG:` message. Source
channel identity is never fabricated from the chosen destination. Unknown source
channels retain the existing route-choice gate. Route selection is intake routing,
not clip approval, and cannot authorize scheduling. The wrapper no longer generates
retired re-rip/reuse controls; existing processor source/history reuse remains.

The existing caption save, clip decision, player and approval APIs remain authoritative.
Approving a clip now appends its existing copy editor instead of reloading the page.
Polling appends newly available rendered cards without replacing active players or inputs.
Previous/next navigation and Approve & Next reuse existing saved decisions; navigation
advances only after copy saves and a current approval are acknowledged. Playback position
is retained locally per exact media revision. Other clips' unsaved text stays in place.

A passive status refresh must not upgrade unsaved copy to another editor's newer revision.
Version conflicts still require reconciliation; they never authorize overwriting saved text.

Scope excludes new issue lifecycles, ScheduleMaster endpoints, collaborator contracts,
rendering, processing, scheduling and publishing changes. Existing Request Adjustment,
Who's Featured, copy approval and final handoff remain unchanged.

Validation: isolated API tests and JavaScript interaction/parse tests. Physical Safari
and deployed browser acceptance are still required; isolated tests do not establish them.

## One authoritative review interface

Telegram accepts source submissions and retains essential processing/failure notifications.
Every request now uses the existing authenticated website for clip, framing, copy,
collaborator and scheduling approvals. Previously issued `rs:` review callbacks and
legacy caption replies cannot change decisions or schedule anything, including for
older requests without a website flag. Named operator recovery commands retain their
existing authorization and exact-revision safeguards.

The four Telegram review/control payload builders and their mutating callback branches
are removed. Compatibility presenters only invoke `review_navigation.notify_safely`.
One receipt-backed message per request is updated across stages and contains one
`Review Clips` URL button. An uncertain Telegram send remains held for reconciliation.
Shared render, copy-checkpoint, approval, retained-media and ScheduleMaster functions
remain in place for the website; no historic decisions are migrated or deleted.

## Existing reliability gaps corrected

Audio Master transcript import preserves measured words, sentence records and speaker
metadata. Approved sermon normalization rebases measured times only; evidence crossing
an approved edge is retained and marked for review rather than invented or silently
accepted. Feedback observation/edit capture supports every explicitly identified brand;
existing brand/format filters still isolate the learning used by future recommendations.
Local source caches become removable on the next cleanup sweep only after existing
all-dependent completion/explicit-closure checks pass. Temporary preview caches retain
their 48-hour completion clock, with active review and delivery dependencies protected.

Regression coverage includes inert historical controls, request ownership, one navigation
receipt, website saves/approvals/handoff, rich transcript rebasing, brand-specific ranking,
shared-source protection and disposable-file retention. Legacy tests asserting Telegram
review menus are replaced by these website-only behavioral regressions. Production
proof is recorded separately; tests alone do not establish a 90% operational success rate.
