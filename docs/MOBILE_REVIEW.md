# Mobile review: focused refresh correction

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
