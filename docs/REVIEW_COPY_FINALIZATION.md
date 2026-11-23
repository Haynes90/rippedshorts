# Review/copy finalization correction

Closes the release gap between verified preview review, complete copy and the
existing Schedule Master handoff. No renderer, scheduler, schema or dependency changes.

## Observed production state (2026-11-23)

The current request has six current editorial/framing approvals: Shorts 1, 2, 3,
5 and Highlights 1, 2. Short 4 is undecided. All four Short captions and the edited
Highlight 2 description are saved. The untouched Highlight 1 description is empty,
but its draft was incorrectly labeled approved after Save. Batch creators are
confirmed. Final copy approval and scheduling request timestamps are absent; the
workflow projection reports no scheduler acknowledgment. Logs show successful
clip/copy POSTs and no finish POST in the inspected current-deployment window.
The owner's phone/browser unsaved text cannot be inferred from server records.

## Causes and changes

- Finished-preview approval creates blank copy placeholders. Highlight approvals
  now invoke the existing `_generate_schedule_copy` after the vote commits. Missing
  existing copy can use the same regenerate route in missing-only mode. The existing
  immutable analysis cache binds generation to request, asset, media revision,
  clip-specific transcript, brand, prompt version and model. Completed drafts are
  reused. Owner-edited copy cannot be regenerated; a concurrent edit wins.
- Empty text saves durably as pending, never approved. Highlight title and complete
  description are separately required. Existing single-editor hashtag transport
  and Short caption process remain intact. The existing metadata prompt explicitly
  requests clip-specific SEO titles, descriptions and relevant keywords.
- Saved-state blockers previously disabled the button that saves unsaved edits.
  The page now evaluates current editor completeness, retaining collaborator,
  selection and hold gates; the server independently validates persisted copy.
- The finish endpoint previously checked all playable siblings, unlike the page's
  approved-only check. Both now check only current approved assets. Unselected and
  rejected clips do not need copy approval. Finish saves only approved copy cards.
- Final approval freezes the reviewed snapshot transactionally. Notifications and
  learning projection cannot turn a committed approval into an error or prevent
  attempting the existing handoff. Interrupted HTTP responses reconcile through
  authoritative review state, without another finish POST. A receipt is acknowledged
  only when persisted; a pending request is never reported as scheduled.

## Sentence review, without changing approved media

The seven retained passages and current feedback were inspected. No saved boundary
adjustment feedback exists. Short 1 retains the molasses rationale, dilution and
payoff; Short 2 retains its compost proportion and rationale; Short 3 contains the
repotting demonstration and close; Short 5 retains the humic-acid explanation.
Short 4 opens abruptly and refers to a prior question, but is unselected. Highlight
1 has a streaming aside at its opening. Highlight 2 contains a plant recap between
related watering answers. These are contextual/editorial concerns, not evidence
that the encoded audio truncates a word. Existing adjustment/revision controls are
the appropriate path if the owner requests changes. No boundaries, approvals,
renders or receipts were changed by this correction.

## Acceptance and remaining live gate

Focused tests cover durable empty drafts, exact text/hashtags, stale edits,
clip-specific generation and cache reuse, owner-write precedence, current-editor
gates, undecided siblings, final approval, duplicate prevention and receipt/uncertain
handoff reconciliation. Existing creator/media workflow tests exercise the original
handoff with isolated provider and outbox adapters. No live scheduling transaction
is executed by tests.

Deployment requires authorization of the tested PR head. After deployment, prepare
only the untouched missing highlight copy, preserve edited copy, let the owner
review/finalize it and explicitly finish scheduling. Read back the existing outbox
and authoritative Schedule Master deliveries, destinations and dates before claiming
end-to-end completion. Global publishing and Agape remain disabled.
