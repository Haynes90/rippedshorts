# Chronological topics — focused correction and bounded recovery proposal

## Scope and verified defect

PR #84 is deployed at `461e8b7144376ac3d4496f20476a378646212fa1`.
The last authorized TDOG recovery passed episode reuse, four conversations,
conversation links and first-conversation topics, then held on second-conversation
topics: `Semantic hierarchy must follow depth-first parent order` (final durable
zero-based topic position 10). The failed response/parent key was not retained;
its particular unknown/forward/closed-parent/type subtype remains unknown.
The demonstrated architectural constraint is the validator's active ancestry
stack: chronological returns to a prior subject cannot be represented reliably
as temporal children. No provider references are repaired in this correction.

## Small correction, existing stages

- `hierarchical_intelligence.chronological_topics`: new topic results are
  chronological occurrences within the existing verified parent conversation.
  The existing strict semantic-start constructor derives inclusive ends; starts
  must be original integer indices, ordered, unique and cover the parent.
- Parent schema is null for new topic generation. The conversation is assigned
  as the temporal parent by the existing compiler. A focused question and answer,
  funny exchange or returning subject may be its own chronological topic.
- Only after all topic spans/identities validate are thematic `related_keys`
  checked against the complete key set. Forward/backward thematic links are
  allowed; unknown targets, wrong types, self-links and duplicates hold.
  This is a separate validation pass in the existing provider stage, not a new
  model call or analysis subsystem. Provider labels are never renamed or guessed.
- Diagnostics retain failure kind, topic position, hashed topic/rejected key,
  reference type, valid-key count/digest, schema version, prompt version/hash,
  immutable stage ID and verified parent bounds. Raw text/private keys are not
  logged. Existing cached failure persistence and watchdog holds remain intact.
- Aggregate recipe is `hierarchical-source-v3`; only topics use a new child recipe
  `chronological-topics-v2`. Episode, conversations and conversation relationships
  keep their exact versions/inputs. An exact-input PASSED legacy topic checkpoint
  is reused under its original ID. Legacy FAILED records/budgets are not reopened.
- `intelligence_workflow`: existing Shorts selector explicitly farms individual
  explanations, useful statements, jokes and Q&A throughout the transcript,
  expanding context/payoffs and permitting cross-topic/conversation boundaries.
  Existing approved rubric and semantic/acoustic validators remain authoritative.
  Highlight prompt emphasizes focused coherent topic arcs, minimum 180 seconds,
  complete context and no arbitrary maximum. Existing concurrent format analysis,
  sequential human review, learning ledger and provenance are retained.

No source acquisition, renderer, website, Telegram delivery, collaborator,
Schedule Master adapter or publishing configuration was changed. No extra
analysis stage, dependency, database or external infrastructure was added.

## Acceptance evidence and limits

New conversational fixture includes tomato seed-saving questions/answers,
fermentation advice, a humorous exchange across a conversation transition,
moisture/storage questions and a returning storage topic. It runs the actual
map/compiler, existing Shorts transcript/sentence validator, existing highlight
minimum-duration construction, both concurrent selector orchestrators, ranking,
feedback and candidate persistence. Only provider responses and external media
preflight are mocked. Both formats produce durable candidate IDs with map,
stage, rubric and scoped Sheets-projection feedback evidence; positive feedback
actually affects ranking. The Short spans a conversation transition. No owner
selection or final approval is manufactured.

Other new cases reject invalid temporal parents, unknown/self/duplicate/typed
links, gaps at the parent start, reversed/noninteger/out-of-bounds starts, duplicate
keys and provider-supplied end indices. Failure evidence persists and repeated
access stays held without further calls. A recovery fixture preserves the exact
legacy passed episode/conversations/links/first-topic rows and failed sibling,
executing only the missing corrected topic stage and moments.

These are isolated behavioral results, not real TDOG candidate generation,
semantic-quality approval, acoustic verification, finished rendering or production
acceptance. No production request or record was accessed or modified in this pass.

## Proposed single bounded recovery — NOT executed or authorized here

1. Review this draft PR, its final CI and full regression evidence; obtain
   separate merge/paused-deployment and one-recovery authorization.
2. Reverify disabled auto-deploy, backed-up DB/approvals/queue/receipts, retained
   persistent source media, unchanged credentials, one worker/no overlap and
   disabled global/Agape publishing. Deploy Ripped Shorts only after approval.
3. Verify request `8999063e-bbc7-4000-bba4-e2490ca27135`, YouTube `9mW0bhmUaw0`,
   video checksum/streams/duration and all 1,189 timed segments. Verify immutable
   IDs and canonical fingerprints for the passed episode, four-conversation
   boundary result, relationships and first-conversation topics. Stop on mismatch.
4. Record one distinct owner-bounded revision on the SAME request targeting
   aggregate `hierarchical-source-v3`. Preserve all old authorization receipts,
   failures and holds; do not reset legacy retry budgets. Dispatch exactly once.
   Expect cache hits for those four passed checkpoints; corrected remaining
   topics run once each, followed by existing complete-source moments and selectors.
5. Any deterministic failure retains successful checkpoints and safe diagnostics
   and holds; no watchdog restart or repeated identical calls. No new intake,
   YouTube download, transcription, old-selection changes or scheduling.
6. Measurable live milestone: actual 9:16 and 16:9 candidate results from the
   complete recording, with inspected topic quality and map/prompt/rubric/scoped
   feedback provenance. Record actual counts, rejected boundary evidence and
   explainable recommendations; no quota or forced novelty. Verify concurrent
   analyses and Shorts-before-highlights review. Deliver the actual request-specific
   review link through Telegram with acknowledgement, never a generated-only URL.
7. Owner selects/reviews clips. Later rendering, playable video/audio, copy and
   collaborator approvals and exactly-once Schedule Master records remain separate
   established gates. Public publishing stays disabled; no invitations or held
   retries are released.

## Tests actually executed

Focused existing hierarchy/relationships/orchestration run: **53 passed**, no
failures/skips (before the additional complete-conversation integration case).
Final new regression file: **17 passed**, no failures/skips. An initial fixture
omitted the existing selector's required duration field; the fixture was corrected,
without changing application behavior or weakening assertions.

Final full suite from `ripped-review-simplification`:

```powershell
$env:PYTHONUTF8='1'
$env:TEMP=(Resolve-Path ../ripped-intelligence/.test-tmp).Path
$env:TMP=$env:TEMP
$env:DATA_DIR="$env:TEMP/chronological-authorized-data"
$env:PYTHONPATH="$(Resolve-Path ../ripped-intelligence/.test-tmp/opencv4);$pwd;$(Resolve-Path ../pr-validation-handoff)"
& ../rippedshorts/.venv/Scripts/python.exe -m pytest -p validation_network tests -q -rs --tb=short --basetemp=../ripped-intelligence/.test-tmp/chronological-authorized --junitxml=../ripped-intelligence/.test-tmp/chronological-authorized.xml
```

**529 passed, 0 failed, 2 skipped, 2 subtests passed; 19.28 seconds.** Four
preexisting framework deprecation warnings. Skips: stored-render test lines142/176,
FFmpeg/ffprobe not on local PATH. Existing GitHub render job installs FFmpeg.
The restricted sandbox stalled at a local HTTP review test; two incomplete runs
were stopped and are not claimed as passing. The completed run used authorized
local socket access with external networking still denied by validation_network.
No production model or publishing operations occurred.

Existing GitHub workflow includes the new regression file and adds yt-dlp as a
test dependency because the real selector integration imports source_ingestion.
Final CI results are reported on the PR; local passing results do not pre-claim CI.
No existing safety assertions were deleted. Existing hierarchy fixtures now use
chronological topics and conversation containment, and selector expectations use
the new version while retaining learning, concurrency and provenance assertions.
