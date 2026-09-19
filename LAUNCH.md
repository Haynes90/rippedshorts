# Basic billable pilot: transcript + 9:16 + 16:9

Current scope: an operator-run paid pilot. Customers use hosted checkout and an intake
form; you verify orders, process them through `/review`, and privately deliver files.
There are no customer login accounts, caption features, automatic publishing, or Stripe
webhooks. Payment status, trial eligibility, billing-cycle allowances and customer
delivery are deliberately manual for the fastest first launch.

## Your setup checklist

1. **Stripe:** complete business/payout setup. Create “Creator Starter” with a recurring
   price of $10 per month, then create a Payment Link. Use test mode initially.
   Stripe's guide: https://docs.stripe.com/no-code/payment-links.
2. **Intake form:** create a Google Form collecting email, paid order/reference or trial
   request, YouTube/Drive URL, selected start/end times, and preferred clip topics.
   Ask users to confirm they can use the submitted material. Keep responses private.
   Link its responses to a customer-only Sheet. Maintain a separate usage tab with
   customer reference, billing-period start/end, minutes used, shorts delivered,
   highlights delivered, trial-used flag, and delivery/expiry date.
3. **Customer storage:** create a separate customer Drive folder and workflow Sheet.
   Grant the processing service account access to those destinations. Use a dedicated
   processing API key/project with a budget; shared billing/rate limits can still affect
   personal work if you reuse the same provider account.
4. **Railway:** create a NEW service and volume. Do not change or redeploy your personal
   Ripped Shorts service. Deploy a separate code repository/copy of this branch. The
   local branch has not been pushed to GitHub. Set `APP_MODULE=pilot_server:app`, mount
   the volume at `/data`, set `DATA_DIR=/data`, and copy the variable names from
   `pilot.env.example`. Do not import personal settings wholesale. Configure one
   replica and one Uvicorn worker. Volume guide: https://docs.railway.com/volumes.
5. **Access:** generate a random studio secret of at least 32 characters. Put it only in
   the new Railway service's `REVIEW_HUB_SECRET` variable. Share neither it nor the
   `/review` operator workspace with customers. Use Railway's public HTTPS domain;
   a custom domain is optional.
6. **Checkout/intake links:** set `PILOT_CHECKOUT_URL` and `PILOT_INTAKE_URL`. The public
   landing page is `/`; it displays a preparation message until both URLs are configured.
   A checkout redirect or form submission is NOT proof of payment. Verify the active
   subscription/payment directly in Stripe before fulfillment, including cancellations.
7. **Launch gate:** run one authorized source through both formats, inspect complete
   speech/framing, download the transcript, and verify the customer can access only
   their delivered files. Test payment in Stripe test mode. Then enable the live link.

## Pilot offer and limits

Starter: $10/month, 60 selected source minutes across up to two videos, up to 10
finished Shorts and 2 Highlights total per billing cycle. Trial: once per customer,
up to 15 minutes from one source, up to 2 Shorts and 1 Highlight, 7-day file access.
Paid access is 30 days. Quality selection can return fewer clips if the source has
insufficient complete thoughts. Existing Shorts are 10–90 seconds; existing Highlights
are at least 3 minutes, so very short selected sections may yield no Highlights.

The API enforces selected minutes **per submission**, a per-project render cap, and
an operator checkbox acknowledging eligibility/remaining allowance. It does NOT yet
enforce a customer's aggregate billing-cycle quota or detect repeat trial accounts.
Use the usage Sheet before every submission/approval. Do not advertise self-service
or unlimited automatic access in this phase. Failed renders should not consume the
customer's output allowance. Check and adjust the ledger for failed source processing.

## What the code does

- `pilot_server.py` serves the public offer and private operator hub. It does not serve
  the legacy public intake/debug routes or register personal Telegram webhooks. In this
  entry point Telegram calls are disabled; no bot is needed for the customer pilot.
- `review_hub.py` provides signed HttpOnly operator sessions, owner checks, project
  submission/status, review, transcript downloads, and finished private Drive links.
- `review_basic.py` merges overlapping selections, allows up to four independent ranges,
  downloads the source, trims each range, and transcribes/selects only trimmed sections.
  Downloading may still retrieve the entire YouTube source. That bandwidth/disk cost is
  not a promise to download only the selected bytes. YouTube access can require cookies.
- The existing verified renderers create both formats. Each candidate carries its own
  selected-section source path, so completion checks cannot extend into unselected
  source speech. The section's original offset is retained in transcript/time displays.
- A new project uses `project_id=request_id` and `account_id=pilot:{customer_reference}`.
  Use a stable reference per customer. Every order gets its own Drive folder title.
- Kept/rejected choices remain in SQLite and recent examples for the SAME customer are
  used in future Shorts selection. Unreviewed clips are not rejection examples. The
  personal Google Doc framework and global approval/boundary learning are not injected
  into these customer selections. This is preference prompting, not custom-model training.
- Original source paths, choices and statuses persist on the volume. Retrying a failed
  selection can reuse extracted sections and transcripts. A completed selection is not
  regenerated over human decisions. The pilot has no automatic watchdog: use the
  operator retry for failed jobs; interrupted/stuck work needs operator inspection.
- Web review access expires using the job's metadata. Private Drive delivery/expiry is
  operator-managed; no existing shared Drive files are automatically deleted. Arrange
  revocation/deletion of customer deliverables on the advertised date. Internal local
  source cache cleanup is also manual in this pilot entry point; monitor volume usage.

## Inspection and safety boundary

Started from `Haynes90/rippedshorts` commit
`882b38ac6fdefc8ba636725b08c8c4493cc9a5e8`, on local branch
`feature/ripped-shorts-mobile-review`. Existing dependencies are FastAPI, SQLite,
Google Drive/Sheets, OpenAI, yt-dlp, FFmpeg/FFprobe, and the existing framing logic.
The new customer path downloads sources directly; it does not enqueue work in your
personal Audio Master or Schedule Master. No new frontend build system/database
subscription is required. The personal service's default entry point remains `main:app`.

Nothing has been pushed or deployed. No real payment, private customer upload, Google
write, Telegram message, or paid model-processing job was performed during local tests.

## Local checks

```sh
python -m pip install -r requirements.txt pytest httpx
python -X utf8 -m pytest tests/test_review_hub.py tests/test_review_integration.py tests/test_review_basic.py -q
```

Use a writable explicit `--basetemp` if the system temporary directory is restricted.
Tests cover authentication, owner isolation, duplicate approval, expiry, selected-range
limits, learning isolation, render caps, transcript downloads and trimmed-source routing.
Provider and payment integration still require the live staging test above.

The original full repository suite had 21 failures/124 passes in this Windows
environment; the initial changes reproduced the same 21 failures. Many are outdated
source-string assertions; four involve restricted temporary directories. This is an
existing baseline, not a clean production certification. No claim of live readiness
should be made before the real source-to-delivery test.

For a local UI fixture only: `python tests/review_demo.py`, open
`http://127.0.0.1:8765/review`, key `local-demo-access-key-not-for-production`.
Its processing is mocked and its records are disposable. It is excluded from Docker.

## r3cycle customer portal (current implementation)

The public landing page now starts signup at `/app`. Customer accounts use scrypt
password hashes and revocable, hashed session tokens. Stripe checkout receives the
signed-in account's reference; the signed webhook links the subscription. The app
fetches the subscription from Stripe before every submission, checking the exact
Starter price, customer, status and billing period. Redirects alone grant no access.
Monthly budgets and the lifetime account trial budget are reserved transactionally.
Requested clip counts are reserved at submission and cap operator renders.
The customer portal shows owned projects, finished links, and transcript downloads.
Operators still approve/render clips at `/review`; no automatic publication occurs.

### Deployment configuration

Use only project `2bc47859-a4f1-4381-b7a1-fb683390eda9`, customer service
`1e52a9a0-1e0f-4dd9-81cc-215eb942766e` (`recycle-app`). Do not connect the personal
production branch. Publish this customer branch or a separate private customer repo
first, then connect that specific source to this empty service.

Mount a persistent Railway volume at `/data`, use one replica, and set
`APP_MODULE=pilot_server:app`, `DATA_DIR=/data`, and the values in `pilot.env.example`.
Use a random 32+ character `REVIEW_HUB_SECRET`. Dedicated customer Drive folder and
workflow Sheet are required. Add `GOOGLE_CREDENTIALS` and `OPENAI_API_KEY` privately.

Additional Stripe settings:
- `STRIPE_SECRET_KEY`: server API secret, from the same Stripe account/mode as checkout.
- `STRIPE_PRICE_ID`: the monthly $10 Starter `price_...` identifier.
- `STRIPE_PAYMENT_LINK_ID`: the verified checkout's `plink_...` identifier, NOT its public URL slug.
- `PILOT_CHECKOUT_URL`: https://buy.stripe.com/3cI14p3xb0fK7Ted3Y5ZC00
- After a Railway domain is generated, register `https://YOUR-DOMAIN/app/stripe/webhook`
  in Stripe for `checkout.session.completed`. Enter its signing secret as
  `STRIPE_WEBHOOK_SECRET` in Railway.
- Set the payment link's after-checkout redirect to `https://YOUR-DOMAIN/app`.
  Customers can also return manually and click Refresh billing & projects.
- Subscription changes/cancellations are read directly from Stripe on access checks;
  stale or reordered subscription webhooks cannot grant access.
- Start checkout from `/app`, not the raw payment link. Earlier direct subscriptions
  without an account reference need operator reconciliation; they are not auto-claimed.

Test with matching Stripe sandbox keys, price, payment link and webhook before live
traffic. The public live link has only been viewed; no real subscription or payment
was made by this implementation. Local tests mock Stripe and processing providers.

### Remaining launch checks and limitations

The website is not deployed yet. Verify a real source download, transcription,
vertical/horizontal render and customer Drive access in the separate service before
accepting customers. Publishing the landing page alone is not a completed launch.
Private Drive delivery may still need operator sharing to the customer's email.

No email verification, automated password recovery, account deletion, or guaranteed
trial abuse prevention across multiple accounts is provided in this small pilot.
The signup page discloses the absence of self-service recovery. Trial limits are per
account, not verified person. Operator review is required for suspected duplicate trials.

Failed jobs keep their allowance reserved for an operator retry; do not ask customers
to resubmit. A process crash between reservation and dispatch leaves a held reservation
for operator reconciliation. Background jobs do not automatically resume on restart;
check accepted/processing jobs after a deploy and reconcile before retrying. Deployment
should happen with no active customer jobs. Expiry hides portal downloads; Drive file
and source-media deletion is still manual. Learning remains customer-specific in the
job database. A cancellation blocks new submissions after entitlement ends, while
existing downloads remain available until their stated expiry.

The older manual intake instructions above are superseded by this portal. The operator
submission screen remains available for support, and its allowance checkbox is a trusted
operator override; use the customer portal for normal account-budget enforcement.
