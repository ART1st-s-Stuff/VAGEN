# Joint-policy scaffold

## Status

This document records the framework boundary. The optimizer-free production
rollout integration has passed its consolidated CPU/Ray regression and
independent review; the target TP8 guided one-turn gate is still pending. Joint
training remains deliberately disabled until its replay, optimizer, return, and
checkpoint boundaries are complete. Milestone M1 implements no actor logits or
PPO loss; the M2 contract fixes the confirmed Scheme-B gradient semantics and
the human-approved rollout ownership described below.

## Confirmed provisional policy

At each real environment turn, the LLM samples CoT `c` and a prior action token
`b`. The complete prior distribution `p` is the softmax over the LLM's action-
token logits at that boundary; `p` is deterministic given the LLM forward, while
the sampled token `b` contributes its log-probability to the LLM policy factor.
The provisional joint policy is

\[
\pi(c,b,a \mid x)
= \pi_{\mathrm{LLM}}(c,b \mid x)
  \pi_{\mathrm{guided}}(a \mid s,p,\bar Q).
\]

The environment executes only the first selected action and then replans from
the next real observation. Unexecuted simulated tail actions are not PPO
environment actions.

The first guided-policy candidate is scheme B:

\[
\pi_{\mathrm{guided}}(a \mid s,p,\bar Q)
= \operatorname{softmax}\left(
    \alpha l_{\mathrm{prior}}(a)
    + \beta\,\operatorname{stopgrad}(\bar Q(s,a))
  \right).
\]

The guided-policy actor loss backpropagates through `l_prior` into the LLM. Q is
always stop-gradient in that loss. Configurations with
`backprop_to_llm != true` are rejected rather than constructing an actor with no
trainable gradient path.

The old Nimloth `ValueHead` remains an action-value critic. It is not renamed to
an actor head and receives no actor-loss gradient. It is trained only on the
actually executed first action using Huber regression against a stop-gradient
discounted return constructed from real environment rewards. An
immediate reward alone is not the Q target, an advantage is not the Q target,
and unexecuted action slots receive no fabricated counterfactual target.
Terminal trajectories bootstrap with zero. A rollout truncation must use an
explicit rollout-time frozen-critic bootstrap. The snapshot owner now preserves
that rollout-time identity, but the training return compiler that consumes it is
still not implemented.

`\bar Q` must be a frozen rollout-time snapshot; replay must use the rollout-
persisted guidance scores and snapshot identity rather than recomputing them
after a critic update.

The rollout coordinator owns a stateless deterministic keyed draw. Its key binds
the run seed, policy step, stable rollout sample/repeat identity, turn index,
snapshot id, contract id, and RNG schema. The same logical decision therefore
keeps the same uniform draw across Ray scheduling, worker restart, and retry;
an explicitly new logical decision receives a new key. Worker-local RNG streams
and the vLLM token-sampling RNG are not used for guided-action selection.

A dedicated CPU Ray actor, created by `AgentLoopManager` before agent-loop
workers, owns the active read-only frozen critic snapshot. The trainable current
critic and its optimizer remain trainer-owned. After one complete global joint
update succeeds, the trainer stages a new immutable snapshot and atomically
activates it for the next rollout batch. A snapshot is never refreshed inside a
batch or inside PPO minibatches, and old behavior records continue to use their
persisted old Q and snapshot identity. Disk checkpoint cadence may be less
frequent, but a complete checkpoint must bind the current critic, optimizer,
active snapshot, global step, and draw-key state needed for exact resume.

## Deferred policy decisions

The following still require explicit decisions before M2/M3 are complete:

- positive `alpha`, non-negative `beta`, prior temperature, discount `gamma`, score dtype, critic loss
  coefficient, and any warmup or KL target;
- critic coverage/calibration for action slots not executed in a given state;
- how simulated tail actions are generated and which non-PPO auxiliary objective,
  if any, trains them.

## M2 contract status

The dependency-light Scheme-B contract, same-generation capture transport, and
optimizer-free production rollout wiring are implemented as one candidate. The
training entry remains disabled:

- all probability semantics (`alpha`, `beta`, prior temperature, score dtype,
  and the required LLM gradient path) are explicit and enter a hashed contract
  id;
- the tensor formula always detaches frozen Q and always keeps prior logits
  connected to the LLM;
- a versioned behavior record binds the action table/token ids, sampled prior
  token, prior logits, frozen all-action Q, guided first action, snapshot id, and
  behavior log-probabilities;
- ledger v2 embeds and revalidates the complete behavior record before claiming
  policy ownership;
- the Nimloth async replica reuses the existing vLLM worker hook to capture the
  generated K latent hidden rows and raw action-boundary logits without a second
  transformer replay; a two-phase TP protocol validates every rank before the
  LM-head collective. Capture schema v2 keeps the episode `request_id` only for
  sticky server routing and binds every forward to a separate unique
  `generation_id`; both identities and the exact token table travel with the
  sidecar;
- the parent Nimloth layer now owns a strict pure capture-to-Q scorer: it checks
  capture v2/session/generation/token identities, feeds hidden rows in the
  frozen snapshot's parameter dtype, and emits an immutable record containing
  raw prior logits and all-action frozen Q. The output score dtype is hashed into
  the policy contract and snapshot identity rather than selected per scoring
  call. `AgentLoopManager` now routes validated same-generation captures to the
  pinned CPU owner before any environment mutation;
- a Navigation-only guided execution envelope now binds the complete behavior
  record, raw LLM response SHA-256, and identity-bearing response-trace digest
  while authorizing a separately selected environment action. The parent pure
  assembler validates request/generation/spec identity, exact token decode,
  canonical response mask/log-probs, and accepts the guided action as an
  external input without owning RNG. Remote execution uses an explicit `step_guided` method,
  revalidates before mutation, and checks the environment's action echo on both
  server and client. The no-concat agent loop now creates this envelope and
  executes it only through the explicit guided capability;
- the rollout coordinator's stateless keyed-draw contract binds run seed,
  policy step, stable sample/repeat identity, turn, validation mode, snapshot,
  contract, and schema. Canonical SHA-256 maps each key to one exact 53-bit
  uniform value; the public sampler accepts the full key rather than a caller-
  chosen draw, applies half-open inverse-CDF selection, and persists/revalidates
  the complete provenance in action-draw schema v2. It imports no RNG and never
  calls the environment. `AgentLoopManager` now pins one snapshot around the
  complete distributed rollout batch and allocates every per-turn key from the
  stable dataset sample id and explicit repeat index before dispatching workers;
- a pure behavior-replay helper revalidates one homogeneous contract/snapshot
  batch, uses only each rollout record's persisted frozen-Q vector, and exposes
  the selected current/behavior guided log-probabilities while preserving the
  current-prior gradient; it is not yet connected to actor FSDP replay;
- a pure selected-action Huber helper gathers only the actually executed action
  and detaches the return target; Huber delta and reduction remain mandatory
  caller inputs, and no critic optimizer or return compiler is connected;
- a dedicated one-CPU/zero-GPU Ray actor owns the immutable active snapshot,
  limits PyTorch intra-op and inter-op threads to one, pins open batches, scores
  captures, stages newer dtype/architecture-compatible snapshots, activates by
  compare-and-swap only with no open batch, and exports only clean checkpoint
  state. Disabled mode does not alter legacy custom worker or agent-loop
  constructor signatures;
- every guided turn persists the batch pin, scoring record, response trace,
  action draw, execution envelope, guided ledger, stable trajectory identity,
  raw response IDs/mask/log-probs, and actual environment reward into
  `DataProto`; historical `turn_idx` remains one-based while the explicit
  `guided_turn_index` and draw-key turn are zero-based and must differ by one;
  manager failure paths, including an ambiguous pin RPC failure, make
  best-effort unpin attempts before returning;
- capture currently requires `data_parallel_size=1` and eager vLLM execution;
  unsupported DP routing and conflicting engine overrides fail closed;
- `RayPPOTrainer` still rejects `joint_policy.enabled=true` before worker
  creation. Only the explicit optimizer-free standalone entry may construct the
  guided runtime, with all policy values, run seed, critic dimensions, and
  snapshot source step supplied by the caller. Actor replay, critic optimization,
  return compilation, global-update snapshot publication, and complete
  checkpoint/resume remain unconnected.

VAGEN-Lite currently has no `[B,A]` action-value head equivalent to the old
Nimloth `ValueHead`. Its existing token critic is scalar per response token, and
its transition reward predictor is an immediate-reward model. Neither is used
as a substitute. The chosen critic state is the mean of the per-slot outputs from the existing
`SharedSlotProjector` applied to same-generation K-slot hidden rows. The parent
critic/snapshot/scoring foundation now validates that path. The keyed sampler,
CPU Ray owner, batch pin, agent-loop execution, and DataProto provenance are now
wired for optimizer-free rollout. Trainer-owned current-critic optimization,
per-global-update publication, actor replay, and complete checkpoint/resume are
still absent.

## M1: decision ledger

M1 records environment facts before adding an actor. It uses VAGEN-Lite's async
**no-concat** agent loop because each environment turn remains an independently
identified sample.

Navigation can opt in with the following settings; the trainer rejects the
ledger unless both async rollout and no-concat mode are active. A K16 Nimloth
checkpoint must also use the explicit K16 environment protocol; relying on the
K1 environment default fails closed. Reference train/validation environment
files are `examples/train/navigation/{train,val}_navigation_nimloth_k16.yaml`.

```yaml
trainer:
  concat_multi_turn: false
decision_ledger:
  enabled: true
# Each Navigation/RemoteEnv config:
config:
  prompt_format: nimloth
  latent_token_count: 16
  max_actions_per_step: 1
```

Each turn then carries one versioned `decision_ledger` with:

- an explicit action-space name and its complete ordered action-name table;
- all actions actually executed by the environment, in order;
- canonical action ids and names checked against that table;
- the source of every executed action;
- whether every action has valid action-policy sampling ownership;
- the environment turn reward;
- environment termination versus rollout truncation;
- action-format validity.

M1 has no action-policy sampler, so every
`decision_is_policy_sampled` entry is false. Upstream VAGEN passes the sampled
assistant text directly to the environment and has no latent fallback adapter.
Turn reward is anchored to the last policy-owned response token; a turn with no
policy-owned token fails closed rather than assigning reward to padding.

The ledger follows this path without lossy first-action conversion:

```text
GymAgentLoop(no-concat)
  -> AgentLoopOutput.extra_fields["decision_ledger"]
  -> DataProto.non_tensor_batch["decision_ledger"]
  -> RayPPOTrainer strict validation and coverage metrics
```

No fork-specific predictor supervision is restored on this upstream baseline.
The ledger records execution facts only; it is not a world-model training
interface.

### M1 invariants

- Ledger schema and action-space versions are explicit.
- List lengths agree exactly, and every action id resolves to its recorded name.
- Only actions reported as executed by the environment are recorded.
- M1 rejects any claim that an action was sampled by the undecided actor.
- A terminal turn and a truncated turn cannot both be true.
- Navigation and remote-client step results preserve strict reward, done,
  format, action-space, and fallback-source types rather than coercing them.
- The upstream assistant action text reaches the environment unchanged; no
  fork-specific `latent_plan` fallback is installed.
- When the opt-in ledger is enabled, missing or malformed ledgers stop training
  before old log-probability computation.
- M1 adds no trainable state and therefore changes no checkpoint ownership.

## Later milestones

### M2: guided-policy protocol and checkpoint ownership

The optimizer-free guided rollout portion is complete. The current training
candidate additionally compiles discounted environment rewards and rollout-time
Frozen-V GAE, trains a GPU DP8 replicated current action-value critic on the
actually executed action, publishes source-step+1 snapshots only after every
rank completes, and saves the current critic/optimizer plus active snapshot in
an atomic global-update checkpoint. Environment configs own reward shaping;
the compiler validates trajectory topology without forcing intermediate rewards
to zero.

### M3: joint PPO

The custom FSDP actor candidate replays current action-boundary logits from the
same transformer forward used for token reference KL. It applies PPO ratio and
clipping only to the executed guided action, always detaches rollout-persisted
frozen Q, and keeps token low-variance KL separate from guided-action entropy.
Task-limit failure is terminal; its configured environment rewards remain in the
discounted return (the ID169 smoke explicitly configures all failure rewards to
zero). Infrastructure failures produce no training row. A terminal observation
stores a separate real CoT+K-slot trace ending at action-start, without an
executed action.

This is still a gated implementation candidate. It must pass complete Torch,
Ray, DP8 short-update, snapshot-publication, and interrupted-resume tests before
the production trainer opt-in is allowed to create workers.

## Validation boundary

Dependency-light tests cover M1 plus Scheme-B config, numerical reference,
behavior/ledger schema identity, overflow handling, and fail-closed wiring.
Complete CPU dependencies validate request-scoped/out-of-order capture, partial
TP failure before LM-head collectives, error cleanup, request/token identity,
and DataProto propagation. The older direct-vLLM path has separate GPU evidence,
and the async same-generation transport passed its target TP8 GPU capture gate.
That gate did not score Q or execute a guided action. The optimizer-free production wiring passed `514` expanded parent/VAGEN tests
plus `75` subtests at parent `06b993b2`, VAGEN `3a01a2a0`, and VERL `084f042b`;
the suite includes a real local Ray actor lifecycle. A separate Claude Opus
read-only review rechecked the final source and returned `APPROVED` with no
P0/P1/P2 findings. The optimizer-free target-TP8 guided one-turn gate then
passed as ID164 on held-out Navigation `base` seed 0 at parent `d40b7bdc`, VAGEN
`3931dc0b`, and VERL `084f042b`: same-generation `[16,2048]` capture and raw
8-action logits were scored by the frozen CPU critic snapshot; coordinator key
`run_seed=42, policy_step=0, sample=standalone:navigation:base:0, repeat=0,
turn=0` selected guided action 2 while the raw LLM prior action remained 0; the
environment executed only guided action 2 and the validator returned `ALL_OK`
for pin/scoring/trace/draw/behavior/execution/ledger identities and reward.
ID163 had produced no model or rollout evidence because a preflight-created
`external/le-wm/__pycache__` correctly tripped the clean-worktree gate; ID164
used a new numeric ID, empty output, and fresh production worktree.

The current training candidate is VAGEN `2a32f2f` with VERL `42cb2f12`.
Fresh server worktrees at the exact parent/VAGEN/VERL commits passed `72` focused
Torch/contract/update/checkpoint tests plus `40` subtests, and an expanded suite
passed `309` tests plus `115` subtests. This includes a real TensorDict/DataProto
compiler, one CPU distributed actor+current-critic update, current critic and
optimizer export/restore, atomic sidecar digest checks, custom-class resolution,
and a real local Ray frozen-owner restore. These gates found and fixed a
TensorDict key-iteration bug and scalar Adam step fingerprint bug.

Target-DP8 FSDP/vLLM execution and an interrupted distributed trainer resume
have not run. `RayPPOTrainer` therefore still refuses production joint training
before worker creation.
