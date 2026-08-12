# Joint-policy scaffold

## Status

This document records the framework boundary while state ownership and rollout
integration remain undecided. Milestone M1 implements no actor logits or PPO
loss; the M2 contract fixes the confirmed Scheme-B gradient semantics.

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
explicit rollout-time frozen-critic bootstrap; this remains blocked until the
critic state and snapshot owner are implemented.

`\bar Q` must be a frozen rollout-time snapshot; replay must use the rollout-
persisted guidance scores and snapshot identity rather than recomputing them
after a critic update.

## Deferred policy decisions

The following still require explicit decisions before M2/M3 are complete:

- positive `alpha`, non-negative `beta`, prior temperature, discount `gamma`, score dtype, critic loss
  coefficient, and any warmup or KL target;
- critic coverage/calibration for action slots not executed in a given state;
- how simulated tail actions are generated and which non-PPO auxiliary objective,
  if any, trains them;
- checkpoint and refresh timing for the frozen critic snapshot.

## M2 contract status

The dependency-light Scheme-B contract is implemented, but operational rollout
remains disabled:

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
- `joint_policy.enabled=true` fails closed while Q ownership, rollout sampling,
  replay, and checkpoint refresh are not connected.

VAGEN-Lite currently has no `[B,A]` action-value head equivalent to the old
Nimloth `ValueHead`. Its existing token critic is scalar per response token, and
its transition reward predictor is an immediate-reward model. Neither is used
as a substitute. The state feeding the old ValueHead must be chosen before Q
ownership can be implemented safely.

## M1: decision ledger

M1 records environment facts before adding an actor. It uses VAGEN-Lite's async
**no-concat** agent loop because each environment turn remains an independently
identified sample.

Navigation can opt in with the following settings; the trainer rejects the
ledger unless both async rollout and no-concat mode are active:

```yaml
trainer:
  concat_multi_turn: false
decision_ledger:
  enabled: true
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

1. sample one real action from the LLM-prior/frozen-Q guided distribution;
2. record the sampled LLM prior token, prior logits/distribution, frozen
   all-action guidance scores, selected-action behavior log-probabilities, and
   critic snapshot identity;
3. replay current LLM logits against the same persisted Q-guidance scores;
4. regress the selected current Q against stop-gradient discounted real-reward
   return while keeping it separate from frozen policy guidance;
5. checkpoint the critic and the exact snapshot-refresh/bootstrap boundary;
6. promote the one executed policy-owned action to a new ledger schema version.

### M3: joint PPO

Implement the LLM and guided-policy component ratios with the guided factor
backpropagating through prior logits, joint ratio/clipping, first-action credit,
entropy/KL terms, selected-action return regression, and critic update ordering.
Simulated tail actions must remain outside environment PPO.

## Validation boundary

Dependency-light tests cover M1 plus Scheme-B config, numerical reference,
behavior/ledger schema identity, overflow handling, and fail-closed wiring. Torch
autograd/parity tests exist but are skipped in the current local environment
because torch is unavailable. Full Ray, multimodal rollout, Q ownership, PPO,
checkpoint, and GPU validation remain outside the completed scope.
