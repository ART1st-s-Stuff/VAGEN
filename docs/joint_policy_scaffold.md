# Joint-policy scaffold

## Status

This document records the framework boundary while some policy-gradient details
remain undecided. Milestone M1 still implements no actor logits or PPO loss.

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

The old Nimloth `ValueHead` remains an action-value critic. It is not renamed to
an actor head and receives no actor-loss gradient. `\bar Q` must be a frozen
rollout-time snapshot; replay must use the rollout-persisted guidance scores and
snapshot identity rather than recomputing them after a critic update.

## Deferred policy decisions

The following still require explicit decisions before M2/M3 are complete:

- whether the guided-policy factor backpropagates through
  `l_prior` into the LLM; without that path it has no trainable parameter during
  one PPO update and its ratio remains one;
- `alpha`, `beta`, prior temperature, and any warmup or KL target;
- critic coverage/calibration for action slots not executed in a given state;
- how simulated tail actions are generated and which non-PPO auxiliary objective,
  if any, trains them;
- checkpoint and refresh timing for the frozen critic snapshot.

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
`decision_is_policy_sampled` entry is false. System-injected fallback tokens are
also excluded from the LLM response loss mask; a filled zero must never pretend
to be a behavior log-probability. Turn reward remains anchored to the last
policy-owned response token, so excluding an injected suffix does not erase the
environment reward.

The ledger follows this path without lossy first-action conversion:

```text
GymAgentLoop(no-concat)
  -> AgentLoopOutput.extra_fields["decision_ledger"]
  -> DataProto.non_tensor_batch["decision_ledger"]
  -> RayPPOTrainer strict validation and coverage metrics
```

The existing `action_label` and `step_reward` fields remain available for the
current predictor experiment. They are not the joint-policy contract.

### M1 invariants

- Ledger schema and action-space versions are explicit.
- List lengths agree exactly, and every action id resolves to its recorded name.
- Only actions reported as executed by the environment are recorded.
- M1 rejects any claim that an action was sampled by the undecided actor.
- A terminal turn and a truncated turn cannot both be true.
- Navigation and remote-client step results preserve strict reward, done,
  format, action-space, and fallback-source types rather than coercing them.
- The latent fallback adapter only runs for `prompt_format=latent_plan`.
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
4. keep trainable current-critic regression separate from frozen policy guidance;
5. checkpoint the critic and the exact snapshot-refresh boundary;
6. promote the one executed policy-owned action to a new ledger schema version.

### M3: joint PPO

After deciding whether guided-policy gradients reach the LLM, implement the LLM
and guided-policy component ratios, joint ratio/clipping, first-action credit,
entropy/KL terms, and critic update ordering. Simulated tail actions must remain
outside environment PPO.

## Validation boundary

M1 unit tests are dependency-light and cover ledger construction, strict
validation, multi-action preservation, fallback ownership and reward anchoring,
remote step decoding, termination semantics, and static wiring order. Full Ray,
multimodal rollout, PPO, checkpoint, and GPU validation are explicitly outside
M1.
