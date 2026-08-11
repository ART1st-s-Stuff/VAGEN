# Joint-policy scaffold

## Status

This document records the framework boundary while the actor parameterization is
still undecided. It is not an actor design and does not select a logit formula.

The intended joint policy is currently written as

\[
\pi(c, p, A \mid x)
= \pi_{\mathrm{LLM}}(c, p \mid x)
  \pi_{\mathrm{actor}}(A \mid s, p),
\]

where `c` is the sampled CoT, `p` is the action prior, and `A` is an ordered
action sequence. Whether `p` is a sampled variable or a deterministic LLM head
output remains unresolved and must be settled before PPO ratios are defined.

## Deferred actor decisions

This scaffold deliberately does not decide:

- how actor logits are computed;
- whether the actor has an independent trainable residual head;
- whether an action-value term is used, stopped, normalized, or snapshotted;
- whether the environment commits a complete action sequence or replans after
  each executed action;
- whether PPO clips one executed action, each sequence position, or a complete
  sequence ratio;
- how actor parameters, optimizer, scheduler, and any target critic are saved.

In particular, the candidate

\[
z(a) = \alpha l_{\mathrm{prior}}(a)
     + \beta\,\operatorname{stopgrad}(Q(s,a))
\]

is only a proposal. No field or class in milestone M1 may imply that it has been
accepted.

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

### M2: actor protocol and checkpoint ownership

Only after the actor design is approved:

1. define the conditional action/sequence distribution;
2. record exact rollout-time component and joint behavior log-probabilities;
3. specify the replay state and critic/target snapshot identity;
4. register every trainable module, optimizer, scheduler, and target state in a
   checkpoint round-trip test;
5. promote policy-owned actions to a new ledger schema version.

### M3: joint PPO

Implement current log-probability replay, ratio construction, credit assignment,
clipping, entropy/KL terms, and actor update only after M2 is complete. If only
the first planned action is executed before replanning, unexecuted tail actions
must not be presented as on-policy environment actions.

## Validation boundary

M1 unit tests are dependency-light and cover ledger construction, strict
validation, multi-action preservation, fallback ownership and reward anchoring,
remote step decoding, termination semantics, and static wiring order. Full Ray,
multimodal rollout, PPO, checkpoint, and GPU validation are explicitly outside
M1.
