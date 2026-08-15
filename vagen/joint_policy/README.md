# `vagen.joint_policy`

This package owns versioned policy and behavior-record contracts that are not
part of stock response-token PPO.

Current scope:

- the custom Nimloth async rollout captures identity-bound K-slot hidden states
  and raw action-boundary logits from the same generation and transports them to
  `DataProto`; capture schema v2 separates sticky episode `request_id` from a
  unique per-forward `generation_id`; this transport does not score Q or change
  the environment action;
- `contract.py` defines the provisional frozen-Q guided policy configuration,
  versioned behavior record, contract identity, and numerical audit reference;
- `torch_policy.py` implements the tensor distribution while always detaching Q
  and preserving the confirmed prior-to-LLM actor gradient;
- `sampling.py` defines the coordinator-owned logical draw key and maps its
  canonical SHA-256 digest to an exact 53-bit uniform value before pure inverse-
  CDF Scheme-B selection. The v2 audit record persists the complete key and
  revalidates the derived draw. Callers cannot supply a bare draw; the module
  imports no RNG, accepts no current Q, and never calls an environment;
- `replay.py` revalidates an immutable behavior-record batch and replays the
  selected guided-action log-prob from current prior logits plus only the
  rollout-persisted frozen Q; callers must provide the expected contract and
  snapshot identities;
- `critic_loss.py` defines a pure selected-action Huber objective: it gathers
  only the actually executed action, detaches return targets, and requires the
  caller to specify `delta` and reduction rather than choosing experiment
  defaults;
- `execution.py` defines the Navigation-only authorization envelope needed to
  preserve the exact raw LLM response while executing a separately sampled
  guided action. Schema v3 binds the full behavior record, raw-response SHA-256,
  parent-validated identity-bearing response-trace digest, and external action-
  draw record digest; remote
  transport uses a distinct `step_guided` method and validates the
  executed-action echo on both server and client;
- `frozen_q_actor.py` wraps the parent immutable snapshot owner in one explicit
  CPU/zero-GPU Ray actor, limits PyTorch native thread pools, and serializes batch
  pin, scoring, stage/activate CAS, and clean checkpoint operations;
- `AgentLoopManager` pins one active snapshot around a complete distributed
  rollout batch and preallocates deterministic per-turn keys from restart-stable
  dataset sample ids. The no-concat Gym loop performs capture validation, CPU Q
  scoring, keyed draw, response-trace/behavior assembly, guided environment
  execution, and ledger/DataProto persistence without rewriting the original
  response;
- the behavior contract binds action names, action token ids, score dtype,
  frozen-Q snapshot identity, selected action, and recorded log-probabilities;
- `training_contract.py`, `outcome.py`, and `terminal_state.py` define valid
  task outcomes, real terminal CoT+K-slot traces, discounted environment-reward
  returns, and rollout-time Frozen-V GAE without using selected-action Q as a
  state baseline; reward shaping is selected explicitly by environment config;
- `training_batch.py` compiles only identity-complete rollout evidence and
  excludes duplicated padding rows from actor, critic, and metrics; K4 rows
  additionally carry real nonterminal/terminal successor hidden states, guided
  actions, and in-memory observation images for every valid depth-1--4 window;
- `actor.py` is the custom FSDP actor candidate. It uses current action-boundary
  logits for executed-guided-action PPO and separates token reference KL from
  guided entropy. The legacy path co-trains a GPU DP-replicated action-value
  critic. The K4 path co-trains replicated projector, horizon-4 predictor, and
  eight-action ValueHead with state, frozen DINO-grid, SIGReg, and selected-
  action Huber losses through one three-group AdamW;
- `update_transaction.py` validates every rank's planning-module and optimizer
  identity before rank zero stages and CAS-activates the next immutable full K4
  planner transport (or the legacy critic-only snapshot);
- `checkpoint.py` writes the joint sidecar and completion marker only at a
  complete global-update boundary. K4 payloads include all three planning
  modules, unified optimizer, active full-planner transport, run identity, and
  stateful dataloader for exact fresh-runtime restore.

The optimizer-free standalone rollout remains available to explicit callers.
The trainer-side code is an integration candidate, not an enabled production
path. Its Torch/Ray, target-DP8 short-update, snapshot-publication, and exact
resume gates passed in ID171, but `RayPPOTrainer` still fails closed before
general production worker creation until humans explicitly choose and authorize
a production contract. No numerical training defaults are supplied here; ID171
values are smoke-only.
