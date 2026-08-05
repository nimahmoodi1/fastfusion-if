# Round 6 — my last fix broke this, and here is why

You were right to test twice. The failure is real, it is my fault, and it is the
direct consequence of the fix from round 5.

---

## What happened

Round 5 fixed this: `capture()` rebound its own attribute to a detached copy, so
the caller kept holding CUDA tensors and `.numpy()` raised. The fix made
`detach_cpu_()` mutate in place.

But `integrated_gradients` reads the captured tensors **after** the context
exits, and then **replays them through the network tail**:

```python
with torch.no_grad(), inst.capture() as obs:
    model(batch)

z_geom = obs.z_geom          # <- now CPU, because round 5 fixed the move
...
out = tail(r)                # <- model is on CUDA  -> RuntimeError
```

Before round 5 this worked *by accident*: the rebinding bug meant `obs` still
held CUDA tensors, which is exactly what the replay needed. One bug was masking
the requirement of the other. Fixing the visible one exposed the hidden one.

Two callers, two opposite needs, one switch that did not exist:

| caller | needs | why |
|---|---|---|
| `cmd_attribute` | tensors on **CPU** | calls `.numpy()` to write `gates.npz` |
| `integrated_gradients` | tensors on the **model device** | feeds them back through the tail |

## The fix

`capture()` now takes an explicit `to_cpu` flag, defaulting to `True`:

```python
with inst.capture() as cap:                 # CPU: safe for .numpy()
with inst.capture(to_cpu=False) as obs:     # model device: safe to replay
```

`integrated_gradients` passes `to_cpu=False`; everything else uses the default.
All ten call sites audited and confirmed.

## Why this cannot happen a third time

The two directions are now pinned by two tests that fail in opposite ways:

- `test_intermediates_are_on_cpu_after_capture` — fails if we stop moving, which
  is the round-5 `.numpy()` bug.
- `test_capture_to_cpu_false_keeps_model_device` — fails if we always move,
  which is this round's device-mismatch bug.

Fixing one without the other now breaks a test rather than the run. Verified
with a torch-free stand-in:

```
capture(to_cpu=True ) -> caller sees cpu    (needed for .numpy())
capture(to_cpu=False) -> caller sees cuda   (needed to replay the tail)
```

The docstring on `capture()` names both error messages, so whoever hits this
next has the answer in front of them.

## Test count

33 defined; `18 passed, 15 skipped` without a checkpoint. With
`--ckpt/--manifest/--cache-dir`, all 33 should pass — including the six that
failed in your log.
