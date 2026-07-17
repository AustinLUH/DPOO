<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/18ac56c9-d112-4e12-aa02-7174b7edd37b" />


Loaded 12 preference pairs  
beta: 0.5  
mean DPO loss: 0.6371  
preference accuracy from DPO logits: 0.583  

Per-example diagnostics:
ex01: logit= 0.350, loss= 0.533, prompt=Explain why RLHF can be unstable.  
ex02: logit= 0.250, loss= 0.576, prompt=What is Direct Preference Optimization?  
ex03: logit=-0.050, loss= 0.718, prompt=Summarize the main advantage of pairwise preference training.  
ex04: logit= 0.400, loss= 0.513, prompt=Why keep a reference policy in DPO?  
ex05: logit= 0.300, loss= 0.554, prompt=What does beta control in DPO?  
ex06: logit=-0.075, loss= 0.731, prompt=Define a chosen response in preference data.  
ex07: logit= 0.300, loss= 0.554, prompt=Explain why scalar reward RL can be difficult for language models.  
ex08: logit=-0.150, loss= 0.771, prompt=What is a rejected response?  
ex09: logit= 0.350, loss= 0.533, prompt=Why might DPO be easier to implement than PPO-based RLHF?  
ex10: logit=-0.125, loss= 0.758, prompt=What is the intuition behind the DPO objective?   
ex11: logit=-0.250, loss= 0.826, prompt=What does a negative DPO margin suggest?  

## Brief answers

**What does a positive DPO logit mean?**

The DPO logit is `z = beta * [(log π_θ(y_w) − log π_θ(y_l)) − (log π_ref(y_w) − log π_ref(y_l))]`.
A positive `z` means the current policy prefers the chosen response over the rejected
response *more strongly* than the reference policy did — the policy has shifted
probability mass toward `y_w` and away from `y_l`, relative to where the reference
started. A negative `z` means the opposite: the policy has drifted toward the
rejected response, or away from the preference the reference already had.

**Why does DPO compare the current policy against a reference policy?**

The reference policy is a fixed anchor, so the objective measures a *relative*
shift in preference rather than an absolute one. Without it, the loss could be
minimized simply by pushing `log π_θ(y_w)` up and `log π_θ(y_l)` down without
bound, which could degrade the model's general fluency or behavior. Subtracting
the reference's log-probability gap implicitly enforces the same KL constraint
that RLHF enforces explicitly via a KL penalty term in its RL objective — DPO
gets this regularization "for free" inside a simple classification-style loss,
without needing a separate reward model or an RL training loop.

**What does the `beta` parameter control?**

`beta` scales how strongly the loss reacts to a given policy–reference gap —
in effect, the strength of the implicit KL penalty against the reference.
Higher `beta` makes the logit swing more sharply for the same underlying
log-probability difference, so the model is pushed harder to satisfy
preferences but is also more sharply penalized for drifting from the
reference. Lower `beta` flattens that response, letting the policy move
further from the reference before the loss reacts strongly.

**Low-loss vs. high-loss example**

Using the toy dataset below (`beta = 0.1`):

| Example | policy(chosen) | policy(rejected) | ref(chosen) | ref(rejected) | logit | loss |
|---|---|---|---|---|---|---|
| A — policy strongly learned the preference | -1.0 | -4.0 | -2.0 | -2.2 | **0.2800** | **0.5629** |
| E — policy strongly reversed the preference | -4.0 | -0.5 | -2.0 | -2.2 | **-0.3700** | **0.8952** |

- In **Example A** (low loss), the policy assigns the chosen response a much
  higher log-probability than the rejected one (`-1.0` vs. `-4.0`, a gap of
  `3.0`), compared to the reference's much smaller gap (`-2.0` vs. `-2.2`, a
  gap of `0.2`). The policy has *improved* its relative preference for the
  chosen response far beyond what the reference already showed, producing a
  strongly positive logit (`0.28`) and pushing the loss down toward `0`
  (`0.5629`, well below `log 2 ≈ 0.693`, the loss at a zero logit).

- In **Example E** (high loss), the policy has flipped the ranking entirely:
  it now scores the rejected response much higher than the chosen one
  (`-0.5` vs. `-4.0`), the opposite of what the preference data calls for.
  This produces a negative logit (`-0.37`) and a loss (`0.8952`) noticeably
  above `log 2`, correctly signaling that this example is being penalized —
  the policy needs to correct course on this pair.

Together, these confirm the objective's behavior: loss shrinks as the policy's
preference for the chosen response strengthens relative to the reference, and
grows as the policy drifts toward or reverses the rejected response — exactly
the ranking signal DPO is designed to enforce.
