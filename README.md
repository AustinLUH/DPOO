The output of pytest -q (same as  python -m pytest)
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/18ac56c9-d112-4e12-aa02-7174b7edd37b" />

## The output of the toy experiment command:##
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
started. A negative `z` means the opposite: the policy prefers the chosen response
less strongly than the reference did (or has reversed the ranking).

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
A higher `beta` makes the logit swing more sharply for the same underlying
log-probability difference, so the model is pushed harder to satisfy
preferences but is also more sharply penalized for drifting from the
reference. A lower `beta` flattens that response, letting the policy move
further from the reference before the loss reacts strongly.

**Low-loss vs. high-loss example**

Toy experiment results (`beta = 0.5`, 12 preference pairs, mean loss = `0.6371`,
preference accuracy = `0.583`):

| Example | logit | loss | prompt |
|---|---|---|---|
| **ex04** (lowest loss) | **0.400** | **0.513** | Why keep a reference policy in DPO? |
| **ex11** (highest loss) | **-0.250** | **0.826** | What does a negative DPO margin suggest? |

- **ex04** has the most positive logit in the set (`0.400`) and correspondingly
  the lowest loss (`0.513`, below `log 2 ≈ 0.693`). A positive logit means the
  policy assigns a larger relative advantage to the chosen response over the
  rejected response than the reference policy did — the policy moved in
  exactly the direction the preference data calls for, so the loss is small.
  Since `loss = log(1 + exp(-logit))`, a larger positive logit drives
  `exp(-logit)` toward 0, pushing the loss toward 0.

- **ex11** has the most negative logit (`-0.250`) and the highest loss
  (`0.826`, above `log 2`). A negative logit means the policy prefers the
  chosen response *less* strongly than the reference did for this pair — the
  ranking may even be reversed relative to what the training data wants.
  Because the logit is negative, `exp(-logit)` grows larger than 1, pushing
  the loss above `log 2` — this example is penalized, and its gradient will
  push the policy to widen the chosen/rejected log-probability gap.

The full table confirms the monotonic relationship the objective is designed
to enforce: every example with a positive logit (ex01, ex02, ex04, ex05, ex07,
ex09) has loss below `0.693`, and every example with a negative logit (ex03,
ex06, ex08, ex10, ex11) has loss above `0.693` — loss decreases smoothly as
the logit increases.
