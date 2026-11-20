# Good-only type KL: exact objective and implementation

This is the original Route-SFT objective plus one extra regularizer, with the
subsequently agreed one-epoch schedule. It is not the marker-calibration / tag-CE
recipe and not the type-BCE or common-action-wrapper variant.

Let $s_i=(H_i,R_i)$ be the exact stored causal prefix ending in `</think>\n`,
$y_i=1$ for Bad and $0$ for Good, and $Y_i$ the original supervised suffix
(including the Bad marker where present, the complete action, and EOS).

$$
\mathcal L=\frac1B\sum_{i=1}^{B}\left[
 -\frac1{|Y_i|}\sum_{t\in Y_i}\log\pi_\theta(y_{i,t}\mid s_i,y_{i,<t})
 +\lambda_r\operatorname{BCE}(q_\theta(s_i),y_i)\right]
 +\frac{\lambda_{\mathrm{type}}}{|G|}\sum_{i\in G}
 D_{\mathrm{KL}}\!\left(\operatorname{sg}[\bar p_{550}(\cdot\mid s_i)]\,
 \Vert\,\bar p_\theta(\cdot\mid s_i)\right).
$$

Defaults: $B=8$, $|G|=6$, $\lambda_r=0.1$, $\lambda_{\mathrm{type}}=0.1$.
The latter is a configurable starting point, not a measured optimum.
The reference is the exact frozen **2B GenEvolve SFT-550 parent** identified
by its checkpoint file hashes; it is not the base model or the current student.
This stage-1 reference must not be confused with the unchanged stage-2 Teacher,
which is a no-gradient view of the current Actor.

## Native two-type probability, not a new classifier

No new action tokens or format conversion are introduced. Native IDs are
`<tool_call>` = `[151657]`, and distinguishing prefix `<answer` = `[27,9217]`.
The following `>` may merge with a newline or JSON delimiter; it is deliberately
not part of the type event. Thus we regularize the **distinguishing prefixes**,
not the probability of an entire valid action tag/body.

$$
u_T(s)=\pi_\theta(151657\mid s),\qquad
u_A(s)=\pi_\theta(27\mid s)\,\pi_\theta(9217\mid s,27),\qquad
\bar p_\theta(T,A\mid s)=\frac{(u_T(s),u_A(s))}{u_T(s)+u_A(s)}.
$$

`<` alone is NOT Answer. The two-type normalization is intentional for this
KL only; route BCE still uses the **full vocabulary**:

$$q_\theta(s)=\operatorname{softmax}(z_\theta(s))_{\texttt{<RETHINK>}}.$$

We compute the type log-probabilities as
`log_softmax([z_tool, z_open + log_softmax(z_after_open)[answer]])`.
The common first-position vocabulary normalizer cancels, so this KL has no
direct gradient to the first-position RETHINK logit. Shared network parameters
can still interact; the second-position vocabulary normalizer is retained.
No claim of complete independence from marker learning is made.

## Boundaries, masks and averaging

- Good: `H + <think>R</think>\n` conditions both policies. No marker, reference
  output, target action, future feedback, or extra evidence is added to it.
- Bad: the original marker/CE/BCE remain untouched; **zero type-KL loss**.
- Both Good Tool and Good Answer rows participate (1,290 and 492 respectively).
  All 1,782 Good records contain nonempty reasoning, but their reasoning token
  positions remain masked. KL does not supervise those positions.
- Under four-rank sample-mean accumulation, each Good contribution is scaled
  by $B/|G|=8/6$. This produces a mean over Good, not an extra factor of $6/8$.
- The student executes an additional `H/R + <` differentiable forward on every
  sample/rank. Bad multiplies the connected KL graph by zero; it does not skip
  FSDP calls or collective backward operations conditionally.
- Setting `good_type_kl_lambda=0` restores the original forward graph and
  CE+BCE objective and skips reference preparation. Use a fresh `WORK` when
  changing any configuration; one epoch remains 297 updates.

## Reference cache and provenance

Four ranks score disjoint Good prefixes with the frozen BF16 parent in eval
mode, before constructing the training actor. A causal forward on `s + <`
provides both the last-prefix and after-`<` logits. Only two float32
log-probabilities per Good are retained. The full reference is then unloaded.
Progress saves every 25 samples; completed shards have SHA-256 receipts.
Cache identity includes parent/data fingerprints, the complete training
configuration, native token IDs, implementation version, dtype and backend.
Changed identities, incomplete coverage, duplicates or corrupted completed
shards fail closed instead of silently using a wrong teacher.

Metrics retain original `suffix_ce`, `route_bce`, and Good/Bad marker
probabilities. Added fields are `good_type_kl`, `weighted_type_kl`,
`good_student_p_answer`, `good_reference_p_answer`; type probabilities are
conditional on Tool/Answer and must not be interpreted as natural finish rates.

## Unchanged components and limits

The sealed reviewed-v3 public data and batch order, suffix CE normalization,
route BCE, LR/optimizer, parameter freezing, 12,288 context limit, online
prompts/history/decoder/parser, generation settings, GRPO+SDL, hint policy,
renderer and evaluation remain those of the original reference commit.
There is no new inference-time constraint or API in stage 1.

This protects a distribution at **teacher-forced Good prefixes** only.
It does not by itself guarantee free-generation reasoning, better final images,
termination on unseen histories, or stable later RL. Verify these with matched
natural tasks and an identical parent/config/budget. Only CPU checks are claimed
by this release unless a separate actual GPU run is explicitly documented.
