# What stage 1 optimizes

Let \(s_i=(H_i,R_i)\) be the stored actor-visible history and original reasoning, up to the **native decision boundary**. Good keeps its sampled action; Bad inserts a single `<RETHINK>` marker before the corrected action. Let \(y_i\in\{0,1\}\) be the route label (1 = Bad), and \(Y_i\) the supervised suffix including EOS and, for Bad, the marker. History and reasoning are conditioning tokens, not CE targets.

$$
\mathcal L=
\frac{1}{8}\sum_{i=1}^{8}\left[
 -\frac{1}{|Y_i|}\sum_{t=1}^{|Y_i|}\log \pi_\theta(Y_{i,t}\mid s_i,Y_{i,<t})
 +0.1\,\mathrm{BCE}\!\left(\pi_\theta(\texttt{<RETHINK>}\mid s_i),y_i\right)
\right]
+\frac{0.1}{6}\sum_{i\in G}
D_{\mathrm{KL}}\!\left(\mathrm{sg}[p_{550}(\cdot\mid s_i)]\,\Vert\,p_\theta(\cdot\mid s_i)\right).
$$

Here \(G\) contains all six Good rows in a fixed batch. BCE uses the **entire vocabulary**: marker logit minus log-sum-exp of all non-marker logits. Suffix CE is averaged separately per row before averaging rows. The marker on Bad is therefore both a suffix target and an independent boundary decision. There is no extra type BCE, opener reweighting, or reasoning-token KL.

### Three disjoint events, not a two-class softmax

For both the frozen parent and the student define

$$
p_T=\pi(\texttt{<tool\_call>}\mid s),\quad
p_A=\pi(\texttt{<}\mid s)\,\pi(\texttt{answer}\mid s,\texttt{<}),\quad
p_O=1-p_T-p_A .
$$

The KL uses the distribution \((p_T,p_A,p_O)\), **without renormalizing over Tool and Answer alone**. Tool is native token 151657. Answer is distinguished by the two-token prefix \([27,9217]\); `<` alone is not Answer. The remainder includes EOS, the marker, and other continuations. The implementation evaluates the complement in log space, combining first-token “neither Tool nor `<`” with “`<` then not `answer`”. These events concern action **openers**, not the validity or quality of the entire JSON program.

Only Good rows receive this KL. Both policies see the same prefix and a hypothetical `<` for the second probability; they do not receive the gold action or later observations as input to the KL. The teacher is the frozen **2B SFT-550**, not the online student or the 8B collector. Its detached three-way log-probabilities are cached per Good. A two-way cache or a cache from a different data/parent contract is not interchangeable. Bad retains the original CE and BCE with no type KL.

Two epochs replay the same sealed 281 batches. Learning-rate warmup and cosine decay span all **562** updates, so the midpoint is not equivalent to a separate 281-step run whose learning rate already reached zero. Saving a resumable state at the midpoint does not trigger evaluation. At deployment no classifier, KL branch or reference model is invoked. This regularizer preserves a teacher-forced Good boundary distribution; it does **not** guarantee a particular free-generation RETHINK rate, finish rate or benchmark score.
