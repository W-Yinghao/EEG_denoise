# Information-theory bridge

Let the frozen EEGNet task head be `Wz+b`, and let `P` remove the common-logit
direction. V34P defines `C=PW`. The representation is decomposed with the SVD of
`C` into its row-space component `z_H` and nullspace coordinates `u`:

\[
z=z_H+Nu,\qquad CN=0,\qquad Cz_H=Cz.
\]

Fiber methods replace only `u`. Consequently every released representation
`z'=z_H+Nu'` has the same centered logits, softmax probabilities, and fixed-head
decision as the input. This is an engineering identity, not an estimate of mutual
information.

The H-only attack is an empirical leakage floor for this exact-function-preserving
channel: information already transmitted by the frozen task head cannot be removed
by changing its fiber. Attacker accuracy is not described as conditional mutual
information, and the method makes no anonymity or causal-nuisance claim.
