# KDA (Kimi Delta Attention): The Recurrent State in Plain Terms

KDA, short for **Kimi Delta Attention**, is an extension of Gated DeltaNet.
Like GDN it keeps one fixed-size recurrent state `S`; the difference is how
that memory is forgotten.

## One-line idea

KDA replaces GDN's single scalar forgetting gate with a **per-channel decay**,
so each memory channel can be retained or erased on its own schedule. This
uses the limited state memory more efficiently.

## What the state does

The state is still a fixed-size matrix holding compressed key-to-value
associations, and each token still follows forget, read, write. The change is
in step 1:

1. **Forget (channel-wise).** Instead of one scalar `alpha_t` for the whole
   matrix, KDA applies a diagonal decay `D_t` whose entries can differ per
   channel. Some channels keep their content; others fade quickly.
2. **Read.** The current key reads the decayed memory, as in GDN.
3. **Write (delta rule).** Only the residual between the target value and the
   read prediction is written, so entries are edited, not duplicated.

Conceptually, with the channel-wise decay in place of the scalar:

$$S_t = D_t S_{t-1} + \text{(delta-rule write for } k_t, v_t\text{)}$$

The output is again the state multiplied by the query, `o_t = S_t q_t`.

## The whiteboard metaphor

- Recall GDN's whiteboard with one **overall dimmer knob** that fades the
  entire board by the same amount each step.
- KDA upgrades that board so **each column (each channel) has its own dimmer
  knob**. The region holding durable facts can stay bright while the region
  holding throwaway details is dimmed aggressively, independently.
- It is the difference between a single room light and separate lights over
  each work zone: finer control means the fixed board is spent on the channels
  that actually need it.

This per-channel control is the key KDA signature and why it can hold more
useful associations than GDN within the same state size.

## Why it matters for LeapQuant

KDA appears in large hybrid models (such as Kimi K3) alongside full-attention
layers, and its state is the fixed-size long-term memory of those KDA layers.
LeapQuant compresses exactly this state once per window. Because KDA already
organizes memory by channel, the per-channel scales line up naturally with
LeapQuant's row/column quantization factors.

## KDA vs GDN

- GDN: one scalar gate decays the whole state uniformly.
- KDA: a diagonal, per-channel decay controls each channel separately.
- When all of KDA's channel factors are equal, it reduces to GDN.
