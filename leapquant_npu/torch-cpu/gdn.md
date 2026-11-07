# GDN (Gated DeltaNet): The Recurrent State in Plain Terms

This note explains what the recurrent **state** is and why GDN needs it, using
a simple mental model rather than the full derivation.

## One-line idea

GDN replaces the ever-growing attention cache with one fixed-size matrix
`S` (the state) that is updated in place. It does not remember every token; it
keeps a running summary of key-to-value associations.

## What the state does

`S` is a small matrix. Think of it as a lookup table built from everything seen
so far, but compressed to a fixed size. At each new token the model does three
things, in order:

1. **Forget (gate).** Fade the old memory by a learned scalar `alpha_t`
   (between 0 and 1).
2. **Read.** See what the current key `k_t` would already predict from the
   faded memory: `alpha_t S_{t-1} k_t`.
3. **Write (delta rule).** Store only the *difference* between the real value
   and that prediction, so an old entry is edited instead of duplicated.

$$S_t = \alpha_t S_{t-1} + \beta_t\,(v_t - \alpha_t S_{t-1} k_t)\,k_t^\top$$

The answer to a query is then just the state multiplied by the query:
`o_t = S_t q_t`.

## The whiteboard metaphor

- **Softmax attention** is a stenographer who keeps every line ever said in a
  notebook that grows without bound. To answer a question, they re-read the
  whole notebook.
- **GDN's state** is a single fixed-size whiteboard. Before writing each new
  fact, the writer turns one **overall dimmer knob** (`alpha_t`) that fades
  everything on the board by the same amount, then writes the new fact. The
  board never gets bigger.
- The **delta rule** is like updating a contact's phone number: you first look
  up that name, erase the old number, and write the new one, rather than
  scribbling a second number next to it.

The single dimmer knob is the key GDN signature: one scalar gate controls how
much of the whole board is retained at each step.

## Why it matters for LeapQuant

The state is the only long-term memory in these layers, so it is exactly what
LeapQuant compresses. GDN keeps the state in fp32 and updates it every step;
LeapQuant quantizes it once per window instead, shrinking its storage while
the whiteboard's contents stay effectively the same.

## GDN vs KDA

GDN uses one scalar gate for the whole state. KDA gives each channel its own
decay (see `kda.md`); GDN is the special case where all those channel gates
happen to be equal.
