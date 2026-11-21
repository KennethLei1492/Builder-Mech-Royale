/**
 * Learned tactic model: a small MLP that, for the current state and each tactic (fight / kite / cover / push),
 * predicts the next 10 s: [logit P(we die), expected eliminations, HP change as a fraction of max HP].
 * Trained offline (sim/train_tactics.py) on simulated matches across many randomised, real-calibrated house-bot
 * configurations; weights live in model_weights.ts. Without weights it always answers 0 (the hand-written fight).
 */
import { WEIGHTS } from "./model_weights.ts"

type Layer = { W: number[][]; b: number[] }
type Model = { mean: number[]; std: number[]; layers: Layer[]; nActions: number } | null

const M = WEIGHTS as Model

function forward(x: number[]): number[] {
  let h = x
  const L = M!.layers
  for (let li = 0; li < L.length; li++) {
    const { W, b } = L[li]
    const out = new Array(b.length)
    for (let j = 0; j < b.length; j++) {
      let s = b[j]
      const row = W[j]
      for (let i = 0; i < h.length; i++) s += row[i] * h[i]
      out[j] = li < L.length - 1 ? (s > 0 ? s : 0.01 * s) : s
    }
    h = out
  }
  return h
}

/** predicted [pDeath, kills, hpDelta] for features f and tactic a */
export function predict(f: number[], a: number): number[] {
  const x = f.map((v, i) => (v - M!.mean[i]) / M!.std[i])
  for (let k = 0; k < M!.nActions; k++) x.push(k === a ? 1 : 0)
  const y = forward(x)
  return [1 / (1 + Math.exp(-y[0])), y[1], y[2], (y[3] ?? 0) * 15]
}

/**
 * usePts: score tactics by predicted final placement points; else by kills - lambda*P(death) + mu*dHP.
 * margin: only leave tactic 0 (the hand-written fight) when another tactic is predicted better by this much.
 */
export function chooseTactic(f: number[], lambda: number, mu: number, usePts = 0, margin = 0): number {
  if (!M) return 0
  const score = (a: number) => {
    const [pd, k, hp, pts] = predict(f, a)
    return usePts ? pts : k - lambda * pd + mu * hp
  }
  const s0 = score(0)
  let best = 0,
    bs = s0 + margin
  for (let a = 1; a < M.nActions; a++) {
    const s = score(a)
    if (s > bs) {
      bs = s
      best = a
    }
  }
  return best
}
