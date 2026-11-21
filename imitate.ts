/**
 * Imitation policy for combat movement, learned from the top-16 teams' bots in recorded live showdowns
 * (sim/live/extract.ts -> sim/train_imitate.py). The same feature function runs offline on replays and live in the
 * bot, so training and play see identical inputs.
 *
 * Output: a distribution over 5 movement choices relative to the nearest visible enemy
 * (0 hold, 1 approach, 2 retreat, 3 strafe left, 4 strafe right) and a boost probability.
 */
import type { World } from "./world.ts"
import { IMIT_WEIGHTS } from "./imitate_weights.ts"

export type ImitEnemy = { x: number; y: number; z: number; yaw: number; hp: number; maxHp: number; hasLaser: boolean; charging: boolean; hitMeRecently: boolean }
export type ImitSelf = {
  x: number; y: number; z: number; hp: number; maxHp: number; energy: number; cubes: number
  weapons: string[]; charging: boolean; taken3: number; dealt3: number
  zone: number[]; t: number; alive: number
}

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)
const angd = (a: number, b: number) => {
  let d = Math.abs(a - b) % (2 * Math.PI)
  return d > Math.PI ? 2 * Math.PI - d : d
}

export function imitationFeatures(me: ImitSelf, vis: ImitEnemy[], W: World): { f: number[]; near: ImitEnemy } {
  const ds = vis.map((e) => Math.hypot(e.x - me.x, e.y - me.y))
  let ni = 0
  for (let i = 1; i < vis.length; i++) if (ds[i] < ds[ni]) ni = i
  const near = vis[ni]
  const nd = ds[ni]
  let second = 1
  for (let i = 0; i < vis.length; i++) if (i !== ni) second = Math.min(second, ds[i] / 250)
  const bearNearToMe = Math.atan2(me.y - near.y, me.x - near.x)
  // cover from the nearest enemy: share of 8 spots 8 m around us its eye cannot see
  let cover = 0
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2
    const x = me.x + Math.cos(a) * 8, y = me.y + Math.sin(a) * 8
    if (W.clearAt(x, y) < 2.5) continue
    if (W.ray(near.x, near.y, near.z + 8, x, y, W.floorAt(x, y) + 5, 2) < 1) cover++
  }
  const zr = me.zone[2] || 1
  const zd = Math.hypot(me.x - me.zone[0], me.y - me.zone[1])
  const toZone = Math.atan2(me.zone[1] - me.y, me.zone[0] - me.x)
  const toNear = Math.atan2(near.y - me.y, near.x - me.x)
  const f = [
    me.hp / Math.max(1, me.maxHp),
    me.maxHp / 1400 - 1,
    clamp(me.energy / 100, 0, 1.6),
    Math.min(me.cubes, 10) / 10,
    me.weapons.includes("laser") ? 1 : 0,
    me.weapons.includes("lightning") ? 1 : 0,
    me.weapons.includes("mortar") ? 1 : 0,
    me.charging ? 1 : 0,
    Math.min(vis.length, 8) / 5,
    Math.min(nd, 250) / 250,
    near.hp / Math.max(1, near.maxHp),
    near.hasLaser ? 1 : 0,
    angd(near.yaw, bearNearToMe) < 0.15 ? 1 : 0,
    near.charging ? 1 : 0,
    near.hitMeRecently ? 1 : 0,
    second,
    Math.min(3, ds.filter((d) => d < 60).length) / 3,
    vis.reduce((s, e) => s + e.hp / Math.max(1, e.maxHp), 0) / vis.length,
    Math.min(me.taken3, 1400) / 1400,
    Math.min(me.dealt3, 1400) / 1400,
    clamp((me.z - W.floorAt(me.x, me.y)) / 20, 0, 2),
    Math.min(W.clearAt(me.x, me.y), 20) / 20,
    clamp((zr - zd) / Math.max(30, zr), -1, 1),
    me.t / 480,
    me.alive / 16,
    clamp((near.z - me.z) / 20, -1, 1),
    cover / 8,
    Math.cos(toZone - toNear),
  ]
  return { f, near }
}

type Layer = { W: number[][]; b: number[] }
type Model = { mean: number[]; std: number[]; layers: Layer[] } | null
const M = IMIT_WEIGHTS as Model
export const hasImitModel = () => !!M

/** [p(hold), p(approach), p(retreat), p(strafe left), p(strafe right), p(boost)] */
export function imitate(f: number[]): number[] | null {
  if (!M) return null
  let h = f.map((v, i) => (v - M.mean[i]) / M.std[i])
  for (let li = 0; li < M.layers.length; li++) {
    const { W, b } = M.layers[li]
    const o = new Array(b.length)
    for (let j = 0; j < b.length; j++) {
      let s = b[j]
      const row = W[j]
      for (let i = 0; i < h.length; i++) s += row[i] * h[i]
      o[j] = li < M.layers.length - 1 ? (s > 0 ? s : 0.01 * s) : s
    }
    h = o
  }
  const mx = Math.max(...h.slice(0, 5))
  const ex = h.slice(0, 5).map((v) => Math.exp(v - mx))
  const z = ex.reduce((a, b) => a + b, 0)
  return [...ex.map((v) => v / z), 1 / (1 + Math.exp(-h[5]))]
}

/** outcome model (sim/train_value.py): best of hold/approach/retreat/strafe by predicted adv - lam * P(die) */
export function valueChoice(f: number[]): number {
  const V = IMIT_WEIGHTS as any
  if (!V || !V.nActions) return -1
  let best = -1, bs = -1e9
  for (let a = 0; a < V.nActions; a++) {
    let h = f.map((v: number, i: number) => (v - V.mean[i]) / V.std[i])
    for (let k = 0; k < V.nActions; k++) h.push(k === a ? 1 : 0)
    for (let li = 0; li < V.layers.length; li++) {
      const { W, b } = V.layers[li]
      const o = new Array(b.length)
      for (let j = 0; j < b.length; j++) {
        let s = b[j]
        for (let i = 0; i < h.length; i++) s += W[j][i] * h[i]
        o[j] = li < V.layers.length - 1 ? (s > 0 ? s : 0.01 * s) : s
      }
      h = o
    }
    const v = h[1] - V.lam / (1 + Math.exp(-h[0]))
    if (v > bs) { bs = v; best = a }
  }
  return best
}
