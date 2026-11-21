/**
 * Survival-first mech pilot.
 *
 * Layers, evaluated every decision (10 Hz):
 *   1. perception   – track every mech/NPC seen (pos, vel, hp, weapons), threats from damage events,
 *                     sounds, beams and incoming shells.
 *   2. weapons      – per-slot fire control: gun with intercept lead, laser charge/release, lightning cone,
 *                     ballistic mortar; loot swaps by loadout value; upgrade picks by a priority table.
 *   3. intent       – zone / fight / retreat-to-cover / regen / loot / patrol, chosen from HP, enemy count
 *                     and zone timing.
 *   4. locomotion   – A* path on the 2 m nav grid + context steering (wall clearance, shells, strafing),
 *                     stuck recovery with jump-jets.
 */
import type { ActorObs, Bot, BotAction, Hello, Observation, WeaponObs } from "./mechfight.ts"
import { draw } from "./mechfight.ts"
import { angDiff, clamp, hyp2, mulberry32, wrap, yawTo } from "./util.ts"
import { getWorld, type World } from "./world.ts"

type Track = {
  id: number
  kind: "mech" | "npc"
  pos: number[]
  vel: number[]
  aim: number[]
  hp: number
  maxHp: number
  cubes: number
  weapons: string[]
  charging: boolean
  seenT: number // last time it was in obs.mechs
  firstT: number
  lastHitMeT: number
  dmgToMe: number
}

type Mode = "zone" | "fight" | "retreat" | "regen" | "loot" | "patrol" | "drop" | "avoid"


const UPGRADE_SCORE: Record<string, number> = {
  armor: 10,
  plating: 9.5,
  nanites: 7,
  lightning_range: 7.5,
  lightning_efficiency: 6.5,
  lightning_chain: 4,
  laser_charge: 7,
  laser_damage: 7,
  laser_pierce: 2,
  gun_rate: 6.5,
  gun_damage: 6.5,
  gun_velocity: 5,
  gun_mag: 5,
  gyros: 5.5,
  servos: 5,
  sensors: 4.5,
  capacitor: 4,
  reactor: 4,
  jets: 3.5,
  mortar_reload: 2,
  mortar_splash: 2,
  mortar_velocity: 2,
}

/** tunables (overridable for local experiments via MF_TUNE='{"k":v}') */
export const TUNE: Record<string, number> = {
  retreatHp1: 0.3, // alone-ish: retreat below this hp fraction if the enemy is healthier
  retreat3: 0.75, // 3+ engaged
  retreat2: 0.5, // 2+ engaged
  powerRatio: 0.5,
  noRetreat: 1,
  rangeGun: 55,
  rangeLightning: 24,
  rangeLaser: 130,
  regenHp: 0.85,
  grabDist: 50,
  strafeMin: 0.45,
  strafeVar: 0.9,
  tang: 0.85,
  fightBoost: 0,
  vLightning: 2.2,
  vLaser: 3.4,
  vGun: 2.9,
  vMortar: 1.1,
  hpWeight: 70,
  zapReserve: 6,
}
try {
  const env = (globalThis as any).process?.env?.MF_TUNE
  if (env) Object.assign(TUNE, JSON.parse(env))
} catch {}

export function createBot(hello: Hello): Bot {
  return new Brain(hello)
}

export class Brain implements Bot {
  name = "ironclad"
  W: World
  cfg: Hello["config"]
  rnd: () => number
  slot: number
  tracks = new Map<number, Track>()
  t = 0
  tick = 0
  // self
  me!: Observation["self"]
  px = 0
  py = 0
  pz = 0
  sf = 1
  muzzle = 7
  eye = 8
  radius = 2.5
  // threat memory
  lastDamageT = -99
  lastDamageBearing = 0
  lastDamageBy: number | null = null
  dmgWindow: { t: number; amt: number }[] = []
  soundHint: { t: number; bearing: number; dist: string; kind: string } | null = null
  shells: { x: number; y: number; t: number }[] = []
  laserThreat = false
  // navigation
  path: number[][] | null = null
  pathGoal: number[] | null = null
  pathT = -99
  stuckT = 0
  lastMoveCmd = 0
  unstickUntil = -1
  unstickDir = 0
  // fight
  targetId = -1
  strafeSign = 1
  strafeFlipT = 0
  mode: Mode = "patrol"
  modeSince = 0
  coverPt: number[] | null = null
  coverT = -99
  lastFireT = -99
  scanYaw = 0
  takeCooldownUntil = -1
  badPickups = new Set<number>()
  crateTarget = -1
  dbg: string[] = []

  constructor(hello: Hello) {
    this.cfg = hello.config
    this.slot = hello.slot
    this.rnd = mulberry32(hello.seed ^ 0x9e3779b9)
    this.W = getWorld(hello.map.dir)
  }

  // ======================================================================= main
  act(obs: Observation): BotAction {
    this.t = obs.t
    this.tick = obs.tick
    this.me = obs.self
    const me = obs.self
    ;[this.px, this.py, this.pz] = me.pos
    this.sf = 1 + (this.cfg.cubes.size_per_cube as number) * me.cubes
    this.muzzle = me.stats.muzzle_height ?? this.cfg.mech.muzzle_height * this.sf
    this.eye = me.stats.eye_height ?? this.cfg.mech.eye_height * this.sf
    this.radius = me.stats.radius ?? this.cfg.mech.radius * this.sf
    this.dbg.length = 0
    this.perceive(obs)

    const a: BotAction = {}
    if (obs.offer) a.pick = this.pickUpgrade(obs)
    this.loot(obs, a)

    const target = this.chooseTarget(obs)
    this.decideMode(obs, target)
    const aimFire = this.weapons(obs, target, a)
    this.locomotion(obs, target, a)
    if (!aimFire) a.aim = this.idleAim(obs, a)
    // reload when quiet
    const g = this.slotOf("gun")
    if (g >= 0) {
      const w = me.weapons[g]!
      if (!target && (w.reload ?? 0) === 0 && (w.ammo ?? 0) < (w.stats.mag ?? 12) * 0.6) a.reload = true
    }
    draw.label(`${this.mode} hp${Math.round(me.hp)} e${Math.round(me.energy)} ${target ? "T" + target.id : ""} ${this.dbg.join(" ")}`.slice(0, 78))
    return a
  }

  // ======================================================================= perception
  perceive(obs: Observation) {
    const t = this.t
    for (const m of obs.mechs) {
      let tr = this.tracks.get(m.id)
      if (!tr) {
        tr = { id: m.id, kind: m.kind, pos: m.pos, vel: m.vel, aim: m.aim, hp: m.hp ?? 1400, maxHp: m.max_hp ?? 1400, cubes: m.cubes ?? 0, weapons: m.weapons ?? [], charging: !!m.charging, seenT: t, firstT: t, lastHitMeT: -99, dmgToMe: 0 }
        this.tracks.set(m.id, tr)
      }
      tr.pos = m.pos
      tr.vel = m.vel
      tr.aim = m.aim
      if (m.hp !== undefined) tr.hp = m.hp
      if (m.max_hp !== undefined) tr.maxHp = m.max_hp
      if (m.cubes !== undefined) tr.cubes = m.cubes
      if (m.weapons) tr.weapons = m.weapons
      tr.charging = !!m.charging
      tr.seenT = t
    }
    // forget the dead / very stale
    for (const [id, tr] of this.tracks) if (t - tr.seenT > 60) this.tracks.delete(id)
    this.laserThreat = false
    for (const e of obs.events) {
      if (e.type === "damaged") {
        if (e.weapon !== "zone") {
          this.lastDamageT = t
          if (e.bearing !== null) this.lastDamageBearing = e.bearing
          this.lastDamageBy = e.by
          if (e.by !== null) {
            const tr = this.tracks.get(e.by)
            if (tr) {
              tr.lastHitMeT = t
              tr.dmgToMe += e.amount
            }
          }
        }
        this.dmgWindow.push({ t, amt: e.amount })
      } else if (e.type === "kill") {
        this.tracks.delete(e.target)
        if (this.targetId === e.target) this.targetId = -1
      } else if (e.type === "take_failed") {
        this.badPickups.add(e.id)
      }
    }
    while (this.dmgWindow.length && t - this.dmgWindow[0].t > 3) this.dmgWindow.shift()
    // sounds: remember the most interesting recent one (weapon fire > steps)
    let best: { t: number; bearing: number; dist: string; kind: string } | null = null
    let bs = -1
    for (const s of obs.sounds) {
      const pr = s.kind === "step" || s.kind === "boost" || s.kind === "jet" ? 1 : s.kind === "death" || s.kind === "crate" || s.kind === "drop" || s.kind === "npc_gun" ? 0 : 2
      const sc = pr * 3 + (s.dist === "near" ? 2 : s.dist === "mid" ? 1 : 0)
      if (pr > 0 && sc > bs) {
        bs = sc
        best = { t: t - s.age, bearing: s.bearing, dist: s.dist, kind: s.kind }
      }
    }
    if (best) this.soundHint = best
    // incoming shells: predict impact points
    this.shells.length = 0
    const gM = this.cfg.weapons.mortar.gravity ?? 22
    for (const p of obs.projectiles) {
      if (p.kind !== "mortar") continue
      let [x, y, z] = p.pos
      let [vx, vy, vz] = p.vel
      for (let s = 0; s < 80; s++) {
        const dt = 0.05
        x += vx * dt
        y += vy * dt
        vz -= gM * dt
        z += vz * dt
        if (this.W.solid(x, y, z) || z < this.W.groundAt(x, y)) break
      }
      this.shells.push({ x, y, t })
    }
    // laser telegraph aimed at me
    for (const tr of this.tracks.values()) {
      if (tr.kind !== "mech" || !tr.charging || t - tr.seenT > 0.15) continue
      const yawToMe = yawTo(tr.pos[0], tr.pos[1], this.px, this.py)
      if (angDiff(tr.aim[0], yawToMe) < 0.12) this.laserThreat = true
    }
  }

  enemies(maxAge = 0.25, maxDist = 1e9): Track[] {
    const out: Track[] = []
    for (const tr of this.tracks.values())
      if (tr.kind === "mech" && this.t - tr.seenT <= maxAge && hyp2(this.px, this.py, tr.pos[0], tr.pos[1]) <= maxDist) out.push(tr)
    return out
  }

  // ======================================================================= targeting
  muzzlePos(): number[] {
    return [this.px, this.py, this.pz + this.muzzle]
  }
  centerOf(tr: Track): number[] {
    const sf = 1 + (this.cfg.cubes.size_per_cube as number) * (tr.cubes || 0)
    const h = tr.kind === "npc" ? this.cfg.npc.height ?? 3.5 : this.cfg.mech.height * sf
    return [tr.pos[0], tr.pos[1], tr.pos[2] + h * 0.5]
  }
  canHit(tr: Track): boolean {
    const m = this.muzzlePos()
    const c = this.centerOf(tr)
    if (this.W.ray(m[0], m[1], m[2], c[0], c[1], c[2], 1) >= 0.97) return true
    const top = tr.kind === "npc" ? 3 : 7.5
    return this.W.ray(m[0], m[1], m[2], c[0], c[1], tr.pos[2] + top, 1) >= 0.97
  }

  chooseTarget(obs: Observation): Track | null {
    const ens = this.enemies(0.25)
    let best: Track | null = null
    let bs = 1e9
    for (const tr of ens) {
      const d = hyp2(this.px, this.py, tr.pos[0], tr.pos[1])
      if (d > 280) continue
      const los = this.canHit(tr)
      let s = d + (tr.hp / Math.max(1, tr.maxHp)) * TUNE.hpWeight
      if (!los) s += 220
      if (this.t - tr.lastHitMeT < 2) s -= 35
      if (tr.id === this.targetId) s -= 30
      if (tr.weapons.includes("laser") && tr.charging) s -= 20
      if (s < bs) {
        bs = s
        best = tr
      }
    }
    if (best) {
      this.targetId = best.id
      return best
    }
    this.targetId = -1
    return null
  }

  /** NPC or crate to farm when no mech is around */
  farmTarget(obs: Observation): { pos: number[]; id: number; kind: "npc" | "crate"; hp: number } | null {
    if (this.enemies(1.5, 200).length) return null
    if (this.me.hp < this.me.max_hp * 0.45) return null
    let best: { pos: number[]; id: number; kind: "npc" | "crate"; hp: number } | null = null
    let bd = 1e9
    for (const m of obs.mechs) {
      if (m.kind !== "npc") continue
      const d = hyp2(this.px, this.py, m.pos[0], m.pos[1])
      if (d > 110 || !this.inSafeZone(m.pos[0], m.pos[1], 10)) continue
      const tr = this.tracks.get(m.id)
      if (!tr || !this.canHit(tr)) continue
      if (d < bd) {
        bd = d
        best = { pos: m.pos, id: m.id, kind: "npc", hp: m.hp ?? 350 }
      }
    }
    for (const c of obs.crates) {
      const d = hyp2(this.px, this.py, c.pos[0], c.pos[1])
      if (d > 90 || d + 15 > bd || !this.inSafeZone(c.pos[0], c.pos[1], 10)) continue
      const m = this.muzzlePos()
      if (this.W.ray(m[0], m[1], m[2], c.pos[0], c.pos[1], c.pos[2] + 1.2, 1) < 0.97) continue
      bd = d + 15
      best = { pos: [c.pos[0], c.pos[1], c.pos[2] - 3.3], id: c.id, kind: "crate", hp: c.hp ?? 300 }
    }
    return best
  }

  // ======================================================================= weapons
  slotOf(kind: string): number {
    const w = this.me.weapons
    if (w[0]?.kind === kind) return 0
    if (w[1]?.kind === kind) return 1
    return -1
  }

  /** gun intercept: returns [yaw, pitch, tof] */
  leadAim(tpos: number[], tvel: number[], zOff: number, speed: number, latency: number): number[] {
    const m = this.muzzlePos()
    let px = tpos[0] + tvel[0] * latency,
      py = tpos[1] + tvel[1] * latency,
      pz = tpos[2] + zOff
    let tof = 0
    const vz = Math.abs(tvel[2]) > 0.5 ? tvel[2] : 0
    for (let i = 0; i < 4; i++) {
      tof = Math.hypot(px - m[0], py - m[1], pz - m[2]) / speed
      const T = latency + tof
      px = tpos[0] + tvel[0] * T
      py = tpos[1] + tvel[1] * T
      pz = tpos[2] + zOff + (vz ? vz * T - 0.5 * this.cfg.mech.gravity * T * T : 0)
      const f = this.W.floorAt(px, py)
      if (pz < f + zOff * 0.5) pz = f + zOff * 0.8
    }
    const yaw = Math.atan2(py - m[1], px - m[0])
    const pitch = Math.atan2(pz - m[2], Math.hypot(px - m[0], py - m[1]))
    return [yaw, pitch, tof, px, py, pz]
  }

  /** ballistic mortar solution: [yaw, pitch, tof] or null */
  mortarAim(tpos: number[], tvel: number[], st: Record<string, number>): number[] | null {
    const v = st.speed ?? 72
    const g = st.gravity ?? 22
    const oz = this.pz + this.cfg.mech.height * this.sf + 0.5
    let px = tpos[0],
      py = tpos[1]
    const pz = tpos[2] + 0.5
    let sol: number[] | null = null
    for (let it = 0; it < 3; it++) {
      const d = hyp2(this.px, this.py, px, py)
      const h = pz - oz
      const A = (g * d * d) / (2 * v * v)
      const disc = d * d - 4 * A * (A + h)
      if (disc < 0 || d < 1) return null
      const sq = Math.sqrt(disc)
      const tans = [(d - sq) / (2 * A), (d + sq) / (2 * A)]
      sol = null
      for (const T of tans) {
        const pitch = Math.atan(T)
        if (pitch < this.cfg.mech.min_pitch || pitch > this.cfg.mech.max_pitch) continue
        const yaw = Math.atan2(py - this.py, px - this.px)
        const tof = d / (v * Math.cos(pitch))
        // trajectory clear?
        let ok = true
        const n = 14
        for (let s = 1; s < n; s++) {
          const tt = (tof * s) / n
          const x = this.px + Math.cos(yaw) * v * Math.cos(pitch) * tt
          const y = this.py + Math.sin(yaw) * v * Math.cos(pitch) * tt
          const z = oz + v * Math.sin(pitch) * tt - 0.5 * g * tt * tt
          if (this.W.solid(x, y, z)) {
            ok = false
            break
          }
        }
        if (ok) {
          sol = [yaw, pitch, tof]
          break
        }
      }
      if (!sol) return null
      px = tpos[0] + tvel[0] * (sol[2] + 0.05)
      py = tpos[1] + tvel[1] * (sol[2] + 0.05)
    }
    return sol
  }

  /** sets a.fire/a.aim; returns true if it set the aim */
  weapons(obs: Observation, target: Track | null, a: BotAction): boolean {
    const me = this.me
    const fire: [boolean, boolean] = [false, false]
    a.fire = fire
    let aimSet = false
    const curYaw = me.aim[0],
      curPitch = me.aim[1]
    let tgt: { pos: number[]; vel: number[]; zOff: number; id: number; isMech: boolean; tr: Track | null } | null = null
    if (target && this.canHit(target)) {
      const c = this.centerOf(target)
      tgt = { pos: target.pos, vel: target.vel, zOff: c[2] - target.pos[2], id: target.id, isMech: true, tr: target }
    } else if (!target) {
      const f = this.farmTarget(obs)
      if (f) {
        const tr = f.kind === "npc" ? this.tracks.get(f.id) ?? null : null
        tgt = { pos: f.pos, vel: tr ? tr.vel : [0, 0, 0], zOff: f.kind === "npc" ? 1.75 : 4.5, id: f.id, isMech: false, tr }
        this.crateTarget = f.id
      }
    }
    // laser: keep holding if charging and nothing to shoot (it auto-fires at hold_s; we release when aimed)
    const ls = this.slotOf("laser")
    const lw = ls >= 0 ? me.weapons[ls]! : null
    if (!tgt) {
      if (lw && (lw.charge ?? 0) > 0) {
        // release into the last known enemy direction rather than wasting the charge randomly
        fire[ls] = (lw.charge ?? 0) < (lw.stats.charge ?? 0.75) + (lw.stats.hold_s ?? 2) - 0.3
      }
      return false
    }
    const d = hyp2(this.px, this.py, tgt.pos[0], tgt.pos[1])
    if (this.mode === "avoid" && tgt.isMech && d > 90 && !(lw && (lw.charge ?? 0) > 0)) return false
    const m = this.muzzlePos()
    // primary aim: gun lead if we have a gun, else direct
    const gs = this.slotOf("gun")
    const gw = gs >= 0 ? me.weapons[gs]! : null
    const gunSpeed = gw?.stats.speed ?? 252
    const [yaw, pitch] = this.leadAim(tgt.pos, tgt.vel, tgt.zOff, gunSpeed, 0.06)
    const direct = [Math.atan2(tgt.pos[1] - m[1], tgt.pos[0] - m[0]), Math.atan2(tgt.pos[2] + tgt.zOff - m[2], Math.max(0.5, d))]
    const hitscan = this.leadAim(tgt.pos, tgt.vel, tgt.zOff, 1e6, 0.07)
    let wantYaw = yaw,
      wantPitch = pitch
    // which weapon drives aim?
    const mo = this.slotOf("mortar")
    const mw = mo >= 0 ? me.weapons[mo]! : null
    const laserActive = lw && (lw.charge ?? 0) > 0
    if (laserActive || (!gw && lw)) {
      wantYaw = hitscan[0]
      wantPitch = hitscan[1]
    }
    let mortarSol: number[] | null = null
    if (mw && tgt.isMech && d > 28 && d < 230 && mw.cooldown <= 0.1) {
      mortarSol = this.mortarAim(tgt.pos, tgt.vel, mw.stats)
      // prefer the mortar aim only when nothing else can shoot right now
      const gunReady = gw && (gw.ammo ?? 0) > 0 && (gw.reload ?? 0) === 0
      if (mortarSol && !gunReady && !laserActive) {
        wantYaw = mortarSol[0]
        wantPitch = mortarSol[1]
      }
    }
    a.aim = [wantYaw, clamp(wantPitch, this.cfg.mech.min_pitch, this.cfg.mech.max_pitch)]
    aimSet = true
    const tolYaw = Math.max(0.006, Math.atan2(this.cfg.mech.radius * 0.8, Math.max(1, d)))
    const errYaw = angDiff(curYaw, wantYaw)
    const errPitch = Math.abs(curPitch - wantPitch)
    const maxTurn = (me.stats.turn_rate ?? 4) * 0.1
    // gun
    const slowTarget = Math.hypot(tgt.vel[0], tgt.vel[1]) < 3
    const gunRange = Math.min((gw?.stats.range ?? 230) * 0.98, slowTarget ? 1e9 : 175)
    if (gw && (gw.reload ?? 0) === 0 && (gw.ammo ?? 1) > 0 && d < gunRange) {
      const e1 = angDiff(curYaw, yaw)
      if (e1 < tolYaw + maxTurn * 0.15 && Math.abs(curPitch - pitch) < tolYaw * 2 + 0.05) fire[gs] = true
    }
    // lightning: instant, nearest target in cone with LOS
    const zs = this.slotOf("lightning")
    if (zs >= 0) {
      const zw = me.weapons[zs]!
      const range = zw.stats.range ?? 45
      const cone = ((zw.stats.cone_deg ?? 30) * Math.PI) / 180 / 2
      const reserve = this.mode === "retreat" ? 25 : TUNE.zapReserve
      let zapOk = d < range - 1 && angDiff(curYaw, direct[0]) < cone * 0.85
      if (!zapOk)
        for (const e of this.enemies(0.15, range - 1)) {
          if (angDiff(curYaw, yawTo(this.px, this.py, e.pos[0], e.pos[1])) < cone * 0.85 && this.canHit(e)) {
            zapOk = true
            break
          }
        }
      if (zapOk && me.energy >= (zw.stats.energy ?? 3) + reserve && zw.cooldown <= 0.1) fire[zs] = true
      if (d < range + 8) {
        // lightning doesn't care about lead; aim straight when it's the main weapon up close
        if (!gw || (gw.reload ?? 0) > 0) {
          a.aim = [direct[0], clamp(direct[1], this.cfg.mech.min_pitch, this.cfg.mech.max_pitch)]
        }
      }
    }
    // laser
    if (lw) {
      const ch = lw.charge ?? 0
      const full = lw.stats.charge ?? 0.75
      const inRange = d < (lw.stats.range ?? 280) * 0.97
      if (ch > 0) {
        const aimed = angDiff(curYaw, hitscan[0]) < tolYaw * 0.8 && Math.abs(curPitch - hitscan[1]) < tolYaw * 1.5 + 0.02
        const ready = ch >= full - 0.02
        const expiring = ch > full + (lw.stats.hold_s ?? 2) - 0.25
        // release (=false) when aimed and full, or when shorted out soon
        fire[ls] = !((ready && aimed) || (expiring && aimed))
        if (expiring && !aimed) fire[ls] = false
      } else if (lw.cooldown <= 0.05 && inRange && tgt.isMech) {
        fire[ls] = true // start charging
      } else if (lw.cooldown <= 0.05 && inRange && !tgt.isMech && !gw) {
        fire[ls] = true
      }
    }
    // mortar
    if (mw && mortarSol && mw.cooldown <= 0.05) {
      if (angDiff(curYaw, mortarSol[0]) < 0.03 && Math.abs(curPitch - mortarSol[1]) < 0.03) fire[mo] = true
      else if (!gw || !fire[gs]) {
        a.aim = [mortarSol[0], mortarSol[1]]
      }
    }
    if (fire[0] || fire[1]) this.lastFireT = this.t
    if (this.mode !== "fight" && this.mode !== "retreat" && !tgt.isMech) this.dbg.push("farm")
    void errYaw
    void errPitch
    return aimSet
  }

  idleAim(obs: Observation, a: BotAction): [number, number] {
    const t = this.t
    // most recent evidence of an enemy
    let yaw: number | null = null
    let recent: Track | null = null
    for (const tr of this.tracks.values()) if (tr.kind === "mech" && (!recent || tr.seenT > recent.seenT)) recent = tr
    if (t - this.lastDamageT < 3) yaw = this.lastDamageBearing
    else if (recent && t - recent.seenT < 6) yaw = yawTo(this.px, this.py, recent.pos[0], recent.pos[1])
    else if (this.soundHint && t - this.soundHint.t < 3) yaw = this.soundHint.bearing
    if (yaw === null) {
      // scan: sweep around the movement direction
      const mv = a.move
      const base = mv && Math.hypot(mv[0], mv[1]) > 0.2 ? Math.atan2(mv[1], mv[0]) : this.me.aim[0]
      this.scanYaw = wrap(this.scanYaw + 0.12)
      yaw = wrap(base + Math.sin(t * 0.9) * 1.6)
    }
    return [yaw, 0]
  }

  pickUpgrade(obs: Observation): number {
    const ch = obs.offer!.choices
    let bi = 0,
      bs = -1
    ch.forEach((c, i) => {
      let s = TUNE["up_" + c.id] ?? UPGRADE_SCORE[c.id] ?? 3
      if (c.id.startsWith("gun_") && this.slotOf("gun") < 0) s = 0.5
      if (c.id.startsWith("laser_") && this.slotOf("laser") < 0) s = 0.5
      if (c.id.startsWith("lightning_") && this.slotOf("lightning") < 0) s = 0.5
      if (c.id.startsWith("mortar_") && this.slotOf("mortar") < 0) s = 0.3
      const have = this.me.upgrades.filter((u) => u === c.id).length
      s -= have * 0.6
      if (s > bs) {
        bs = s
        bi = i
      }
    })
    return bi
  }

  // ======================================================================= loot
  wval(kind: string): number {
    return kind === "lightning" ? TUNE.vLightning : kind === "laser" ? TUNE.vLaser : kind === "gun" ? TUNE.vGun : kind === "mortar" ? TUNE.vMortar : 0
  }
  loadoutGain(kind: string): { gain: number; drop: number | null } {
    const w = this.me.weapons
    if (w.some((x) => x?.kind === kind)) return { gain: 0, drop: null }
    const v = this.wval(kind)
    if (!w[0]) return { gain: v, drop: null }
    if (!w[1]) return { gain: v, drop: null }
    let worst = 0
    let wv = 1e9
    for (let i = 0; i < 2; i++) {
      const x = w[i]!
      if (x.kind === "laser" && (x.charge ?? 0) > 0) continue
      const xv = this.wval(x.kind)
      if (xv < wv) {
        wv = xv
        worst = i
      }
    }
    if (wv === 1e9) return { gain: 0, drop: null }
    return { gain: v - wv - 0.4, drop: worst }
  }

  loot(obs: Observation, a: BotAction) {
    if (this.t < this.takeCooldownUntil) return
    const R = this.cfg.loot.pickup_radius
    for (const p of obs.pickups) {
      if (this.badPickups.has(p.id)) continue
      if (hyp2(this.px, this.py, p.pos[0], p.pos[1]) > R - 0.3 || Math.abs(p.pos[2] - this.pz) > 3.8) continue
      const { gain, drop } = this.loadoutGain(p.kind)
      if (gain <= 0) continue
      // don't swap mid-fight unless the new weapon is clearly better
      if (drop !== null && this.mode === "fight" && gain < 0.6) continue
      a.take = p.id
      if (drop !== null) a.drop = drop
      this.takeCooldownUntil = this.t + 0.25
      return
    }
  }

  // ======================================================================= zone
  zoneAt(z: Observation["zone"], t: number): number[] {
    if (t <= z.shrink_start) return [z.center[0], z.center[1], z.radius]
    if (t >= z.shrink_end) return [z.next_center[0], z.next_center[1], z.next_radius]
    const f = (t - z.shrink_start) / Math.max(0.01, z.shrink_end - z.shrink_start)
    return [z.center[0] + (z.next_center[0] - z.center[0]) * f, z.center[1] + (z.next_center[1] - z.center[1]) * f, z.radius + (z.next_radius - z.radius) * f]
  }
  /** the circle we must be inside soon (current, or next when the shrink is near) */
  safeCircle(): number[] {
    const z = this.obsZone
    const speed = this.me.stats.walk_speed ?? 11.7
    const dNext = Math.max(0, hyp2(this.px, this.py, z.next_center[0], z.next_center[1]) - z.next_radius * 0.8)
    const travel = dNext / speed
    if (this.t + travel + 18 > z.shrink_start) return [z.next_center[0], z.next_center[1], z.next_radius]
    const c = this.zoneAt(z, this.t + 5)
    return c
  }
  obsZone!: Observation["zone"]
  inSafeZone(x: number, y: number, margin: number): boolean {
    const c = this.safeCircle()
    return hyp2(x, y, c[0], c[1]) < c[2] - margin
  }

  // ======================================================================= intent
  /** enemies that are actually on us: hit us recently or are close */
  engagedSet(maxD = 130): Track[] {
    const out: Track[] = []
    for (const e of this.enemies(0.5, maxD)) {
      const d = hyp2(this.px, this.py, e.pos[0], e.pos[1])
      const closing = (e.vel[0] * (this.px - e.pos[0]) + e.vel[1] * (this.py - e.pos[1])) / Math.max(1, d) > 4
      if (this.t - e.lastHitMeT < 8 || d < 75 || (closing && d < 110)) out.push(e)
    }
    return out
  }

  decideMode(obs: Observation, target: Track | null) {
    this.obsZone = obs.zone
    const me = this.me
    const hpf = me.hp / me.max_hp
    const near = this.enemies(0.5, 150)
    const eng = this.engagedSet()
    const t = this.t
    let mode: Mode
    const sc = this.safeCircle()
    const dz = hyp2(this.px, this.py, sc[0], sc[1])
    const outside = dz > sc[2] - 4
    const zoneUrgent = (!me.in_zone && obs.zone.dps >= 20) || (outside && this.t > obs.zone.shrink_start - 10)
    let enemyPower = 0
    for (const e of eng) enemyPower += Math.max(150, e.hp) * this.threatDps(e)
    const myPower = me.hp * this.myDps() * (1 + (me.stats.armor ?? 0))
    let retreat = false
    if (eng.length >= 1) {
      const weakest = eng.reduce((a, b) => (a.hp < b.hp ? a : b))
      const finishing = eng.length <= 2 && weakest.hp < 260 && me.hp > 350 && hyp2(this.px, this.py, weakest.pos[0], weakest.pos[1]) < 90
      if (hpf < TUNE.retreatHp1 && (eng.length >= 2 || eng[0].hp > me.hp * 1.1)) retreat = true
      if (eng.length >= 3 && hpf < TUNE.retreat3) retreat = true
      if (eng.length >= 2 && hpf < TUNE.retreat2) retreat = true
      if (myPower < enemyPower * TUNE.powerRatio) retreat = true
      if (finishing && eng.length === 1) retreat = false
      if (TUNE.noRetreat) retreat = false
    } else if (t - this.lastDamageT < 1.5 && hpf < 0.35 && this.lastDamageBy !== null) retreat = true
    if (retreat) mode = "retreat"
    else if (eng.length) mode = "fight"
    else if (near.length) mode = "avoid"
    else if (zoneUrgent) mode = "zone"
    else if (hpf < TUNE.regenHp && t - this.lastDamageT < 30) mode = "regen"
    else mode = "patrol"
    if ((mode === "fight" || mode === "avoid") && zoneUrgent && !me.in_zone) mode = "zone"
    if (this.mode === "retreat" && mode !== "retreat" && t - this.modeSince < 1.5 && near.length) mode = "retreat"
    if (mode !== this.mode) {
      this.mode = mode
      this.modeSince = t
    }
    void target
  }

  threatDps(e: Track): number {
    let d = 90
    for (const w of e.weapons) d += w === "lightning" ? 120 : w === "laser" ? 110 : w === "mortar" ? 60 : 0
    return d
  }
  myDps(): number {
    let d = 0
    for (const w of this.me.weapons) if (w) d += w.kind === "lightning" ? 120 : w.kind === "laser" ? 110 : w.kind === "mortar" ? 60 : 90
    return d * (1 + 0.04 * Math.min(10, this.me.cubes))
  }

  // ======================================================================= locomotion
  locomotion(obs: Observation, target: Track | null, a: BotAction) {
    const me = this.me
    const t = this.t
    let goal: number[] | null = null
    let desire: number[] = [0, 0] // local steering intent (unit-ish), used when no path goal
    let boost = false
    let speedScale = 1
    const sc = this.safeCircle()
    const toSafe = () => {
      const d = hyp2(this.px, this.py, sc[0], sc[1])
      const r = sc[2] * 0.55
      if (d <= r) return null
      const k = r / d
      return [sc[0] + (this.px - sc[0]) * k, sc[1] + (this.py - sc[1]) * k]
    }
    switch (this.mode) {
      case "zone": {
        goal = toSafe() ?? [sc[0], sc[1]]
        boost = !me.in_zone && me.energy > 15
        break
      }
      case "fight": {
        const tr = target ?? this.enemies(0.3, 170)[0]
        if (!tr) break
        const d = hyp2(this.px, this.py, tr.pos[0], tr.pos[1])
        const R = this.preferredRange()
        const ux = (tr.pos[0] - this.px) / Math.max(1, d),
          uy = (tr.pos[1] - this.py) / Math.max(1, d)
        if (t > this.strafeFlipT) {
          this.strafeSign = this.rnd() < 0.5 ? -1 : 1
          this.strafeFlipT = t + TUNE.strafeMin + this.rnd() * TUNE.strafeVar
        }
        let radial = clamp((d - R) / Math.max(15, R * 0.5), -1, 1)
        if (!this.canHit(tr)) {
          // no line of fire: path toward it to get a shot (or wait for it if we hold lightning and it's close)
          goal = [tr.pos[0], tr.pos[1]]
          break
        }
        const tang = this.laserThreat ? 1 : TUNE.tang
        desire = [ux * radial + -uy * this.strafeSign * tang, uy * radial + ux * this.strafeSign * tang]
        // several on us: back away from the others while we work on this one
        const others = this.engagedSet().filter((e) => e.id !== tr.id)
        if (others.length) {
          let ax = 0,
            ay = 0
          for (const e of others) {
            const de = Math.max(5, hyp2(this.px, this.py, e.pos[0], e.pos[1]))
            ax += (this.px - e.pos[0]) / de
            ay += (this.py - e.pos[1]) / de
          }
          const n = Math.hypot(ax, ay) || 1
          desire[0] += (ax / n) * 0.9
          desire[1] += (ay / n) * 0.9
          if (radial > 0) {
            desire[0] -= ux * radial
            desire[1] -= uy * radial
          }
        }
        if (this.laserThreat) {
          // a laser is charging at us: duck behind the nearest cover if it's a step away
          const qc = this.quickCover()
          if (qc) {
            goal = qc
            this.dbg.push("duck")
          }
          if (me.energy > 20) boost = true
        }
        if (radial > 0.6 && this.slotOf("lightning") >= 0 && me.energy > 55 && d < 90) boost = true
        if (TUNE.fightBoost && me.energy > TUNE.fightBoost) boost = true
        // keep in the zone
        const dz = hyp2(this.px, this.py, sc[0], sc[1])
        if (dz > sc[2] - 15) {
          desire[0] += ((sc[0] - this.px) / dz) * 1.2
          desire[1] += ((sc[1] - this.py) / dz) * 1.2
        }
        break
      }
      case "avoid": {
        // enemies around that are not on us yet: slip out of their sight, don't walk into them
        const cp = this.findCover(sc, true)
        if (cp && hyp2(this.px, this.py, cp[0], cp[1]) > 3) goal = cp
        break
      }
      case "retreat": {
        const cp = this.findCover(sc)
        if (cp) goal = cp
        else {
          const ens = this.enemies(1, 200)
          let ax = 0,
            ay = 0
          for (const e of ens) {
            const d = Math.max(5, hyp2(this.px, this.py, e.pos[0], e.pos[1]))
            ax += (this.px - e.pos[0]) / d / d
            ay += (this.py - e.pos[1]) / d / d
          }
          const n = Math.hypot(ax, ay) || 1
          desire = [ax / n, ay / n]
        }
        boost = me.energy > 20 && t - this.lastDamageT < 2
        break
      }
      case "regen": {
        // hold still in cover; move only if exposed or outside the zone
        goal = toSafe()
        if (!goal) {
          const cp = this.findCover(sc, true)
          if (cp && hyp2(this.px, this.py, cp[0], cp[1]) > 3) goal = cp
        }
        break
      }
      case "patrol":
      default: {
        goal = this.lootGoal(obs, sc)
        if (!goal) {
          goal = toSafe()
          if (!goal) {
            const f = this.farmTarget(obs)
            if (f && f.kind === "crate" && hyp2(this.px, this.py, f.pos[0], f.pos[1]) > 35) goal = [f.pos[0], f.pos[1]]
          }
          if (!goal) {
            // wait at a post with a wall at our back
            const post = this.findPost(sc)
            if (post && hyp2(this.px, this.py, post[0], post[1]) > 2.5) goal = post
          }
        }
        break
      }
    }
    // grab cubes (and wanted weapons) that are close, whatever we're doing
    if (this.mode === "fight" || this.mode === "avoid" || this.mode === "regen" || this.mode === "retreat") {
      const grab = this.nearGrab(obs)
      if (grab) {
        goal = grab
        this.dbg.push("grab")
      }
    }
    // dodging shells
    for (const s of this.shells) {
      const d = hyp2(this.px, this.py, s.x, s.y)
      const R = (this.cfg.weapons.mortar.splash_radius ?? 7) + this.radius + 3
      if (d < R) {
        const k = (R - d) / R
        desire[0] += ((this.px - s.x) / Math.max(0.5, d)) * 3 * k
        desire[1] += ((this.py - s.y) / Math.max(0.5, d)) * 3 * k
        boost = boost || me.energy > 10
        goal = null
      }
    }
    let dir: number[] = [0, 0]
    if (goal) {
      const wp = this.followPath(goal)
      if (wp) {
        const d = hyp2(this.px, this.py, wp[0], wp[1])
        dir = [(wp[0] - this.px) / Math.max(0.01, d), (wp[1] - this.py) / Math.max(0.01, d)]
        if (this.mode === "regen" && hyp2(this.px, this.py, goal[0], goal[1]) < 2) dir = [0, 0]
      }
      dir[0] += desire[0] * 0.3
      dir[1] += desire[1] * 0.3
    } else dir = desire
    let mag = Math.hypot(dir[0], dir[1])
    if (mag > 0.05) {
      dir = this.steer(dir[0] / mag, dir[1] / mag)
      mag = 1
    } else dir = [0, 0]
    a.move = [dir[0] * speedScale, dir[1] * speedScale]
    // stuck detection
    const sp = Math.hypot(me.vel[0], me.vel[1])
    if (mag > 0.5 && sp < 1.2 && me.grounded) this.stuckT += 0.1
    else this.stuckT = Math.max(0, this.stuckT - 0.05)
    if (this.stuckT > 0.8 && t > this.unstickUntil) {
      this.unstickUntil = t + 1.4
      this.unstickDir = Math.atan2(dir[1], dir[0])
      this.stuckT = 0
      this.pathT = -99
    }
    if (t < this.unstickUntil) {
      a.jump = me.grounded
      boost = me.energy > 8
      a.move = [Math.cos(this.unstickDir), Math.sin(this.unstickDir)]
    }
    a.boost = boost
  }

  preferredRange(): number {
    const hasL = this.slotOf("lightning") >= 0 && this.me.energy > 12
    const hasLaser = this.slotOf("laser") >= 0
    const hasGun = this.slotOf("gun") >= 0
    if (hasL) return TUNE.rangeLightning
    if (hasLaser && !hasGun) return 120
    if (hasLaser) return TUNE.rangeLaser
    return TUNE.rangeGun
  }

  /** a nearby walkable point out of every known enemy's sight, inside the safe circle */
  findCover(sc: number[], quiet = false): number[] | null {
    if (this.t - this.coverT < 0.6 && this.coverPt) return this.coverPt
    const ens = this.enemies(quiet ? 20 : 4, 260)
    this.coverT = this.t
    if (!ens.length) {
      this.coverPt = null
      return null
    }
    const W = this.W
    let best: number[] | null = null
    let bs = 1e9
    const here = W.comp[W.cellOf(this.px, this.py)]
    for (const r of [0, 10, 20, 32, 46]) {
      const nAng = r === 0 ? 1 : 16
      for (let k = 0; k < nAng; k++) {
        const ang = (k / nAng) * Math.PI * 2
        const x = this.px + Math.cos(ang) * r,
          y = this.py + Math.sin(ang) * r
        const c = W.cellOf(x, y)
        if (r > 0 && (!W.nav[c] || W.comp[c] !== here)) continue
        const z = W.floorAt(x, y)
        let exposed = 0
        let minD = 1e9
        for (const e of ens) {
          const d = hyp2(x, y, e.pos[0], e.pos[1])
          minD = Math.min(minD, d)
          if (d > 230) continue
          if (W.ray(e.pos[0], e.pos[1], e.pos[2] + 8, x, y, z + 5, 1.5) >= 0.99) exposed++
        }
        let s = exposed * 120 + r * 0.9 - Math.min(minD, 120) * 0.5
        if (hyp2(x, y, sc[0], sc[1]) > sc[2] - 8) s += 250
        if (r > 0 && !W.walkable(this.px, this.py, x, y, 2.6)) s += 25
        for (const sh of this.shells) if (hyp2(x, y, sh.x, sh.y) < 12) s += 200
        if (s < bs) {
          bs = s
          best = [x, y]
        }
      }
    }
    this.coverPt = best
    return best
  }

  post: number[] | null = null
  postT = -99
  /** a waiting spot inside the safe circle, next to a wall (no rear hits), with a moderate field of view */
  findPost(sc: number[]): number[] | null {
    const t = this.t
    if (this.post && t - this.postT < 10 && !this.enemies(1, 150).length && hyp2(this.post[0], this.post[1], sc[0], sc[1]) < sc[2] * 0.7) return this.post
    this.postT = t
    const W = this.W
    const here = W.comp[W.cellOf(this.px, this.py)]
    let best: number[] | null = null
    let bs = 1e9
    for (let n = 0; n < 40; n++) {
      const ang = this.rnd() * Math.PI * 2
      const r = 10 + this.rnd() * 70
      const x = this.px + Math.cos(ang) * r,
        y = this.py + Math.sin(ang) * r
      if (hyp2(x, y, sc[0], sc[1]) > sc[2] * 0.6) continue
      const c = W.cellOf(x, y)
      if (!W.nav[c] || W.comp[c] !== here) continue
      const cl = W.clearAt(x, y)
      if (cl > 6) continue
      const z = W.floorAt(x, y) + 6
      let open = 0
      for (let k = 0; k < 12; k++) {
        const a2 = (k / 12) * Math.PI * 2
        open += W.ray(x, y, z, x + Math.cos(a2) * 120, y + Math.sin(a2) * 120, z, 2)
      }
      open /= 12
      let s = Math.abs(open - 0.45) * 100 + cl * 4 + r * 0.15 + hyp2(x, y, sc[0], sc[1]) * 0.05
      for (const e of this.tracks.values()) {
        if (e.kind !== "mech" || t - e.seenT > 40) continue
        const de = hyp2(x, y, e.pos[0], e.pos[1])
        if (de < 150) s += (150 - de) * 0.6
      }
      if (s < bs) {
        bs = s
        best = [x, y]
      }
    }
    if (best) this.post = best
    return this.post
  }

  /** nearest point within ~12 m, straight-line walkable, hidden from every mech charging a laser at us */
  quickCover(): number[] | null {
    const W = this.W
    const chargers: Track[] = []
    for (const tr of this.tracks.values()) if (tr.kind === "mech" && tr.charging && this.t - tr.seenT < 0.2) chargers.push(tr)
    if (!chargers.length) return null
    for (const r of [4, 7, 10, 13]) {
      let best: number[] | null = null
      let bs = 1e9
      for (let k = 0; k < 16; k++) {
        const ang = (k / 16) * Math.PI * 2
        const x = this.px + Math.cos(ang) * r,
          y = this.py + Math.sin(ang) * r
        if (W.clearAt(x, y) < this.radius + 0.2 || !W.walkable(this.px, this.py, x, y, this.radius)) continue
        const z = W.floorAt(x, y)
        let hidden = true
        for (const e of chargers) {
          if (W.ray(e.pos[0], e.pos[1], e.pos[2] + 7, x, y, z + 4.5, 1) >= 0.99 || W.ray(e.pos[0], e.pos[1], e.pos[2] + 7, x, y, z + 8, 1) >= 0.99) {
            hidden = false
            break
          }
        }
        if (!hidden) continue
        const s = r
        if (s < bs) {
          bs = s
          best = [x, y]
        }
      }
      if (best) return best
    }
    return null
  }

  nearGrab(obs: Observation): number[] | null {
    let best: number[] | null = null
    let bd = TUNE.grabDist
    const ok = (x: number, y: number, z: number) => z <= this.W.floorAt(x, y) + 2.5 && this.W.walkable(this.px, this.py, x, y, this.radius)
    for (const c of obs.cubes) {
      const d = hyp2(this.px, this.py, c.pos[0], c.pos[1])
      if (d < bd && ok(c.pos[0], c.pos[1], c.pos[2])) {
        bd = d
        best = [c.pos[0], c.pos[1]]
      }
    }
    for (const p of obs.pickups) {
      if (this.badPickups.has(p.id) || this.loadoutGain(p.kind).gain < 0.5) continue
      const d = hyp2(this.px, this.py, p.pos[0], p.pos[1])
      if (d < bd && ok(p.pos[0], p.pos[1], p.pos[2])) {
        bd = d
        best = [p.pos[0], p.pos[1]]
      }
    }
    return best
  }

  lootGoal(obs: Observation, sc: number[]): number[] | null {
    let best: number[] | null = null
    let bs = 0
    const consider = (x: number, y: number, z: number, value: number) => {
      if (!this.inSafeZone(x, y, 12)) return
      if (z > this.W.floorAt(x, y) + 2.5) return // on a roof
      const d = hyp2(this.px, this.py, x, y)
      const s = value / (d + 25)
      if (s > bs) {
        bs = s
        best = [x, y]
      }
    }
    for (const c of obs.cubes) consider(c.pos[0], c.pos[1], c.pos[2], 40)
    for (const p of obs.pickups) {
      if (this.badPickups.has(p.id)) continue
      const { gain } = this.loadoutGain(p.kind)
      if (gain > 0) consider(p.pos[0], p.pos[1], p.pos[2], 30 * gain)
    }
    for (const d of obs.drops) {
      if (d.lands_in > 12) continue
      const dd = hyp2(this.px, this.py, d.pos[0], d.pos[1])
      if (dd > 160) continue
      // contested? skip if a known enemy is near it
      let contested = false
      for (const e of this.enemies(3)) if (hyp2(e.pos[0], e.pos[1], d.pos[0], d.pos[1]) < 40) contested = true
      if (!contested) consider(d.pos[0], d.pos[1], d.pos[2], 110 * (this.me.hp < this.me.max_hp * 0.8 ? 1.4 : 1))
    }
    if (best) this.dbg.push("loot")
    return best
  }

  followPath(goal: number[]): number[] | null {
    const t = this.t
    const W = this.W
    // direct if walkable and short
    const dg = hyp2(this.px, this.py, goal[0], goal[1])
    if (dg < 40 && W.walkable(this.px, this.py, goal[0], goal[1], 2.6)) {
      this.path = null
      return goal
    }
    const moved = !this.pathGoal || hyp2(goal[0], goal[1], this.pathGoal[0], this.pathGoal[1]) > Math.max(6, dg * 0.15)
    if (!this.path || moved || t - this.pathT > 3) {
      this.path = W.path(this.px, this.py, goal[0], goal[1], 40000)
      this.pathGoal = goal
      this.pathT = t
    }
    if (!this.path || !this.path.length) return goal
    // advance along the path
    while (this.path.length > 1 && (hyp2(this.px, this.py, this.path[0][0], this.path[0][1]) < 3 || W.walkable(this.px, this.py, this.path[1][0], this.path[1][1], 2.6))) this.path.shift()
    if (this.path.length === 1 && hyp2(this.px, this.py, this.path[0][0], this.path[0][1]) < 2) return goal
    return this.path[0]
  }

  /** context steering: pick the walkable direction closest to the desired one */
  steer(dx: number, dy: number): number[] {
    const W = this.W
    const base = Math.atan2(dy, dx)
    const r = this.radius
    let best = [dx, dy]
    let bs = -1e9
    for (let k = 0; k < 16; k++) {
      const off = k === 0 ? 0 : (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 8)
      const ang = base + off
      const cx = Math.cos(ang),
        cy = Math.sin(ang)
      let ok = 1
      const f0 = W.floorAt(this.px, this.py)
      for (const s of [1.5, 3.5]) {
        const x = this.px + cx * s,
          y = this.py + cy * s
        const cl = W.clearAt(x, y)
        if (cl < r + 0.15 && this.pz < W.floorAt(x, y) + 1) ok = Math.min(ok, cl / (r + 0.15) - 0.5)
        const df = W.floorAt(x, y) - Math.max(f0, this.pz)
        if (df > 1.25) ok = Math.min(ok, 0)
      }
      const s = Math.cos(off) + ok * 2
      if (s > bs) {
        bs = s
        best = [cx, cy]
      }
    }
    return best
  }
}
