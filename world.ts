/**
 * Static map model: terrain, solid prisms, walkable floor, wall clearance, a 2 m navigation grid with A*,
 * and line-of-sight / line-of-fire ray tests. Built once at `hello` (≈ 0.5 s) and shared by the bot and the
 * local simulator.
 */
import { readFileSync } from "node:fs"
import { join } from "node:path"

export type Prism = { kind: string; base: number; top: number; ring: number[]; holes: number[][] | number[] }

const INF = 1e9

export class World {
  minX: number
  minY: number
  nx: number
  ny: number
  bminX: number
  bmaxX: number
  bminY: number
  bmaxY: number
  /** terrain height at vertex (i,j) */
  ground: Float32Array
  /** up to two solid intervals per sample (lo > hi means none) */
  lo1: Float32Array
  hi1: Float32Array
  lo2: Float32Array
  hi2: Float32Array
  /** walkable floor for a ground-layer walker */
  floor: Float32Array
  /** 1 = a ground walker cannot stand here (wall or edge) */
  wall: Uint8Array
  /** distance (m) to the nearest wall sample */
  clear: Float32Array
  /** highest solid top at the sample (for roof detection / cover) */
  roof: Float32Array
  // nav grid (2 m)
  NS = 2
  gnx: number
  gny: number
  nav: Uint8Array // 1 walkable
  navFloor: Float32Array
  navClear: Float32Array
  comp: Int32Array // connected component id
  // A* scratch
  private gScore: Float32Array
  private parent: Int32Array
  private stamp: Uint32Array
  private closed: Uint32Array
  private gen = 1
  private heap: Int32Array
  private heapF: Float32Array

  constructor(dir: string) {
    const wj = JSON.parse(readFileSync(join(dir, "v2/world.json"), "utf8"))
    const t = wj.terrain
    this.minX = t.minX
    this.minY = t.minY
    this.nx = t.nx
    this.ny = t.ny
    this.bminX = wj.bounds.minX
    this.bmaxX = wj.bounds.maxX
    this.bminY = wj.bounds.minY
    this.bmaxY = wj.bounds.maxY
    const N = this.nx * this.ny
    const buf = readFileSync(join(dir, "v2/terrain.i16"))
    const raw = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 2))
    const scale = t.scale ?? 0.01
    this.ground = new Float32Array(N)
    for (let k = 0; k < N; k++) this.ground[k] = raw[k] * scale
    this.lo1 = new Float32Array(N).fill(INF)
    this.hi1 = new Float32Array(N).fill(-INF)
    this.lo2 = new Float32Array(N).fill(INF)
    this.hi2 = new Float32Array(N).fill(-INF)
    this.roof = new Float32Array(N).fill(-INF)
    const deck = new Float32Array(N).fill(-INF)
    for (const p of wj.prisms as Prism[]) this.rasterize(p, deck)
    this.floor = new Float32Array(N)
    this.wall = new Uint8Array(N)
    this.computeFloor(deck)
    this.clear = new Float32Array(N)
    this.distanceTransform()
    const NS = this.NS
    this.gnx = Math.floor(this.nx / NS)
    this.gny = Math.floor(this.ny / NS)
    const G = this.gnx * this.gny
    this.nav = new Uint8Array(G)
    this.navFloor = new Float32Array(G)
    this.navClear = new Float32Array(G)
    this.comp = new Int32Array(G).fill(-1)
    this.gScore = new Float32Array(G)
    this.parent = new Int32Array(G)
    this.stamp = new Uint32Array(G)
    this.closed = new Uint32Array(G)
    this.heap = new Int32Array(G * 4)
    this.heapF = new Float32Array(G * 4)
    this.buildNav()
  }

  // ------------------------------------------------------------------ build
  private rasterize(p: Prism, deck: Float32Array) {
    const rings: number[][] = [p.ring]
    const holes = p.holes as unknown
    if (Array.isArray(holes)) {
      if (holes.length && Array.isArray(holes[0])) for (const h of holes as number[][]) rings.push(h)
      else if (holes.length) rings.push(holes as number[])
    }
    let y0 = INF,
      y1 = -INF
    for (let k = 1; k < p.ring.length; k += 2) {
      y0 = Math.min(y0, p.ring[k])
      y1 = Math.max(y1, p.ring[k])
    }
    const j0 = Math.max(0, Math.ceil(y0 - this.minY))
    const j1 = Math.min(this.ny - 1, Math.floor(y1 - this.minY))
    const xs: number[] = []
    const isBridge = p.kind === "bridge"
    for (let j = j0; j <= j1; j++) {
      const y = this.minY + j
      xs.length = 0
      for (const r of rings) {
        const n = r.length / 2
        for (let a = 0, b = n - 1; a < n; b = a++) {
          const ya = r[2 * a + 1],
            yb = r[2 * b + 1]
          if (ya > y !== yb > y) {
            const xa = r[2 * a],
              xb = r[2 * b]
            xs.push(xa + ((y - ya) * (xb - xa)) / (yb - ya))
          }
        }
      }
      xs.sort((u, v) => u - v)
      for (let q = 0; q + 1 < xs.length; q += 2) {
        const i0 = Math.max(0, Math.ceil(xs[q] - this.minX))
        const i1 = Math.min(this.nx - 1, Math.floor(xs[q + 1] - this.minX))
        for (let i = i0; i <= i1; i++) {
          const k = j * this.nx + i
          const g = this.ground[k]
          if (p.top > this.roof[k]) this.roof[k] = p.top
          if (isBridge && p.top - g <= 6.5) {
            if (p.top > deck[k]) deck[k] = p.top
            continue
          }
          this.addInterval(k, p.base, p.top)
        }
      }
    }
  }

  private addInterval(k: number, lo: number, hi: number) {
    if (this.lo1[k] > this.hi1[k]) {
      this.lo1[k] = lo
      this.hi1[k] = hi
    } else if (lo <= this.hi1[k] + 0.5 && hi >= this.lo1[k] - 0.5) {
      this.lo1[k] = Math.min(this.lo1[k], lo)
      this.hi1[k] = Math.max(this.hi1[k], hi)
    } else if (this.lo2[k] > this.hi2[k]) {
      this.lo2[k] = lo
      this.hi2[k] = hi
    } else {
      this.lo2[k] = Math.min(this.lo2[k], lo)
      this.hi2[k] = Math.max(this.hi2[k], hi)
    }
    // keep interval 1 the lower one
    if (this.lo2[k] <= this.hi2[k] && this.lo2[k] < this.lo1[k]) {
      const a = this.lo1[k],
        b = this.hi1[k]
      this.lo1[k] = this.lo2[k]
      this.hi1[k] = this.hi2[k]
      this.lo2[k] = a
      this.hi2[k] = b
    }
  }

  private computeFloor(deck: Float32Array) {
    const { nx, ny } = this
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i
        let f = Math.max(this.ground[k], deck[k])
        // low platforms you can step onto
        if (this.lo1[k] <= this.hi1[k] && this.lo1[k] <= f + 1.2 && this.hi1[k] <= f + 1.2) f = Math.max(f, this.hi1[k])
        let w = 0
        if (this.lo1[k] <= this.hi1[k] && this.lo1[k] < f + 9 && this.hi1[k] > f + 1.2) w = 1
        if (this.lo2[k] <= this.hi2[k] && this.lo2[k] < f + 9 && this.hi2[k] > f + 1.2) w = 1
        const x = this.minX + i,
          y = this.minY + j
        if (x < this.bminX + 1 || x > this.bmaxX - 1 || y < this.bminY + 1 || y > this.bmaxY - 1) w = 1
        this.floor[k] = f
        this.wall[k] = w
      }
    // ledges: a step taller than ~1.6 m between neighbours is impassable going up; mark the low side's
    // neighbour pair as wall-ish only when very tall (stadium rim etc.)
    for (let j = 1; j < ny - 1; j++)
      for (let i = 1; i < nx - 1; i++) {
        const k = j * nx + i
        if (this.wall[k]) continue
        const f = this.floor[k]
        const d = Math.max(Math.abs(this.floor[k + 1] - f), Math.abs(this.floor[k - 1] - f), Math.abs(this.floor[k + nx] - f), Math.abs(this.floor[k - nx] - f))
        if (d > 2.2) this.wall[k] = 2
      }
  }

  /** exact Euclidean distance transform (Felzenszwalb) of the wall mask, in metres */
  private distanceTransform() {
    const { nx, ny } = this
    const f = new Float64Array(Math.max(nx, ny))
    const d = new Float64Array(Math.max(nx, ny))
    const v = new Int32Array(Math.max(nx, ny))
    const z = new Float64Array(Math.max(nx, ny) + 1)
    const tmp = new Float64Array(nx * ny)
    const BIG = 1e12
    const dt1 = (n: number) => {
      let kk = 0
      v[0] = 0
      z[0] = -Infinity
      z[1] = Infinity
      for (let q = 1; q < n; q++) {
        let s = (f[q] + q * q - (f[v[kk]] + v[kk] * v[kk])) / (2 * q - 2 * v[kk])
        while (s <= z[kk]) {
          kk--
          s = (f[q] + q * q - (f[v[kk]] + v[kk] * v[kk])) / (2 * q - 2 * v[kk])
        }
        kk++
        v[kk] = q
        z[kk] = s
        z[kk + 1] = Infinity
      }
      kk = 0
      for (let q = 0; q < n; q++) {
        while (z[kk + 1] < q) kk++
        d[q] = (q - v[kk]) * (q - v[kk]) + f[v[kk]]
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) f[j] = this.wall[j * nx + i] ? 0 : BIG
      dt1(ny)
      for (let j = 0; j < ny; j++) tmp[j * nx + i] = d[j]
    }
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) f[i] = tmp[j * nx + i]
      dt1(nx)
      for (let i = 0; i < nx; i++) this.clear[j * nx + i] = Math.sqrt(d[i])
    }
  }

  private buildNav() {
    const { gnx, gny, NS } = this
    for (let J = 0; J < gny; J++)
      for (let I = 0; I < gnx; I++) {
        const G = J * gnx + I
        const k = (J * NS + 1) * this.nx + (I * NS + 1)
        this.navClear[G] = this.clear[k]
        this.navFloor[G] = this.floor[k]
        this.nav[G] = this.clear[k] >= 2.9 ? 1 : 0
      }
    // components (8-connected with step check)
    let c = 0
    const q = new Int32Array(gnx * gny)
    for (let s = 0; s < gnx * gny; s++) {
      if (!this.nav[s] || this.comp[s] >= 0) continue
      let h = 0,
        tl = 0
      q[tl++] = s
      this.comp[s] = c
      while (h < tl) {
        const u = q[h++]
        const ui = u % gnx,
          uj = (u / gnx) | 0
        for (let dj = -1; dj <= 1; dj++)
          for (let di = -1; di <= 1; di++) {
            if (!di && !dj) continue
            const vi = ui + di,
              vj = uj + dj
            if (vi < 0 || vj < 0 || vi >= gnx || vj >= gny) continue
            const w = vj * gnx + vi
            if (this.comp[w] >= 0 || !this.edgeOk(u, w, di, dj)) continue
            this.comp[w] = c
            q[tl++] = w
          }
      }
      c++
    }
  }

  private edgeOk(u: number, w: number, di: number, dj: number) {
    if (!this.nav[w]) return false
    if (Math.abs(this.navFloor[w] - this.navFloor[u]) > 2.4) return false
    if (di && dj) {
      const gnx = this.gnx
      if (!this.nav[u + di] || !this.nav[u + dj * gnx]) return false
    }
    return true
  }

  // ------------------------------------------------------------------ queries
  idx(x: number, y: number) {
    let i = Math.round(x - this.minX),
      j = Math.round(y - this.minY)
    if (i < 0) i = 0
    else if (i >= this.nx) i = this.nx - 1
    if (j < 0) j = 0
    else if (j >= this.ny) j = this.ny - 1
    return j * this.nx + i
  }
  groundAt(x: number, y: number) {
    let fx = x - this.minX,
      fy = y - this.minY
    fx = Math.max(0, Math.min(this.nx - 1.001, fx))
    fy = Math.max(0, Math.min(this.ny - 1.001, fy))
    const i = Math.floor(fx),
      j = Math.floor(fy)
    const ax = fx - i,
      ay = fy - j
    const k = j * this.nx + i
    const g = this.ground
    return (g[k] * (1 - ax) + g[k + 1] * ax) * (1 - ay) + (g[k + this.nx] * (1 - ax) + g[k + this.nx + 1] * ax) * ay
  }
  floorAt(x: number, y: number) {
    return this.floor[this.idx(x, y)]
  }
  clearAt(x: number, y: number) {
    return this.clear[this.idx(x, y)]
  }
  inBounds(x: number, y: number) {
    return x > this.bminX && x < this.bmaxX && y > this.bminY && y < this.bmaxY
  }
  /** is the point inside solid (terrain or prism)? */
  solid(x: number, y: number, z: number) {
    const k = this.idx(x, y)
    if (z < this.ground[k] - 0.2) return true
    if (z >= this.lo1[k] && z <= this.hi1[k]) return true
    if (z >= this.lo2[k] && z <= this.hi2[k]) return true
    return false
  }
  /** first fraction t∈[0,1] along a→b that is solid, or 1 if clear. step in metres. */
  ray(ax: number, ay: number, az: number, bx: number, by: number, bz: number, step = 1): number {
    const dx = bx - ax,
      dy = by - ay,
      dz = bz - az
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const n = Math.max(1, Math.ceil(L / step))
    for (let s = 1; s < n; s++) {
      const t = s / n
      if (this.solid(ax + dx * t, ay + dy * t, az + dz * t)) return t
    }
    return 1
  }
  los(a: number[], b: number[], step = 1) {
    return this.ray(a[0], a[1], a[2], b[0], b[1], b[2], step) >= 1
  }

  // ------------------------------------------------------------------ navigation
  cellOf(x: number, y: number) {
    let I = Math.floor((x - this.minX) / this.NS),
      J = Math.floor((y - this.minY) / this.NS)
    I = Math.max(0, Math.min(this.gnx - 1, I))
    J = Math.max(0, Math.min(this.gny - 1, J))
    return J * this.gnx + I
  }
  cellX(c: number) {
    return this.minX + (c % this.gnx) * this.NS + 1
  }
  cellY(c: number) {
    return this.minY + ((c / this.gnx) | 0) * this.NS + 1
  }
  /** nearest walkable nav cell to (x,y) (optionally in component `comp`) within r cells */
  nearestNav(x: number, y: number, r = 12, comp = -1): number {
    const c0 = this.cellOf(x, y)
    if (this.nav[c0] && (comp < 0 || this.comp[c0] === comp)) return c0
    const I0 = c0 % this.gnx,
      J0 = (c0 / this.gnx) | 0
    let best = -1,
      bd = INF
    for (let dj = -r; dj <= r; dj++)
      for (let di = -r; di <= r; di++) {
        const I = I0 + di,
          J = J0 + dj
        if (I < 0 || J < 0 || I >= this.gnx || J >= this.gny) continue
        const c = J * this.gnx + I
        if (!this.nav[c] || (comp >= 0 && this.comp[c] !== comp)) continue
        const d = di * di + dj * dj
        if (d < bd) {
          bd = d
          best = c
        }
      }
    return best
  }

  /** straight walkable segment on the nav grid (sampled) */
  walkable(ax: number, ay: number, bx: number, by: number, minClear = 2.7) {
    const dx = bx - ax,
      dy = by - ay
    const L = Math.hypot(dx, dy)
    const n = Math.max(1, Math.ceil(L / 1))
    let pf = this.floorAt(ax, ay)
    for (let s = 1; s <= n; s++) {
      const t = s / n
      const x = ax + dx * t,
        y = ay + dy * t
      const k = this.idx(x, y)
      if (this.clear[k] < minClear) return false
      const f = this.floor[k]
      if (f - pf > 1.3) return false
      pf = f
    }
    return true
  }

  /**
   * Weighted A* from (sx,sy) to (tx,ty). Returns waypoint list (world xy, excluding start), or null.
   * `cost(c)` optional extra cost per cell entered (e.g. danger); maxExpand bounds the work.
   */
  path(sx: number, sy: number, tx: number, ty: number, maxExpand = 60000, extra?: (c: number) => number): number[][] | null {
    const s = this.nearestNav(sx, sy, 6)
    if (s < 0) return null
    const t = this.nearestNav(tx, ty, 25, this.comp[s])
    if (t < 0) return null
    if (s === t) return [[tx, ty]]
    const gnx = this.gnx
    const gen = ++this.gen
    const tI = t % gnx,
      tJ = (t / gnx) | 0
    const W = 1.4
    const hfn = (c: number) => {
      const dx = Math.abs((c % gnx) - tI),
        dy = Math.abs(((c / gnx) | 0) - tJ)
      return (Math.max(dx, dy) + 0.4142 * Math.min(dx, dy)) * W
    }
    let hn = 0
    const heap = this.heap,
      hf = this.heapF
    const push = (c: number, f: number) => {
      let i = hn++
      while (i > 0) {
        const p = (i - 1) >> 1
        if (hf[p] <= f) break
        heap[i] = heap[p]
        hf[i] = hf[p]
        i = p
      }
      heap[i] = c
      hf[i] = f
    }
    const pop = () => {
      const top = heap[0]
      const lc = heap[--hn],
        lf = hf[hn]
      let i = 0
      for (;;) {
        let ch = 2 * i + 1
        if (ch >= hn) break
        if (ch + 1 < hn && hf[ch + 1] < hf[ch]) ch++
        if (hf[ch] >= lf) break
        heap[i] = heap[ch]
        hf[i] = hf[ch]
        i = ch
      }
      heap[i] = lc
      hf[i] = lf
      return top
    }
    this.stamp[s] = gen
    this.gScore[s] = 0
    this.parent[s] = -1
    push(s, hfn(s))
    let expanded = 0
    let found = false
    let bestC = s,
      bestH = hfn(s)
    while (hn > 0) {
      const u = pop()
      if (this.closed[u] === gen) continue
      this.closed[u] = gen
      if (u === t) {
        found = true
        break
      }
      if (++expanded > maxExpand) break
      const ui = u % gnx,
        uj = (u / gnx) | 0
      const gu = this.gScore[u]
      for (let dj = -1; dj <= 1; dj++)
        for (let di = -1; di <= 1; di++) {
          if (!di && !dj) continue
          const vi = ui + di,
            vj = uj + dj
          if (vi < 0 || vj < 0 || vi >= gnx || vj >= this.gny) continue
          const v = vj * gnx + vi
          if (this.closed[v] === gen || !this.edgeOk(u, v, di, dj)) continue
          let step = di && dj ? 1.4142 : 1
          const cl = this.navClear[v]
          if (cl < 4.5) step += (4.5 - cl) * 0.35
          if (extra) step += extra(v)
          const g = gu + step
          if (this.stamp[v] !== gen || g < this.gScore[v]) {
            this.stamp[v] = gen
            this.gScore[v] = g
            this.parent[v] = u
            const h = hfn(v)
            if (h < bestH) {
              bestH = h
              bestC = v
            }
            push(v, g + h)
          }
        }
    }
    const end = found ? t : bestC
    if (end === s) return null
    const cells: number[] = []
    for (let c = end; c >= 0 && c !== s; c = this.parent[c]) cells.push(c)
    cells.reverse()
    const pts = cells.map((c) => [this.cellX(c), this.cellY(c)])
    if (found) pts[pts.length - 1] = this.walkable(this.cellX(t), this.cellY(t), tx, ty) ? [tx, ty] : pts[pts.length - 1]
    return this.smooth(sx, sy, pts)
  }

  /** string-pull: drop waypoints that can be skipped with a walkable straight line */
  smooth(sx: number, sy: number, pts: number[][]): number[][] {
    const out: number[][] = []
    let ax = sx,
      ay = sy
    let i = 0
    while (i < pts.length) {
      let j = Math.min(pts.length - 1, i + 40)
      while (j > i && !this.walkable(ax, ay, pts[j][0], pts[j][1])) j--
      out.push(pts[j])
      ax = pts[j][0]
      ay = pts[j][1]
      i = j + 1
    }
    return out
  }
}

const cache = new Map<string, World>()
/** one World per map dir per process (the simulator shares it between bots) */
export function getWorld(dir: string): World {
  let w = cache.get(dir)
  if (!w) {
    w = new World(dir)
    cache.set(dir, w)
  }
  return w
}
