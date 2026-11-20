# ironclad: a Mech Royale bot (`mechfight/1`)

A TypeScript bot that runs on Bun. The `Dockerfile` at the top builds it, and the image speaks the
JSON-lines protocol on stdin and stdout.

| File | Role |
| --- | --- |
| `main.ts` | Entry point. Wires the brain into the kit's stdin/stdout loop. |
| `mechfight.ts` | The kit's protocol wrapper and types, unchanged. |
| `world.ts` | Map model built at `hello` (about 0.3 s): terrain, prisms, walkable floor, wall clearance, a 2 m nav grid with A*, and ray tests for line of sight and line of fire. |
| `brain.ts` | The pilot: perception and tracking, fire control for all four weapons (gun lead, laser charge and release, lightning cone, ballistic mortar), loot and upgrade choices, and the zone, fight, retreat-to-cover, regen and loot intents. Movement uses path following plus context steering. |
| `util.ts` | Small math helpers and a seeded RNG. |

Run the conformance check from the kit folder:

```sh
python3 check/check.py -- bun /path/to/main.ts
docker build -t ironclad .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" ironclad
```
