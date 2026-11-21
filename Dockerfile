# Mech Royale bot. The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB,
# as uid 65534, with the map mounted read-only at /arena.
FROM oven/bun:1.4.4-slim
WORKDIR /bot
COPY package.json *.ts ./
ENTRYPOINT ["bun", "/bot/main.ts"]
