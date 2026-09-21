# The studio. Build context is the repository root.

FROM node:22-bookworm-slim AS deps
WORKDIR /app
RUN npm install -g pnpm@10 --no-fund --no-audit
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:22-bookworm-slim AS builder
WORKDIR /app
RUN npm install -g pnpm@10 --no-fund --no-audit
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# No generation happens at build time, so no API URL is needed here: the
# Local AI origin is read at request time inside the server action.
RUN pnpm build

FROM node:22-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
# No pnpm in the runtime image: `next` is invoked directly, which drops an
# entire global-install layer and one process from the container.
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/.next ./.next
COPY --chown=node:node public ./public
COPY --chown=node:node package.json next.config.ts tsconfig.json next-env.d.ts ./
USER node
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["./node_modules/.bin/next", "start", "--hostname", "0.0.0.0", "--port", "3000"]
