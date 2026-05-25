FROM node:22-alpine
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml /app/
RUN pnpm install --no-frozen-lockfile
COPY . /app
RUN pnpm build
EXPOSE 3000
CMD ["pnpm", "start"]
