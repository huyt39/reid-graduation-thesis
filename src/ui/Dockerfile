FROM node:22-alpine
WORKDIR /app
RUN corepack enable
COPY package.json /app/
RUN pnpm install --no-frozen-lockfile
COPY . /app
EXPOSE 3000
CMD ["pnpm", "dev"]
