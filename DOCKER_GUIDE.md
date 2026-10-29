# Docker Deployment Guide

Deployment target: **Next.js web (port 3000) + FastAPI backend (port 8000)**.

The web image is frontend-only. It calls the backend at the URL compiled into
`NEXT_PUBLIC_API_URL` at build time (currently `http://localhost:8000`), so on a
single EC2 instance run the container with `--network host`. Keep port 8000 closed in the
security group: only the web port needs to be public; the backend trusts the web proxy.

Replace every `<placeholder>` before running.

---

## Published image

| Item | Value |
|---|---|
| Docker Hub | `varunmalewar/papeer_project:latest` (also `varunmalewar/papeer-web:latest`) |
| Digest | `sha256:a32c2f5302c955b27e86045a88180d82ba60bc2684cdbc4e034fb936859210f4` |
| Size | 2.23 GB on disk (475 MB compressed) |
| Port | 3000 |

---

## Step 1 — Install Docker on EC2 (Ubuntu)

Run everything in `docker_install_on_AWS.txt`, then:

```bash
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

Ubuntu 26.04 (`resolute`) is supported by Docker's apt repository.

---

## Step 2 — Build the image (optional)

Only needed if you changed the frontend. The pushed image already exists.

```bash
cd web
docker build \
  --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 \
  --build-arg NEXT_PUBLIC_NEON_AUTH_URL=<your-neon-auth-url> \
  --build-arg NEON_AUTH_BASE_URL=<your-neon-auth-url> \
  --build-arg NEON_AUTH_COOKIE_SECRET=<build-only-placeholder-32-chars-min> \
  -t varunmalewar/papeer_project:latest .
```

Notes:
- `NEXT_PUBLIC_API_URL` is **inlined at build time**. Changing it later with `-e`
  does nothing; rebuild instead.
- `NEON_AUTH_COOKIE_SECRET` here is a throwaway value so `next build` can evaluate
  the auth module. It is not stored in the image; pass the real one at runtime.
- `npm ci` fails on this lockfile; the Dockerfile uses `npm install` on purpose.

---

## Step 3 — Push to Docker Hub

```bash
docker login
docker push varunmalewar/papeer_project:latest
```

---

## Step 4 — Run the backend on the same instance

The web image expects the API on `localhost:8000`.

```bash
sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
cd ~/Advanced_Rag_Codes-main/14_rag_papeer_project
uv sync --no-dev
```

`uv` reads `.python-version` (3.12) and installs that interpreter itself, so you
do not need Ubuntu's system Python version to match.

Create `.env` in the project root (never bake secrets into the image):

```bash
OPENAI_API_KEY=<...>
GOOGLE_API_KEY=<...>
QDRANT_URL=<...>
QDRANT_API_KEY=<...>
DATABASE_URL=<...>
PAPEER_API_KEY=<same value you pass to the web container>
```

Start it:

```bash
set -a; . ./.env; set +a
uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
```

For production, run it under systemd or `nohup`/`tmux` so it survives logout.

---

## Step 5 — Run the web container

```bash
docker pull varunmalewar/papeer_project:latest

docker run -d \
  --name papeer-web \
  --restart unless-stopped \
  --network host \
  -e NEON_AUTH_BASE_URL="https://<your-endpoint>.neonauth.<region>.aws.neon.tech/neondb/auth" \
  -e NEON_AUTH_COOKIE_SECRET="<32+ char secret>" \
  -e PAPEER_API_KEY="<a-long-random-secret>" \
  varunmalewar/papeer_project:latest
```

- `--network host` is Linux-only and makes `localhost:8000` reach the backend.
  Do not add `-p` with host networking.
- `NEON_AUTH_COOKIE_SECRET` must be 32+ characters. Changing it logs everyone out.
- If the backend runs on another machine, rebuild with
  `--build-arg NEXT_PUBLIC_API_URL=http://<backend-host>:8000` and use bridge
  networking with `-p 3000:3000`.

---

## Step 6 — Open the app

```
http://<ec2-public-ip>:3000
```

Open port **3000** in the EC2 security group. Port 8000 only if you want direct
API access.

---

## Verify

```bash
docker ps
docker logs papeer-web
curl -I http://localhost:3000
curl http://localhost:8000/health
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Missing required config: cookies.secret` | `NEON_AUTH_COOKIE_SECRET` not passed at runtime |
| Chat/upload returns 500, `ECONNREFUSED 8000` | Backend not running, or web not using `--network host` |
| Login redirect loop / auth error | `NEON_AUTH_BASE_URL` wrong or domain not trusted in Neon |
| `no space left on device` | Need ~5 GB free: `docker system prune -af` |
| API returns 401 | `PAPEER_API_KEY` differs between web container and backend `.env` |
| UI loads but data never appears | `NEXT_PUBLIC_API_URL` baked wrong; rebuild the image |

---

## Useful commands

```bash
docker ps
docker logs -f papeer-web
docker stop papeer-web
docker rm papeer-web
docker system prune -af
```
