# SegNCA — Setup Guide (Kali Linux)

Full-stack app: a FastAPI backend serving live inference from two real trained checkpoints
(SegNCA, 8,320 params · TinyUNet, 483,153 params), and a React/Vite frontend with a live demo,
an interactive robustness playground, and a benchmarks dashboard.

Two ways to run it: **manual** (recommended first — you see every moving part) or **Docker**
(one command, once manual works).

---

## 0. What you need installed

| Tool | Why | Check with |
|---|---|---|
| Python 3.10+ | backend, PyTorch inference | `python3 --version` |
| Node.js 20+ | frontend build | `node --version` |
| git | cloning/version control | `git --version` |
| ~2 GB free disk | PyTorch + node_modules | `df -h` |

Kali ships Python 3 by default, but usually an older Node.js in its main repo — we'll install a
current one via `nvm` rather than `apt`, so version mismatches don't bite you later.

---

## 1. System prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential curl git
```

`build-essential` matters here — some PyTorch dependency wheels fall back to compiling from
source on less common architectures, and you want the compiler present rather than a cryptic
failure mid-install.

---

## 2. Install Node.js via nvm

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc          # or ~/.zshrc if you use zsh

nvm install 20
nvm use 20
node --version             # should print v20.x.x
npm --version
```

---

## 3. Get the project onto disk

If you downloaded the files from this conversation, put them all in one folder matching this
layout:

```
segnca-app/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── checkpoints/
│   │   ├── segnca_best.pt
│   │   └── tinyunet_best.pt
│   └── data/
│       ├── images.npy
│       ├── labels.npy
│       ├── train_idx.npy
│       ├── val_idx.npy
│       ├── final_metrics.json
│       └── robustness.json
├── frontend/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   ├── .env.example
│   └── Dockerfile
├── docker-compose.yml
├── run_backend.sh
└── run_frontend.sh
```

Then:
```bash
cd ~/segnca-app          # or wherever you placed it
chmod +x run_backend.sh run_frontend.sh
```

If instead you're starting from a git repo you pushed this to:
```bash
git clone <your-repo-url> segnca-app
cd segnca-app
```

---

## 4. Backend — manual setup

```bash
cd segnca-app/backend
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`torch` is the biggest download here (several hundred MB, CPU build). On a slow connection this
is the step to expect to wait on — everything else installs in seconds.

Start it:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Verify it's actually serving, in a second terminal:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","nca_params":8320,"unet_params":483153}
```

Or skip the manual steps above and just run:
```bash
cd segnca-app
./run_backend.sh
```
which creates the venv, installs dependencies, and starts the server for you.

**Leave this terminal running** — it's your API server.

---

## 5. Frontend — manual setup

Open a **new terminal**:

```bash
cd segnca-app/frontend
cp .env.example .env       # points the frontend at http://localhost:8000
npm install
npm run dev -- --host 0.0.0.0
```

You should see:
```
VITE ready
➜  Local:   http://localhost:5173/
➜  Network: http://<your-lan-ip>:5173/
```

Or, again, just:
```bash
cd segnca-app
./run_frontend.sh
```

Open `http://localhost:5173` in a browser. The hero section's status line should read
`connected · 8,320 + 483,153 params loaded` — if it says `backend unreachable`, the API isn't
running or `.env` doesn't match the port it's on.

---

## 6. Using it on your LAN / from another device

Both dev servers were started with `--host 0.0.0.0`, so they're reachable from other machines on
your network, not just `localhost`. Find your Kali box's IP:

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
```

Then on the other device, visit `http://<that-ip>:5173`. If it doesn't load, Kali's firewall
(if you've enabled `ufw`) may be blocking it:

```bash
sudo ufw status                 # check if ufw is even active — often it isn't by default
sudo ufw allow 5173/tcp
sudo ufw allow 8000/tcp
```

If you do this, also update `frontend/.env` to point `VITE_API_URL` at `http://<your-ip>:8000`
instead of `localhost`, and restart the frontend — otherwise the browser on the other device will
try to reach an API on itself, not your Kali box.

---

## 7. Production build (instead of the dev server)

The dev server is for iterating. For something that behaves like a deployed product:

```bash
cd segnca-app/frontend
npm run build          # outputs static files to dist/
npm install -g serve
serve -s dist -l 4173
```

Keep the backend running the same way as before (`uvicorn`) — in production you'd normally run it
behind a proper process manager, see §9.

---

## 8. Docker alternative (one command, once you've confirmed manual works)

Install Docker on Kali if you don't have it:
```bash
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER     # log out/in after this so `docker` works without sudo
```

Then, from the project root:
```bash
cd segnca-app
docker compose up --build
```

This builds both images and starts backend on `:8000` and frontend on `:4173`. First build is
slow (compiling the PyTorch layer); subsequent ones are cached.

---

## 9. Making it persistent (systemd, optional)

If you want the backend to survive reboots / terminal closes, run it as a systemd service instead
of a foreground process:

```bash
sudo tee /etc/systemd/system/segnca-backend.service > /dev/null << 'EOF'
[Unit]
Description=SegNCA FastAPI backend
After=network.target

[Service]
Type=simple
User=%i
WorkingDirectory=/home/%i/segnca-app/backend
ExecStart=/home/%i/segnca-app/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now segnca-backend@$USER
sudo systemctl status segnca-backend@$USER
```

For the frontend in this same persistent setup, put the built `dist/` behind `nginx` rather than
running `serve` as a service:

```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/segnca > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    root /home/YOUR_USER/segnca-app/frontend/dist;
    index index.html;
    location / { try_files $uri /index.html; }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
EOF
sudo ln -s /etc/nginx/sites-available/segnca /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```
With this nginx config, set `VITE_API_URL=` (empty, same-origin) in `.env` before rebuilding, so
the frontend calls `/api/...` on its own origin and nginx forwards it — no CORS, no separate port
to remember.

---

## 10. Troubleshooting

**`ERR_CONNECTION_REFUSED` in the browser console, health badge stuck on "connecting…"**
The backend isn't running or is on a different port than `frontend/.env` expects. Check terminal
1 for uvicorn output, confirm `curl http://localhost:8000/api/health` works from the same
machine the browser is on.

**CORS errors in the browser console**
Only happens if you edited `allow_origins` in `backend/main.py` away from `["*"]`. Fine for a
local demo; if you lock it down for a real deployment, list your actual frontend origin.

**`pip install -r requirements.txt` hangs or fails on torch**
Slow connection, not a broken install — the CPU wheel is a few hundred MB. Let it finish, or
check `pip config list` for a stale index URL if it fails outright.

**Port already in use (`8000` or `5173`)**
```bash
sudo lsof -i :8000
kill <pid>
```
or just run on a different port (`uvicorn main:app --port 8001`, updating `.env` to match).

**Predictions look identical every time you click "run inference"**
SegNCA is stochastic by design (a random ~50% of cells update each step) — the `/api/predict`
response includes `dice_std`, so click a few times and watch the number move slightly. That's
expected, not a bug — see the architecture section on the page itself.
