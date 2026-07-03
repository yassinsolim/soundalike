# soundalike in Spotify — right-click → *Find soundalikes*

Add a **“Find soundalikes”** item to the right-click menu of any track in the
Spotify desktop app. Click it and a panel shows songs that genuinely *sound*
like the one you clicked — powered by the local neural audio model, not tags.

![flow](../../docs/soundalike-results.png)

There are two ways to use soundalike. Pick based on your Spotify install:

| | Web app (works with **any** Spotify) | Spicetify (in-app right-click) |
|---|---|---|
| Setup | none beyond `soundalike serve` | patch the Spotify client |
| Works with Microsoft-Store Spotify | ✅ | ❌ (needs the standalone app) |
| Trigger | paste a song / **Copy Song Link** | right-click a track |

---

## Option A — the web app (recommended, zero client patching)

```bash
pip install -e ".[ml]"      # first time only
soundalike serve            # opens http://127.0.0.1:8787
```

Then either type `Title — Artist`, or — the frictionless way — in Spotify
**right-click a song → Share → Copy Song Link** and paste it. You get instant
soundalikes with an “Open in Spotify” button on each. This works with the
Microsoft-Store build of Spotify too.

---

## Option B — Spicetify (true in-app right-click)

Spicetify patches the Spotify **desktop** client to add custom menu items. It
**requires the standalone Spotify** from <https://www.spotify.com/download> —
the **Microsoft-Store version cannot be patched** (this is a Spicetify
limitation, not ours). If you have the Store version, either use Option A or
reinstall Spotify from spotify.com first.

### 1. Install Spicetify

PowerShell (Windows):

```powershell
iwr -useb https://raw.githubusercontent.com/spicetify/cli/main/install.ps1 | iex
```

(macOS/Linux and details: <https://spicetify.app/docs/getting-started>.)

### 2. Install this extension

```powershell
# copy the extension into Spicetify's Extensions folder
copy integrations\spicetify\soundalike.js "$(spicetify config-dir)\Extensions\"

spicetify config extensions soundalike.js
spicetify apply
```

### 3. Run the local engine and use it

```bash
soundalike serve --no-browser
```

Now right-click any song in Spotify → **Find soundalikes**. A panel opens with
vibe-matched tracks; click one to jump to it in Spotify.

The extension only talks to `http://127.0.0.1:8787` on your own machine — no
data leaves your computer, and nothing runs unless you started `soundalike serve`.

---

## How it works

```
right-click track ─▶ Spotify track id ─▶ local server /api/recommend
                                              │
                     already in the library?  ├─ yes ─▶ cached embedding (instant)
                                              └─ no  ─▶ 30s Deezer preview ─▶ neural encoder
                                                                                    │
                        rank 87k-track index by audio+vibe similarity ◀────────────┘
```

Everything heavy (the 87k-track index + the neural encoder) is loaded **once**
when the server starts, so each right-click returns in well under a second for
library tracks.
