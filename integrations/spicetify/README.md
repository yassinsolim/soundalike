# soundalike in Spotify — right-click → *Find soundalikes*

Add a **“Find soundalikes”** item to the right-click menu of any track in the
Spotify desktop app. Click it and a panel shows songs that genuinely *sound*
like the one you clicked — powered by the local neural audio model, not tags.

![Soundalike results inside Spotify](../../docs/spicetify-results.png)

There are two ways to use soundalike. Pick based on your Spotify install:

| | Web app (works with **any** Spotify) | Spicetify (in-app right-click) |
|---|---|---|
| Setup | none beyond `soundalike serve` | patch Spotify once; optional automatic local server |
| Supported Spotify app | any app or browser | patchable desktop app; not Windows Store or Linux Snap |
| Trigger | paste a song / **Copy Song Link** | right-click a track |

## Prepare Soundalike

Install [Python 3.9 or newer](https://www.python.org/downloads/) and Git, then
create an isolated environment from the repository root.

PowerShell (Windows):

```powershell
git clone https://github.com/yassinsolim/soundalike.git
Set-Location soundalike
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ml]"
```

Terminal (macOS/Linux):

```bash
git clone https://github.com/yassinsolim/soundalike.git
cd soundalike
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ml]"
```

If you already cloned the repository, start at its root and skip the first two
commands. Activate this environment again before running or configuring the
local server.

---

## Option A — the web app (recommended, zero client patching)

```bash
soundalike serve            # opens http://127.0.0.1:8787
```

Then either type `Title — Artist`, or — the frictionless way — in Spotify
**right-click a song → Share → Copy Song Link** and paste it. You get instant
soundalikes with an “Open in Spotify” button on each. This works with the
Microsoft-Store build of Spotify too.

---

## Option B — Spicetify (true in-app right-click)

Spicetify patches the Spotify **desktop** client to add custom menu items.
Windows users need standalone Spotify from <https://www.spotify.com/download>;
the Microsoft Store build cannot be patched. Linux users need a native or
Flatpak installation; Snap apps cannot be patched. These are Spicetify
limitations, not Soundalike limitations. Use Option A if your Spotify package
cannot be patched.

Before installing Spicetify, open the desktop Spotify app once, sign in, and
then close it. This creates the application and `prefs` paths Spicetify needs.

### 1. Install Spicetify

PowerShell (Windows):

```powershell
iwr -useb https://raw.githubusercontent.com/spicetify/cli/main/install.ps1 | iex

# Confirm this shell can find Spicetify and its standalone Spotify paths.
spicetify --version
spicetify config spotify_path
spicetify config prefs_path
```

If `spicetify` is not recognized after installation, open a new PowerShell
window. The official installer normally places the executable at
`$env:LOCALAPPDATA\spicetify\spicetify.exe`. If the new window still cannot
find it, add that directory to the current session and retry:

```powershell
$env:Path = "$env:LOCALAPPDATA\spicetify;$env:Path"
spicetify --version
```

Terminal (macOS):

```bash
curl -fsSL https://raw.githubusercontent.com/spicetify/cli/main/install.sh | sh

# Only needed if Spicetify does not detect Spotify automatically:
spicetify config spotify_path "/Applications/Spotify.app/Contents/Resources"

spicetify --version
spicetify config spotify_path
spicetify config prefs_path
```

The installer supports both Apple silicon and Intel Macs. Restart Terminal if
`spicetify` is not immediately available.

Terminal (Linux):

```bash
curl -fsSL https://raw.githubusercontent.com/spicetify/cli/main/install.sh | sh

# Restart the shell if necessary, then verify Spotify was detected.
spicetify --version
spicetify config spotify_path
spicetify config prefs_path
```

If the command is still unavailable, add the install directory to your shell
profile and restart the shell:

```bash
echo 'export PATH="$PATH:$HOME/.spicetify"' >> "$HOME/.bashrc"  # Bash
# echo 'export PATH="$PATH:$HOME/.spicetify"' >> "$HOME/.zshrc"  # Zsh
```

Linux Spotify packages need one additional setup:

- **APT:** `sudo chmod a+wr /usr/share/spotify` and
  `sudo chmod a+wr -R /usr/share/spotify/Apps`.
- **AUR:** use the same commands with `/opt/spotify`.
- **spotify-launcher:** run
  `spicetify config spotify_path "$HOME/.local/share/spotify-launcher/install/usr/share/spotify"`.
- **Flatpak:** locate the active package rather than hard-coding its CPU
  architecture:

  ```bash
  flatpak_root="$(flatpak info --show-location com.spotify.Client)"
  spotify_path="$flatpak_root/files/extra/share/spotify"
  spicetify config spotify_path "$spotify_path"
  spicetify config prefs_path "$HOME/.var/app/com.spotify.Client/config/spotify/prefs"

  # System-wide Flatpak installs may need write permission:
  sudo chmod a+wr "$spotify_path"
  sudo chmod a+wr -R "$spotify_path/Apps"
  ```

- **Snap:** uninstall it and install a patchable Spotify package; Snap cannot
  be modified by Spicetify.

The [official Linux package notes](https://spicetify.app/docs/getting-started#linux-specific-setup)
also cover NixOS and uncommon package layouts.

### 2. Install this extension

PowerShell (Windows):

```powershell
# Run from the soundalike repository root.
$extensions = Join-Path $env:APPDATA "spicetify\Extensions"
New-Item -ItemType Directory -Path $extensions -Force | Out-Null
Copy-Item integrations\spicetify\soundalike.js -Destination $extensions -Force

spicetify config extensions soundalike.js
spicetify apply
```

Do not use `$(spicetify config-dir)` as a path. That command opens the config
directory in Explorer but does not print its location, so PowerShell receives
an empty destination. The documented Windows extension directory is
`%APPDATA%\spicetify\Extensions`.

The warning `Config "extensions" unchanged` is harmless: it means
`soundalike.js` was already enabled. `spicetify apply` must still finish with
`success Refreshed extensions`.

Terminal (macOS/Linux):

```bash
# Run from the soundalike repository root.
extensions="$HOME/.config/spicetify/Extensions"
mkdir -p "$extensions"
cp integrations/spicetify/soundalike.js "$extensions/"

spicetify config extensions soundalike.js
spicetify apply
```

### 3. Run the local engine and use it

PowerShell (Windows):

```powershell
# Run from the soundalike repository root, with your virtual environment active.
soundalike serve --no-browser

# In a second PowerShell window, verify the server is ready:
Invoke-RestMethod http://127.0.0.1:8787/health
```

Terminal (macOS/Linux):

```bash
# Run from the soundalike repository root, with your virtual environment active.
soundalike serve --no-browser

# In a second Terminal window:
curl --fail http://127.0.0.1:8787/health
```

Now right-click any song in Spotify → **Find soundalikes**. The panel opens
immediately with the seed artwork and recommendation titles. Spotify album
covers and verified artist names fill in progressively; click a row to open
that track's Spotify search.

The extension installation is persistent: Spicetify remembers `soundalike.js`
and loads it whenever patched Spotify starts. The local recommendation engine
is separate and does **not** start automatically unless you enable the next
optional step.

### 4. Start the local engine automatically (optional)

#### Windows

Run this once from the repository root. Activate the same environment where
you installed `.[ml]` first; the installer records its exact Python path.

```powershell
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\integrations\spicetify\install-autostart.ps1
```

This creates a user-level **Soundalike Local Server** scheduled task, starts it
immediately, and starts it again at every Windows sign-in. It runs hidden and
does not need administrator access. Verify it at any time:

```powershell
powershell -ExecutionPolicy Bypass -File .\integrations\spicetify\install-autostart.ps1 -Action Status
Invoke-RestMethod http://127.0.0.1:8787/health
```

The current server log is
`$env:LOCALAPPDATA\Soundalike\server.log`. To remove auto-start and stop its
server:

```powershell
powershell -ExecutionPolicy Bypass -File .\integrations\spicetify\install-autostart.ps1 -Action Uninstall
```

Rerun the installer if you move the repository or replace its virtual
environment. Normal source updates need no task change.

#### macOS

Use a per-user `launchd` agent. Run these commands from the repository root
with the project virtual environment active:

```bash
python_bin="$(command -v python)"
repo_root="$PWD"
label="app.soundalike.local-server"
plist="$HOME/Library/LaunchAgents/$label.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

PLIST_PATH="$plist" LABEL="$label" PYTHON_BIN="$python_bin" \
REPO_ROOT="$repo_root" python - <<'PY'
import os
import plistlib

home = os.path.expanduser("~")
repo = os.environ["REPO_ROOT"]
config = {
    "Label": os.environ["LABEL"],
    "ProgramArguments": [
        os.environ["PYTHON_BIN"], "-m", "soundalike.cli", "serve", "--no-browser"
    ],
    "WorkingDirectory": repo,
    "EnvironmentVariables": {"PYTHONPATH": os.path.join(repo, "src")},
    "RunAtLoad": True,
    "KeepAlive": {"SuccessfulExit": False},
    "ThrottleInterval": 30,
    "StandardOutPath": os.path.join(home, "Library", "Logs", "soundalike.log"),
    "StandardErrorPath": os.path.join(home, "Library", "Logs", "soundalike.log"),
}
with open(os.environ["PLIST_PATH"], "wb") as stream:
    plistlib.dump(config, stream)
PY

launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$plist"
launchctl kickstart -k "gui/$(id -u)/$label"
curl --fail http://127.0.0.1:8787/health
```

To remove the macOS agent:

```bash
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/app.soundalike.local-server.plist"
rm "$HOME/Library/LaunchAgents/app.soundalike.local-server.plist"
```

#### Linux

Most desktop distributions use a per-user `systemd` service. Run this from the
repository root with the project virtual environment active:

```bash
python_bin="$(command -v python)"
repo_root="$PWD"
unit="$HOME/.config/systemd/user/soundalike.service"
mkdir -p "$(dirname "$unit")"

cat > "$unit" <<EOF
[Unit]
Description=Soundalike local recommendation server
After=network-online.target

[Service]
Type=simple
WorkingDirectory="$repo_root"
Environment="PYTHONPATH=$repo_root/src"
ExecStart="$python_bin" -m soundalike.cli serve --no-browser
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now soundalike.service
curl --fail http://127.0.0.1:8787/health
```

Check status and logs with `systemctl --user status soundalike.service` and
`journalctl --user -u soundalike.service`. Remove it with:

```bash
systemctl --user disable --now soundalike.service
rm "$HOME/.config/systemd/user/soundalike.service"
systemctl --user daemon-reload
```

On every supported platform, the extension reads the clicked track's title,
artist, and artwork through Spotify's already-authenticated internal GraphQL
client. Only the title and artist query are sent to
`http://127.0.0.1:8787` on your own machine. No separate Soundalike Spotify
login and no remote Soundalike server are required.

### Troubleshooting and updates

- **No “Find soundalikes” menu item:** confirm the file exists at
  `%APPDATA%\spicetify\Extensions\soundalike.js` on Windows or
  `~/.config/spicetify/Extensions/soundalike.js` on macOS/Linux, then run
  `spicetify config extensions soundalike.js`, `spicetify apply`, and fully
  restart Spotify. The extension waits for both the context-menu and React JSX
  APIs before registering; older copies that only waited for `ContextMenu` can
  fail during Spotify startup with a `ReactJSX` error.
- **Titles appear but covers stay blank:** Spotify's catalog lookup is still
  loading or did not find a confident title-and-artist match. Recommendations
  remain usable; confirm Spotify is online, then reopen the panel.
- **“Server not reachable”:** verify `/health`. If you enabled auto-start,
  check its status and log using step 4; otherwise keep
  `soundalike serve --no-browser` running.
- **After a Spotify update:** run `spicetify backup apply`. If Spicetify itself
  reports an available update, run `spicetify update` first.
- **After updating soundalike:** copy `soundalike.js` again using step 2, then
  run `spicetify apply`.

---

## How it works

```
right-click track ─▶ Spotify title + artist ─▶ local server /api/recommend
                                                    │
                           already in the library?  ├─ yes ─▶ cached embedding (instant)
                                                    └─ no  ─▶ 30s Deezer preview ─▶ neural encoder
                                                                                          │
                             rank 272,853-track index by audio+vibe similarity ◀──────────┘
                                                    │
                                                    └─▶ Spotify artwork + verified artist
```

On first use, the manifest may download the checksum-pinned 299 MB production
index; the bundled ~87k index remains the offline fallback. The selected
272,853-track index and neural encoder are loaded **once** when the server
starts, so subsequent right-clicks avoid model cold-start work.
