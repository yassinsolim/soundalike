# Setup guide — connecting your accounts

The offline recommender (`soundalike similar` / `profile` / `stats`) needs **no setup**.
This guide is only for the **live** features that use your real Spotify taste and the
Last.fm similarity engine.

Your secrets go in a local `.env` file, which is git-ignored — they never get committed.

```bash
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```

---

## 1. Spotify (required for `login`, `whoami`, `pull`, `recommend`)

You need a free Spotify **app** to get a Client ID. This does **not** share your password —
you'll approve access in your browser via OAuth.

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create app**.
   - **App name / description:** anything (e.g. "soundalike").
   - **Redirect URI:** add exactly this and click **Add**:
     ```
     http://127.0.0.1:8888/callback
     ```
   - **Which API/SDKs are you planning to use?** check **Web API**.
   - Accept the terms and **Save**.
3. Open the app's **Settings** and copy the **Client ID**.
   (No Client Secret is needed — PKCE doesn't use one.)
4. Put it in your `.env`:
   ```
   SPOTIFY_CLIENT_ID=paste_your_client_id_here
   SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
   ```
5. **Add yourself to the app's allowlist (required for playlist creation).**
   New apps run in *Development mode*, where write actions like creating playlists are only
   allowed for accounts on the app's allowlist. In the app's **Settings → User Management**,
   click **Add user**, enter any name and the **email of your Spotify account**, and save.
   Reading your library/top/recent works without this, but `--playlist` needs it.
6. Authorize:
   ```bash
   soundalike login
   ```
   A browser tab opens; approve access. The token is cached in `~/.soundalike/` and refreshed
   automatically. Verify with `soundalike whoami`.

> **Development mode is fine.** New apps start in development mode, which is all you need for
> your own account (up to 5 allowlisted users). You do *not* need to request a quota extension
> for personal use.

> **Redirect URI must match exactly**, including `http`, `127.0.0.1` (not `localhost`), the
> port `8888`, and `/callback`. A mismatch is the #1 cause of login errors.

---

## 2. Last.fm (required only for `--engine lastfm`)

The Last.fm engine finds similar tracks for *any* song, not just the bundled dataset.

1. Create a free API account: <https://www.last.fm/api/account/create>
   (Callback URL can be left blank; we only make read requests.)
2. Copy the **API key** and add it to `.env`:
   ```
   LASTFM_API_KEY=paste_your_api_key_here
   ```
3. Use it:
   ```bash
   soundalike recommend --source liked --engine lastfm -n 25
   ```

---

## Troubleshooting

- **`Error: SPOTIFY_CLIENT_ID is not set`** — you haven't created `.env` or the value is blank.
- **`INVALID_CLIENT: Invalid redirect URI`** — the redirect URI in the dashboard doesn't match
  `SPOTIFY_REDIRECT_URI` exactly. Fix one to match the other.
- **`Spotify API 403: Forbidden` when using `--playlist`** — your account isn't on the app's
  allowlist. Open the app → **Settings → User Management** → add your Spotify account's name and
  email, then run `soundalike login` again. (Reading tracks works without this; only writes need it.)
- **Browser opens but nothing happens** — make sure port `8888` is free, or change both the
  dashboard redirect URI and `.env` to another port (e.g. `http://127.0.0.1:9090/callback`).
- **`recommend --engine content` finds few matches** — expected: the bundled dataset is ~855
  songs. Use `--engine lastfm` for full coverage.
- **Re-authorize from scratch** — delete `~/.soundalike/spotify_token.json` and run
  `soundalike login` again.
