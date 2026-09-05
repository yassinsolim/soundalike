# Soundalike for iOS and Android

A small companion app that answers one question: what else sounds like this?

Spotify does not let anything modify its mobile apps, so this cannot be an
extension the way the desktop Spicetify version is. Instead it registers itself
as a share target. You are listening to something in Spotify, you hit share, you
pick Soundalike, and you get twenty tracks that sound similar. Tapping any of
them opens it back in Spotify.

There is no sign-in. You never connect a Spotify account, and the app has no
Spotify developer credentials in it. It reads the public page for the track you
shared to learn the title and artist, looks that up in the same 272,853 track
catalog the website and the Spicetify extension use, and asks the same
recommendation service for matches.

## What it does

- Accepts a shared Spotify track from the iOS share sheet or the Android share
  menu.
- Lets you search the catalog directly if you would rather type a song name.
- Shows twenty results with cover art, artist, and measured tempo.
- Opens any result in the Spotify app.
- Has an optional Good, Mixed, or Off rating so you can tell me when the matches
  are wrong.

## What it does not do

- It does not play music. Playback always happens in Spotify.
- It does not read your library, your history, or your playlists.
- It does not know who you are. Feedback is anonymous and carries only the seed
  track, the results you were shown, and the reasons you picked.

## Updates

The app can update its own JavaScript without going through a store release, so
a bad match or a broken screen can be fixed in minutes instead of waiting days
for review. This uses EAS Update, which is run by Expo.

On launch the app asks Expo whether a newer bundle exists. That request carries
the platform it is running on, a runtime version, and the IP address any web
request would carry. It does not carry an account, a device identifier, or
anything about what you have been listening to. Nothing is uploaded.

The check happens in the background and never blocks startup. If an update is
found it is downloaded quietly and applied the next time the app is opened, so
a slow or missing network just means you keep running the version you already
have.

The runtime version uses the `fingerprint` policy, which derives it from the
native project itself, including every autolinked module and config plugin. A
JavaScript update can only reach a build whose native code actually matches it,
so adding a native dependency cannot accidentally push a bundle that crashes on
launch for people who already installed the app.

To publish one:

```bash
npx eas-cli update --branch preview --message "what changed"
```

Only JavaScript and assets can be shipped this way. Anything that changes native
code, which includes adding a dependency or touching the share extension, needs
a new build.

## Running it locally

You need Node 20 or newer.

```bash
cd mobile
npm install
```

To poke at the interface quickly in a browser:

```bash
npm run web
```

The browser build is only useful for checking layout and the search flow. Share
intents are a native feature and are switched off on web, and cover art will
fail because Deezer does not send CORS headers to browsers. Neither problem
exists on a real device.

For the real thing you need a development build, because the share extension is
native code that Expo Go does not include:

```bash
npx expo prebuild
npm run ios       # or: npm run android
```

## Icons

The icon is generated so it stays in step with the web app's brand, which is a
diamond mark on a green to violet gradient:

```bash
python scripts/make-icons.py
```

That rewrites everything in `assets/`. The Android background widens the
gradient on purpose, because Android crops an adaptive icon to its centre and a
plain corner to corner ramp would lose both brand colours.

## Checks

```bash
npm test        # unit tests for parsing, resolution, artwork, and feedback
npm run typecheck
```

## Building for a device

Builds go through [EAS](https://docs.expo.dev/build/introduction/). Register the
project once:

```bash
npx eas init
```

Then produce an installable build:

```bash
npm run build:android   # APK you can sideload for testing
npm run build:ios       # TestFlight build, needs an Apple Developer account
```

For store releases use the production profile:

```bash
npx eas build --platform android --profile production
npx eas build --platform ios --profile production
```

Submission uses `npm run submit:ios` and `npm run submit:android`. Fill in the
placeholder Apple and Google identifiers in `eas.json` first.

Do not commit the `ios/` and `android/` folders. They are generated, they are
already ignored, and EAS recreates them on every build.

The `.easignore` at the repository root matters here. EAS uploads the whole
repository, not just this folder, and the repository carries a 71 MB audio index
and a 12 MB model checkpoint that the app never reads. Without that file every
build uploads about 182 MB instead of a couple of megabytes.

## How a shared link becomes results

1. The share text is scanned for a Spotify track id. Short `spotify.link` URLs
   are followed until a real track id appears.
2. The public embed page for that track is read to get the title, the artist,
   and a cover image. If that fails, the app falls back to the public oEmbed
   endpoint and to any "Title by Artist" line the share sheet included.
3. The title and artist are looked up in the Soundalike catalog. An exact
   agreement on both continues straight to results. Anything less shows a short
   list so you can pick the right track.
4. The chosen track is sent to the recommendation service, which returns the
   ranked matches.

Steps 1 through 3 are the part most likely to need attention over time, since
they depend on pages Spotify controls. They live in `src/lib/resolve.ts` and are
covered by tests.

## Layout

```
App.tsx              screen switching and the share intent listener
src/lib/spotify.ts   parsing links and share text
src/lib/resolve.ts   turning a shared track into a catalog seed
src/lib/api.ts       talking to the search and recommendation endpoints
src/lib/artwork.ts   cover art lookup
src/lib/feedback.ts  the optional rating payload
src/components/      the four screens and the feedback bar
test/                unit tests
```

## Credits

Track names, artists, and the shared cover image come from Spotify's public
pages. Result cover art comes from Deezer's public API. Soundalike is not
affiliated with either company.
