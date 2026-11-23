# Releasing Soundalike for Android

## Cut a release

1. Set `versionCode` (must go up with every release) and `versionName` in `app/build.gradle.kts`.
2. Run what CI runs, then build the signed APK:

   ```powershell
   . D:\android\env.ps1
   cd android
   .\gradlew.bat :core:test :app:lintDebug :app:assembleRelease
   ```

   The APK is `app/build/outputs/apk/release/app-release.apk`, shrunk with R8. For Google Play build the bundle
   instead: `.\gradlew.bat :app:bundleRelease` (`app/build/outputs/bundle/release/app-release.aab`).
3. Check the signature and note the hash:

   ```powershell
   & "$env:ANDROID_HOME\build-tools\37.0.0\apksigner.bat" verify --print-certs app\build\outputs\apk\release\app-release.apk
   Get-FileHash app\build\outputs\apk\release\app-release.apk -Algorithm SHA256
   ```

   The signer's SHA-256 must be `1E:CE:72:59:6B:BF:02:D1:61:FE:8E:6F:83:EA:31:70:F5:69:C4:65:A7:79:21:60:80:D3:45:BC:34:48:EF:3B`,
   the one in `webapp/well-known/assetlinks.json`; with another key the `/find` links stop opening the app.
4. Install it over the previous release on a phone (`adb install -r`), run the short pass in
   `README.md` ("What was verified"), then attach the APK to the GitHub release with its SHA-256.

The same source, the same Gradle wrapper and the same JDK give the same code and resources. Nothing is fetched at
build time except the dependencies pinned in `gradle/libs.versions.toml`.

## The signing key

The key is not in the repository and must never be. Gradle reads the property `soundalike.keystoreProperties`
(set it in `GRADLE_USER_HOME/gradle.properties`), which names a properties file with `storeFile`, `storePassword`,
`keyAlias` and `keyPassword`. Without it the release APK is left unsigned.

On the development PC: `D:\android\keys\soundalike-release.jks` (PKCS12, alias `soundalike`, RSA 4096, valid to
2056), passwords in `D:\android\keys\soundalike-release.properties`. Back both up somewhere safe: an app can only
be updated by the key that first signed it. If the app goes on Google Play with Play App Signing, this key becomes
the upload key and Play's signing certificate has to be added to `assetlinks.json`.

## Google Play checklist

Read the current policy pages when submitting; they change, and the points below were not confirmed in a Play
Console review.

- **Store listing.** Name "Soundalike". No Spotify, Apple or YouTube name or logo in the title, icon or
  screenshots' captions; the description may say which apps it works with. Link the privacy policy
  (`android/PRIVACY.md`, hosted at a public address).
- **Notification access** (`NotificationListenerService`). It is sensitive access to personal data, so Play's User
  Data policy applies: the use must be the app's evident purpose, disclosed prominently in the app before the
  system screen opens, and described in the privacy policy. The first-run guide and the results sheet both explain
  it before sending the user to the system screen. Be ready to state in the review notes that the service reads
  media sessions only and no notification content.
- **Display over other apps** (`SYSTEM_ALERT_WINDOW`) and **usage access** (`PACKAGE_USAGE_STATS`). Both exist only
  for the optional floating button and are requested only when the user turns it on. If review objects, remove the
  floating button, both permissions and `ForegroundApp.kt`; nothing else depends on them.
- **Data safety form.** Data collected: none tied to a person. Data shared: none. The app sends the looked-up
  song's title, artist and ids to its own server to perform the search; that is "app functionality", not linked to
  identity, not stored with an identifier, sent over https. No account, no advertising id, no analytics, no crash
  reporting library.
- **Target API.** `targetSdk` 36. Play requires a recent target level for new apps; raise it when Play asks.
- **Content rating and ads.** No ads, no user-generated content, no purchases.
- **Accessibility service: not used, by decision.** It would let the app read and act on other apps' screens,
  Play restricts it to accessibility purposes with a declaration and video, and it could not add anything inside
  Spotify that the media session, the share sheet and the floating button do not already give.
- **No modification of other apps.** The app talks to music apps only through Android's public interfaces: the
  share sheet, media sessions and links the apps themselves register.
