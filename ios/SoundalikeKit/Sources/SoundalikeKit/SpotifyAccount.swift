import CryptoKit
import Foundation
import Security

/// Optional Spotify connection (Web API, PKCE, no client secret).
///
/// With it, Soundalike reads the song Spotify is playing instead of listening for it,
/// and can queue results in Spotify. The Spotify app "soundalike" is in Development
/// mode, so only accounts added under User Management in the Spotify dashboard can
/// connect (Spotify allows a handful); everyone else uses listening or the share sheet.
public struct SpotifyAccount: Sendable {
    public static let clientID = "6aae8635d21e471cabdeacd2f226640e"
    public static let redirectURI = "soundalike://spotify-callback"
    public static let callbackScheme = "soundalike"
    static let scopes = "user-read-currently-playing user-read-playback-state user-modify-playback-state app-remote-control"

    public var http: HTTP
    let keychainGroup: String?

    /// `keychainGroup` is the app group, so the app and its extensions share the token.
    public init(http: HTTP = HTTP(), keychainGroup: String? = "group.app.yassin.soundalike") {
        self.http = http
        self.keychainGroup = keychainGroup
    }

    // MARK: Sign-in

    public struct Authorization: Sendable {
        /// Opens the Spotify app's own consent screen: no password, one tap on "Agree".
        public let appURL: URL
        /// The same request on accounts.spotify.com, for when Spotify isn't installed.
        public let webURL: URL
        let verifier: String
    }

    public static func authorization() -> Authorization {
        let verifier = randomURLSafe(64)
        let challenge = Data(SHA256.hash(data: Data(verifier.utf8))).base64URLEncoded
        let query: [String: String?] = [
            "client_id": clientID, "response_type": "code", "redirect_uri": redirectURI,
            "code_challenge_method": "S256", "code_challenge": challenge, "scope": scopes,
            "utm_source": "app.yassin.soundalike", "nosignup": "true", "nolinks": "true",
        ]
        // Same URL the official SpotifyLogin SDK opens (github.com/spotify/ios-auth).
        return Authorization(
            appURL: URL(string: "spotify-action://authorize")!.appending(query: query),
            webURL: URL(string: "https://accounts.spotify.com/authorize")!.appending(query: query),
            verifier: verifier
        )
    }

    public static func isCallback(_ url: URL) -> Bool {
        url.absoluteString.hasPrefix(redirectURI)
    }

    /// Finishes sign-in with the URL Spotify redirected to.
    public func complete(_ authorization: Authorization, callback: URL) async throws {
        let items = URLComponents(url: callback, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard let code = items.first(where: { $0.name == "code" })?.value else { throw SpotifyError.signInFailed }
        try await requestToken([
            "grant_type": "authorization_code", "code": code, "redirect_uri": Self.redirectURI,
            "client_id": Self.clientID, "code_verifier": authorization.verifier,
        ])
    }

    public var isConnected: Bool { loadToken() != nil }

    public func disconnect() {
        SecItemDelete(keychainQuery() as CFDictionary)
    }

    /// A valid access token (refreshed when needed), for the Spotify app connection (App Remote).
    public func accessToken() async throws -> String {
        guard var token = loadToken() else { throw SpotifyError.notConnected }
        if token.expires < Date().addingTimeInterval(60) {
            try await requestToken(["grant_type": "refresh_token", "refresh_token": token.refresh, "client_id": Self.clientID])
            guard let fresh = loadToken() else { throw SpotifyError.notConnected }
            token = fresh
        }
        return token.access
    }

    // MARK: Web API

    /// The song Spotify is playing on any of the user's devices, or nil when nothing is.
    public func currentlyPlaying() async throws -> Seed? {
        let (data, status) = try await api("GET", "me/player/currently-playing?additional_types=track")
        guard status == 200, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["is_playing"] as? Bool == true, let item = object["item"] as? [String: Any],
              let title = item["name"] as? String else { return nil }
        let artist = ((item["artists"] as? [[String: Any]])?.first?["name"] as? String) ?? ""
        let images = ((item["album"] as? [String: Any])?["images"] as? [[String: Any]]) ?? []
        let id = item["id"] as? String
        return Seed(
            title: title, artist: artist,
            isrc: (item["external_ids"] as? [String: Any])?["isrc"] as? String,
            artworkURL: (images.first?["url"] as? String).flatMap(URL.init(string:)),
            source: .spotify,
            sharedURL: id.map { URL(string: "https://open.spotify.com/track/\($0)")! },
            spotifyID: id
        )
    }

    /// Spotify's own id for a result, by ISRC.
    /// Cached, so the player bar can look up the next song ahead of time and Next plays at once.
    public func trackID(isrc: String) async throws -> String? {
        if let cached = TrackIDCache.shared.id(for: isrc) { return cached }
        let (data, status) = try await api("GET", "search?type=track&limit=1&q=" + ("isrc:" + isrc).addingPercentEncoding(withAllowedCharacters: .alphanumerics)!)
        guard status == 200, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let id = (((object["tracks"] as? [String: Any])?["items"] as? [[String: Any]])?.first?["id"] as? String)
        if let id { TrackIDCache.shared.store(id, for: isrc) }
        return id
    }

    /// Spotify's id for a result: from the server's link when there is one, else by ISRC.
    public func trackID(for match: Match) async throws -> String? {
        if let url = match.spotifyURL, case .spotifyTrack(let id, _)? = SharedLink.parse(url) { return id }
        guard let isrc = match.isrc else { return nil }
        return try await trackID(isrc: isrc)
    }

    /// Spotify's ids for a whole list, looked up side by side; nil where a song isn't on Spotify.
    public func trackIDs(for matches: [Match]) async -> [String?] {
        await withTaskGroup(of: (Int, String?).self) { group in
            for (index, match) in matches.enumerated() {
                group.addTask { (index, try? await trackID(for: match)) }
            }
            var ids = [String?](repeating: nil, count: matches.count)
            for await (index, id) in group { ids[index] = id }
            return ids
        }
    }

    /// Plays a list of tracks in Spotify, starting at one of them (Premium only). The list
    /// becomes what Spotify is playing, so its own skip buttons (lock screen, headphones,
    /// CarPlay) walk it, and choosing anything else in Spotify replaces it. Unlike the queue,
    /// nothing is left behind.
    public func play(trackIDs: [String], startingAt start: String) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "uris": trackIDs.map { "spotify:track:\($0)" },
            "offset": ["uri": "spotify:track:\(start)"],
        ])
        try await onDevice("PUT", "me/player/play", body: body)
    }

    /// Sends a player request. Spotify answers 404 "no active device" when its app is open on
    /// the phone but hasn't played yet (right after Connect Spotify, say); the request then
    /// goes to the phone's Spotify by device id instead of sending the user to Spotify.
    private func onDevice(_ method: String, _ path: String, body: Data? = nil) async throws {
        var (_, status) = try await api(method, path, body: body)
        if status == 404, let device = try await phoneDevice() {
            (_, status) = try await api(method, path + (path.contains("?") ? "&" : "?") + "device_id=" + device, body: body)
        }
        guard (200..<300).contains(status) else { throw status == 404 ? SpotifyError.noActiveDevice : SpotifyError.requestFailed(status) }
    }

    /// The phone's Spotify among the user's devices, percent-encoded; nil when it isn't running.
    /// Other devices are left alone, so music never starts on a computer across the room.
    private func phoneDevice() async throws -> String? {
        let (data, status) = try await api("GET", "me/player/devices")
        guard status == 200, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let devices = object["devices"] as? [[String: Any]] else { return nil }
        let phone = devices.first { ($0["type"] as? String)?.lowercased() == "smartphone" && $0["is_restricted"] as? Bool != true }
        return (phone?["id"] as? String)?.addingPercentEncoding(withAllowedCharacters: .alphanumerics)
    }

    /// Adds a track to the user's Spotify queue (Premium only).
    public func queue(trackID: String) async throws {
        try await onDevice("POST", "me/player/queue?uri=spotify%3Atrack%3A\(trackID)")
    }

    /// What the Spotify player is doing, for a remote control.
    public struct PlaybackState: Equatable, Sendable {
        public var seed: Seed
        public var isPlaying: Bool
        /// Seconds into the song.
        public var position: TimeInterval
        public var duration: TimeInterval
        /// The device Spotify is playing on. Commands name it, because a backgrounded
        /// Spotify app doesn't count as the active device and would get a 404.
        public var deviceID: String?
    }

    /// The Spotify player on any of the user's devices, playing or paused; nil when there is none.
    public func playbackState() async throws -> PlaybackState? {
        let (data, status) = try await api("GET", "me/player?additional_types=track")
        guard status == 200, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let item = object["item"] as? [String: Any], let title = item["name"] as? String else { return nil }
        let artist = ((item["artists"] as? [[String: Any]])?.map { $0["name"] as? String ?? "" } ?? []).joined(separator: ", ")
        let images = ((item["album"] as? [String: Any])?["images"] as? [[String: Any]]) ?? []
        let id = item["id"] as? String
        let seed = Seed(
            title: title, artist: artist,
            isrc: (item["external_ids"] as? [String: Any])?["isrc"] as? String,
            artworkURL: (images.first?["url"] as? String).flatMap(URL.init(string:)),
            source: .spotify,
            sharedURL: id.map { URL(string: "https://open.spotify.com/track/\($0)")! },
            spotifyID: id
        )
        return PlaybackState(
            seed: seed,
            isPlaying: object["is_playing"] as? Bool ?? false,
            position: TimeInterval(object["progress_ms"] as? Int ?? 0) / 1000,
            duration: TimeInterval(item["duration_ms"] as? Int ?? 0) / 1000,
            deviceID: (object["device"] as? [String: Any])?["id"] as? String
        )
    }

    public enum Command: Sendable {
        case play, pause, next, previous
        case seek(TimeInterval)
    }

    /// Sends a player command to the device Spotify is playing on (Premium only).
    public func send(_ command: Command, deviceID: String? = nil) async throws {
        var (method, path): (String, String) = switch command {
        case .play: ("PUT", "me/player/play")
        case .pause: ("PUT", "me/player/pause")
        case .next: ("POST", "me/player/next")
        case .previous: ("POST", "me/player/previous")
        case .seek(let seconds): ("PUT", "me/player/seek?position_ms=\(Int(seconds * 1000))")
        }
        if let deviceID, let encoded = deviceID.addingPercentEncoding(withAllowedCharacters: .alphanumerics) {
            path += (path.contains("?") ? "&" : "?") + "device_id=" + encoded
        }
        let (_, status) = try await api(method, path)
        guard (200..<300).contains(status) else { throw status == 404 ? SpotifyError.noActiveDevice : SpotifyError.requestFailed(status) }
    }

    // MARK: Plumbing

    struct Token: Codable {
        var access: String
        var refresh: String
        var expires: Date
    }

    func api(_ method: String, _ path: String, body: Data? = nil) async throws -> (Data, Int) {
        guard var token = loadToken() else { throw SpotifyError.notConnected }
        if token.expires < Date().addingTimeInterval(60) {
            try await requestToken(["grant_type": "refresh_token", "refresh_token": token.refresh, "client_id": Self.clientID])
            guard let fresh = loadToken() else { throw SpotifyError.notConnected }
            token = fresh
        }
        var request = URLRequest(url: URL(string: "https://api.spotify.com/v1/" + path)!, timeoutInterval: 10)
        request.httpMethod = method
        request.setValue("Bearer \(token.access)", forHTTPHeaderField: "Authorization")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        do {
            let (data, response) = try await http.session.data(for: request)
            return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
        } catch {
            throw SoundalikeError.offline
        }
    }

    func requestToken(_ form: [String: String]) async throws {
        var request = URLRequest(url: URL(string: "https://accounts.spotify.com/api/token")!, timeoutInterval: 15)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        request.httpBody = Data(form.map { "\($0.key)=\($0.value.addingPercentEncoding(withAllowedCharacters: allowed)!)" }.joined(separator: "&").utf8)
        let (data, response) = try await http.session.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let access = object["access_token"] as? String else {
            if form["grant_type"] == "refresh_token" { disconnect() }
            throw SpotifyError.signInFailed
        }
        let refresh = object["refresh_token"] as? String ?? loadToken()?.refresh ?? ""
        let expires = Date().addingTimeInterval(TimeInterval(object["expires_in"] as? Int ?? 3600))
        saveToken(Token(access: access, refresh: refresh, expires: expires))
    }

    func keychainQuery() -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "app.yassin.soundalike.spotify",
            kSecAttrAccount as String: "token",
        ]
        if let keychainGroup { query[kSecAttrAccessGroup as String] = keychainGroup }
        return query
    }

    func loadToken() -> Token? {
        var query = keychainQuery()
        query[kSecReturnData as String] = true
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return nil }
        return try? JSONDecoder().decode(Token.self, from: data)
    }

    func saveToken(_ token: Token) {
        guard let data = try? JSONEncoder().encode(token) else { return }
        SecItemDelete(keychainQuery() as CFDictionary)
        var item = keychainQuery()
        item[kSecValueData as String] = data
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(item as CFDictionary, nil)
    }

    static func randomURLSafe(_ bytes: Int) -> String {
        var buffer = [UInt8](repeating: 0, count: bytes)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes, &buffer)
        return Data(buffer).base64URLEncoded
    }
}

public enum SpotifyError: LocalizedError, Equatable {
    case notConnected
    case signInFailed
    case noActiveDevice
    case requestFailed(Int)

    public var errorDescription: String? {
        switch self {
        case .notConnected: "Connect Spotify first."
        case .signInFailed: "Spotify sign-in didn't work. This account may not be on the Soundalike beta list."
        case .noActiveDevice: "Start playing something in Spotify first."
        case .requestFailed(let code): "Spotify returned an error (\(code))."
        }
    }
}

extension Data {
    var base64URLEncoded: String {
        base64EncodedString().replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
    }
}

/// Spotify track ids by ISRC for this run of the app.
final class TrackIDCache: @unchecked Sendable {
    static let shared = TrackIDCache()
    private let lock = NSLock()
    private var ids: [String: String] = [:]

    func id(for isrc: String) -> String? {
        lock.withLock { ids[isrc] }
    }

    func store(_ id: String, for isrc: String) {
        lock.withLock { ids[isrc] = id }
    }
}
