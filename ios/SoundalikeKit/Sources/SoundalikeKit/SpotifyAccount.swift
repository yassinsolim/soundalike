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
    static let scopes = "user-read-currently-playing user-read-playback-state user-modify-playback-state"

    public var http: HTTP
    let keychainGroup: String?

    /// `keychainGroup` is the app group, so the app and its extensions share the token.
    public init(http: HTTP = HTTP(), keychainGroup: String? = "group.app.yassin.soundalike") {
        self.http = http
        self.keychainGroup = keychainGroup
    }

    // MARK: Sign-in

    public struct Authorization: Sendable {
        public let url: URL
        let verifier: String
        let state: String
    }

    public static func authorization() -> Authorization {
        let verifier = randomURLSafe(64)
        let challenge = Data(SHA256.hash(data: Data(verifier.utf8))).base64URLEncoded
        let state = randomURLSafe(16)
        let url = URL(string: "https://accounts.spotify.com/authorize")!.appending(query: [
            "client_id": clientID, "response_type": "code", "redirect_uri": redirectURI,
            "code_challenge_method": "S256", "code_challenge": challenge, "scope": scopes, "state": state,
        ])
        return Authorization(url: url, verifier: verifier, state: state)
    }

    /// Finishes sign-in with the URL Spotify redirected to.
    public func complete(_ authorization: Authorization, callback: URL) async throws {
        let items = URLComponents(url: callback, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard items.first(where: { $0.name == "state" })?.value == authorization.state,
              let code = items.first(where: { $0.name == "code" })?.value else { throw SpotifyError.signInFailed }
        try await requestToken([
            "grant_type": "authorization_code", "code": code, "redirect_uri": Self.redirectURI,
            "client_id": Self.clientID, "code_verifier": authorization.verifier,
        ])
    }

    public var isConnected: Bool { loadToken() != nil }

    public func disconnect() {
        SecItemDelete(keychainQuery() as CFDictionary)
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
    public func trackID(isrc: String) async throws -> String? {
        let (data, status) = try await api("GET", "search?type=track&limit=1&q=" + ("isrc:" + isrc).addingPercentEncoding(withAllowedCharacters: .alphanumerics)!)
        guard status == 200, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return (((object["tracks"] as? [String: Any])?["items"] as? [[String: Any]])?.first?["id"] as? String)
    }

    /// Adds a track to the user's Spotify queue (Premium only).
    public func queue(trackID: String) async throws {
        let (_, status) = try await api("POST", "me/player/queue?uri=spotify%3Atrack%3A\(trackID)")
        guard (200..<300).contains(status) else { throw status == 404 ? SpotifyError.noActiveDevice : SpotifyError.requestFailed(status) }
    }

    // MARK: Plumbing

    struct Token: Codable {
        var access: String
        var refresh: String
        var expires: Date
    }

    func api(_ method: String, _ path: String) async throws -> (Data, Int) {
        guard var token = loadToken() else { throw SpotifyError.notConnected }
        if token.expires < Date().addingTimeInterval(60) {
            try await requestToken(["grant_type": "refresh_token", "refresh_token": token.refresh, "client_id": Self.clientID])
            guard let fresh = loadToken() else { throw SpotifyError.notConnected }
            token = fresh
        }
        var request = URLRequest(url: URL(string: "https://api.spotify.com/v1/" + path)!, timeoutInterval: 10)
        request.httpMethod = method
        request.setValue("Bearer \(token.access)", forHTTPHeaderField: "Authorization")
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
