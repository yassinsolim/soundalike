import Foundation

/// Thin wrapper so every request has a timeout and maps transport failures to `.offline`.
public struct HTTP: Sendable {
    public var session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func get(_ url: URL, timeout: TimeInterval = 15, headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        return try await send(request)
    }

    /// POSTs `body` with its `Content-Length`.
    public func post(_ url: URL, body: Data, timeout: TimeInterval = 15, headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = "POST"
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        request.setValue(String(body.count), forHTTPHeaderField: "Content-Length")
        request.httpBody = body
        return try await send(request)
    }

    private func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw SoundalikeError.offline }
            return (data, http)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
        } catch let error as SoundalikeError {
            throw error
        } catch {
            throw SoundalikeError.offline
        }
    }

    public func json(_ url: URL, timeout: TimeInterval = 15, headers: [String: String] = [:]) async throws -> Any {
        let (data, _) = try await get(url, timeout: timeout, headers: headers)
        return try JSONSerialization.jsonObject(with: data)
    }
}

extension URL {
    func appending(query: [String: String?]) -> URL {
        var components = URLComponents(url: self, resolvingAgainstBaseURL: false)!
        var items = components.queryItems ?? []
        for (name, value) in query.sorted(by: { $0.key < $1.key }) {
            if let value, !value.isEmpty { items.append(URLQueryItem(name: name, value: value)) }
        }
        components.queryItems = items
        // "+" survives URLComponents unescaped and many servers read it as a space.
        components.percentEncodedQuery = components.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B")
        return components.url!
    }
}

/// Loose title and artist comparison, so "Song (Remastered 2011)" still matches "Song".
public enum TextMatch {
    public static func normalize(_ value: String) -> String {
        let folded = value.folding(options: [.diacriticInsensitive, .caseInsensitive, .widthInsensitive], locale: nil)
        let scalars = folded.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
        return String(scalars).split(separator: " ").joined(separator: " ")
    }

    /// Equal after normalising, or one starts with the other ("Song" / "Song - Remastered",
    /// "Artist" / "Artist, Guest").
    public static func similar(_ a: String, _ b: String) -> Bool {
        let a = normalize(a), b = normalize(b)
        return !a.isEmpty && !b.isEmpty && (a == b || a.hasPrefix(b) || b.hasPrefix(a))
    }

    public static func coreTitle(_ value: String) -> String {
        var working = value
        if let range = working.range(of: #"\s*[\(\[]?\s*(feat\.?|ft\.?|featuring|with)\s+.*$"#, options: [.regularExpression, .caseInsensitive]) {
            working.removeSubrange(range)
        }
        var previous = ""
        while previous != working {
            previous = working
            if let range = working.range(of: #"\s*[\(\[][^\)\]]*[\)\]]\s*$|\s+-\s+[^-]*$"#, options: .regularExpression) {
                working.removeSubrange(range)
            }
        }
        return normalize(working.isEmpty ? value : working)
    }

    public static func primaryArtist(_ value: String) -> String {
        let first = value.components(separatedBy: CharacterSet(charactersIn: ",&")).first ?? value
        let cut = first.range(of: #"\s+(feat\.?|ft\.?|featuring|x|and)\s+.*$"#, options: [.regularExpression, .caseInsensitive])
        return normalize(cut.map { String(first[..<$0.lowerBound]) } ?? first)
    }

    /// Higher is better; 0 means unrelated. Live and remix versions lose to the plain one.
    public static func score(title: String, artist: String, candidateTitle: String, candidateArtist: String) -> Int {
        var score = 0
        let wanted = coreTitle(title), got = coreTitle(candidateTitle)
        if wanted == got { score += 4 } else if !wanted.isEmpty, got.contains(wanted) || wanted.contains(got) { score += 1 }
        if primaryArtist(artist) == primaryArtist(candidateArtist) { score += 3 }
        guard score > 0 else { return 0 }
        let lowered = candidateTitle.lowercased()
        if !title.lowercased().contains("live"), lowered.contains("live") { score -= 1 }
        if !title.lowercased().contains("remix"), lowered.contains("remix") { score -= 1 }
        return max(score, 0)
    }
}
