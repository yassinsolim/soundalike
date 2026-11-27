import Foundation
import Testing
@testable import SoundalikeKit
@testable import SoundalikeUI

private func fixture(_ name: String) throws -> Data {
    try Data(contentsOf: #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures")))
}

@Suite struct PartParsingTests {
    @Test func partsPartAndExcerpt() throws {
        let results = try SoundalikeClient.parseMobile(try fixture("mobile-parts.json"), source: .spotify, sharedURL: nil)
        #expect(results.seed.deezerID == 908604612)
        let parts = try #require(results.parts)
        #expect(parts.minSeconds == 10 && parts.maxSeconds == 30 && parts.previewSeconds == 30)
        #expect(parts.songOffset == 48)
        #expect(parts.excerpt == ExcerptLimits(minSeconds: 10, maxSeconds: 30, maxBytes: 6_291_456))
        #expect(results.part == PartSpan(start: 10, end: 20.5))
    }

    @Test func wholePreviewHasNoPartNorExcerpt() throws {
        let results = try SoundalikeClient.parseMobile(try fixture("mobile-whole.json"), source: .spotify, sharedURL: nil)
        let parts = try #require(results.parts)
        #expect(parts.songOffset == nil)
        #expect(parts.excerpt == nil)
        #expect(results.part == nil)
    }

    @Test func serverWithoutPartsHasNoControl() throws {
        let body = #"{"ok":true,"query":{"title":"a","artist":"b"},"results":[{"title":"c","artist":"d","match":50}]}"#
        let results = try SoundalikeClient.parseMobile(Data(body.utf8), source: .spotify, sharedURL: nil)
        #expect(results.parts == nil)
        #expect(results.part == nil)
    }

    @Test func previewURLs() throws {
        let results = try SoundalikeClient.parseMobile(try fixture("mobile-parts.json"), source: .spotify, sharedURL: nil)
        #expect(results.matches[0].previewURL?.host == "cdnt-preview.dzcdn.net")
        #expect(results.matches[1].previewURL == nil)
        let fresh = Data(#"{"ok": true, "preview": "https://cdnt-preview.dzcdn.net/api/1/x.mp3?hdnea=1", "cover": ""}"#.utf8)
        #expect(SoundalikeClient.parsePreview(fresh)?.absoluteString == "https://cdnt-preview.dzcdn.net/api/1/x.mp3?hdnea=1")
        #expect(SoundalikeClient.parsePreview(Data(#"{"ok": false, "error": "no preview"}"#.utf8)) == nil)
    }

    @Test func excerptAnswer() throws {
        let seed = Seed(title: "Blinding Lights", artist: "The Weeknd", source: .spotify, deezerID: 908604612)
        let results = try SoundalikeClient.parseAnysong(try fixture("excerpt.json"), seed: seed)
        #expect(results.matches.map(\.title) == ["STAY", "Blinding Lights"])
        #expect(results.parts?.excerpt?.maxBytes == 6_291_456)
        #expect(results.seed.deezerID == 908604612)
    }

    @Test func partQueryHasAtMostOneDecimal() {
        #expect(PartSpan(start: 10, end: 20).query == "10-20")
        #expect(PartSpan(start: 2.5, end: 12.54).query == "2.5-12.5")
        #expect(PartSpan(start: 0.04, end: 29.99).query == "0-30")
    }
}

@Suite struct PartLabelTests {
    @Test func songTimesOnlyWhenTheOffsetIsKnown() {
        let span = PartSpan(start: 10, end: 25)
        #expect(PartTimes(parts: PartLimits(songOffset: 48)).span(span) == "0:58–1:13 of the song")
        #expect(PartTimes(parts: PartLimits(songOffset: nil)).span(span) == "0:10–0:25 of the preview")
    }

    @Test func chips() {
        let seed = Seed(title: "a", artist: "b", source: .spotify)
        var results = SoundalikeResults(seed: seed, matches: [], route: "mobile", parts: PartLimits(songOffset: 48), part: PartSpan(start: 10, end: 25))
        #expect(PartRow.chip(for: results) == "Matching 0:58–1:13 of the song")
        results.part = nil
        #expect(PartRow.chip(for: results) == nil)
        results.clip = ClipInfo(source: .file, start: 62, end: 77)
        #expect(PartRow.chip(for: results) == "Matching your clip (1:02–1:17 of your file)")
        results.clip = ClipInfo(source: .recording, start: 0, end: 15)
        #expect(PartRow.chip(for: results) == "Matching your clip (0:00–0:15 of your recording)")
    }

    @Test func clipSourcesOnlyWithExcerpt() {
        #expect(PartPanel.clipLengths(PartLimits(), allowsClips: true).isEmpty)
        let excerpt = PartLimits(excerpt: ExcerptLimits(minSeconds: 10, maxSeconds: 30, maxBytes: 6_291_456))
        #expect(PartPanel.clipLengths(excerpt, allowsClips: true) == [10, 15, 20])
        #expect(PartPanel.clipLengths(excerpt, allowsClips: false).isEmpty)
        // Clamped to the server's limits and size.
        #expect(AudioClip.lengths(for: ExcerptLimits(minSeconds: 12, maxSeconds: 30, maxBytes: 6_291_456)) == [15, 20])
        #expect(AudioClip.lengths(for: ExcerptLimits(minSeconds: 10, maxSeconds: 30, maxBytes: 700_000)) == [10, 15])
    }

    @Test func previewLengthsFollowTheLimits() {
        #expect(PartPanel.previewLengths(PartLimits()) == [10, 15, 20])
        #expect(PartPanel.previewLengths(PartLimits(minSeconds: 15)) == [15, 20])
    }
}

@Suite struct WAVTests {
    private func u32(_ data: Data, _ at: Int) -> UInt32 { data.subdata(in: at..<at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian }
    private func u16(_ data: Data, _ at: Int) -> UInt16 { data.subdata(in: at..<at + 2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian }

    @Test func header() {
        let wav = WAV.encode([0, 1, -1, 0.5], sampleRate: 22_050)
        #expect(wav.count == 44 + 8)
        #expect(String(decoding: wav[0..<4], as: UTF8.self) == "RIFF")
        #expect(u32(wav, 4) == UInt32(wav.count - 8))
        #expect(String(decoding: wav[8..<16], as: UTF8.self) == "WAVEfmt ")
        #expect(u32(wav, 16) == 16)
        #expect(u16(wav, 20) == 1) // PCM
        #expect(u16(wav, 22) == 1) // mono
        #expect(u32(wav, 24) == 22_050)
        #expect(u32(wav, 28) == 44_100) // bytes per second
        #expect(u16(wav, 32) == 2)
        #expect(u16(wav, 34) == 16)
        #expect(String(decoding: wav[36..<40], as: UTF8.self) == "data")
        #expect(u32(wav, 40) == 8)
        #expect(Int16(bitPattern: u16(wav, 46)) == 32767)
        #expect(Int16(bitPattern: u16(wav, 48)) == -32767)
    }

    @Test func sizeOfAClip() {
        // 20 s at 22,050 Hz, 16-bit mono: 882,000 bytes of samples, well under the server's 6 MB.
        #expect(WAV.size(frames: 20 * AudioClip.sampleRate) == 882_044)
        #expect(WAV.encode([Float](repeating: 0.1, count: 15 * AudioClip.sampleRate)).count == WAV.size(frames: 15 * AudioClip.sampleRate))
    }

    @Test func resampleAndSilence() throws {
        let tone = (0..<48_000).map { Float(sin(Double($0) * 2 * .pi * 440 / 48_000)) * 0.2 }
        let clip = try AudioClip.resample(tone, from: 48_000)
        #expect(abs(clip.count - 22_050) < 64)
        #expect(!AudioClip.isSilent(clip))
        #expect(AudioClip.isSilent([Float](repeating: 0.0005, count: 1000)))
        #expect(abs((AudioClip.normalized(tone).map(abs).max() ?? 0) - 0.98) < 0.001)
    }
}

@Suite struct ExcerptErrorTests {
    private func error(_ status: Int, _ body: String = "{}", retryAfter: String? = nil) -> SoundalikeError? {
        SoundalikeClient.excerptError(status: status, retryAfter: retryAfter, body: Data(body.utf8))
    }

    @Test func mapping() {
        #expect(error(200) == nil)
        #expect(error(429, retryAfter: "10") == .rateLimited(retryAfter: 10))
        #expect(error(503, retryAfter: "2") == .busy(retryAfter: 2))
        #expect(error(503) == .busy(retryAfter: nil))
        #expect(error(404, #"{"ok":false,"error":"unknown track"}"#) == .clip("Soundalike couldn't find this song to match your clip against."))
        #expect(error(404, #"{"ok":false,"error":"not found"}"#) == .clip("Matching your own clip isn't available right now."))
        #expect(error(413)?.errorDescription?.contains("too large") == true)
        #expect(error(415)?.errorDescription?.contains("can't read") == true)
        #expect(error(422, #"{"ok":false,"error":"the clip is silent"}"#)?.errorDescription?.contains("silent") == true)
        #expect(error(422, #"{"ok":false,"error":"clip must be 10 to 30 s"}"#)?.errorDescription?.contains("10 to 30 seconds") == true)
        #expect(error(408)?.errorDescription?.contains("too long to upload") == true)
        #expect(error(502) == .server(502))
        #expect(SoundalikeError.rateLimited(retryAfter: 10).errorDescription == "That's a lot of clips at once. Try again in 10 seconds.")
        #expect(SoundalikeError.busy(retryAfter: 2).errorDescription == "Soundalike is matching another clip. Try again in 2 seconds.")
    }
}

/// Answers requests from a closure, so the client's fallbacks run without a network.
final class StubProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) -> (Int, [String: String], Data))?
    nonisolated(unsafe) static var requests: [URLRequest] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        let (status, headers, body) = Self.handler?(request) ?? (500, [:], Data())
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    static func client() -> SoundalikeClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        requests = []
        return SoundalikeClient(http: HTTP(session: URLSession(configuration: configuration)))
    }
}

@Suite(.serialized) struct PartRequestTests {
    let seed = Seed(title: "Blinding Lights", artist: "The Weeknd", source: .spotify, deezerID: 908604612)

    private func part(_ request: URLRequest) -> String? {
        URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "part" }?.value
    }

    @Test func partIsSentAndEchoed() async throws {
        let client = StubProtocol.client()
        let body = try fixture("mobile-parts.json")
        StubProtocol.handler = { _ in (200, [:], body) }
        let results = try await client.similar(to: seed, part: PartSpan(start: 10, end: 20.5))
        #expect(part(StubProtocol.requests[0]) == "10-20.5")
        #expect(results.part == PartSpan(start: 10, end: 20.5))
    }

    @Test func partNotEchoedIsDropped() async throws {
        let client = StubProtocol.client()
        let body = try fixture("mobile-whole.json")
        StubProtocol.handler = { _ in (200, [:], body) }
        let results = try await client.similar(to: seed, part: PartSpan(start: 10, end: 20))
        #expect(StubProtocol.requests.count == 1)
        #expect(results.part == nil)
        #expect(PartRow.chip(for: results) == nil)
    }

    @Test func refusedPartIsAskedOnceWithout() async throws {
        let client = StubProtocol.client()
        let body = try fixture("mobile-whole.json")
        StubProtocol.handler = { request in
            URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems?.contains { $0.name == "part" } == true
                ? (404, [:], Data(#"{"ok":false,"error":{"code":"invalid_parameter","message":"no"}}"#.utf8))
                : (200, [:], body)
        }
        let results = try await client.similar(to: seed, part: PartSpan(start: 10, end: 20))
        #expect(StubProtocol.requests.count == 2)
        #expect(part(StubProtocol.requests[0]) == "10-20")
        #expect(part(StubProtocol.requests[1]) == nil)
        #expect(results.part == nil)
        #expect(results.matches.count == 1)
    }

    @Test func clipIsPostedWithTheSongAndLength() async throws {
        let client = StubProtocol.client()
        let body = try fixture("excerpt.json")
        StubProtocol.handler = { request in
            request.url?.host == "api.deezer.com" ? (200, [:], Data(#"{"isrc":"X","album":{"cover_big":"https://c/x.jpg"}}"#.utf8)) : (200, [:], body)
        }
        let wav = WAV.encode([Float](repeating: 0.1, count: 10 * AudioClip.sampleRate))
        let results = try await client.similar(toClip: wav, seed: seed)
        let post = try #require(StubProtocol.requests.first { $0.url?.path == "/api/anysong/excerpt" })
        #expect(post.httpMethod == "POST")
        #expect(post.value(forHTTPHeaderField: "Content-Length") == String(wav.count))
        let items = URLComponents(url: post.url!, resolvingAgainstBaseURL: false)?.queryItems ?? []
        #expect(items.first { $0.name == "id" }?.value == "908604612")
        #expect(results.route == "excerpt")
        #expect(results.matches.first?.artworkURL != nil)
    }

    @Test func refusedClipThrowsItsMessage() async throws {
        let client = StubProtocol.client()
        StubProtocol.handler = { _ in (429, ["Retry-After": "10"], Data(#"{"ok":false,"error":"too many requests"}"#.utf8)) }
        await #expect(throws: SoundalikeError.rateLimited(retryAfter: 10)) {
            try await client.similar(toClip: WAV.encode([0.1]), seed: seed)
        }
    }
}

/// Against production: `SOUNDALIKE_LIVE=1 swift test --filter LiveTests`.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["SOUNDALIKE_LIVE"] == "1"))
struct LiveTests {
    @Test func previewPartIsEchoed() async throws {
        let seed = Seed(title: "Blinding Lights", artist: "The Weeknd", source: .spotify)
        let results = try await SoundalikeClient().similar(to: seed, k: 5, part: PartSpan(start: 10, end: 20))
        #expect(results.route == "mobile")
        #expect(results.part == PartSpan(start: 10, end: 20))
        #expect(results.parts != nil)
        #expect(results.seed.deezerID != nil)
        #expect(results.matches.contains { $0.previewURL != nil })
        let fresh = try await SoundalikeClient().previewURL(deezerID: try #require(results.seed.deezerID))
        #expect(fresh.host?.hasSuffix("dzcdn.net") == true)
        print("live: part \(results.part!.query), parts \(results.parts!), chip \(PartRow.chip(for: results) ?? "-")")
    }
}
