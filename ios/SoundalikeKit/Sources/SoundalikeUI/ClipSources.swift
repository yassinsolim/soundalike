#if os(iOS)
import AVFoundation
import SoundalikeKit
import SwiftUI
import UniformTypeIdentifiers

/// "Listen to this part": the song plays out loud in the music app at the part the listener wants, and the
/// microphone records 10, 15 or 20 seconds of it. The recording stays in memory, is sent once and is not kept.
struct ListenSource: View {
    let lengths: [Double]
    @ObservedObject var player: PreviewPlayer
    let send: (PartChoice) -> Void
    @StateObject private var recorder = ClipRecorder()
    @State private var length: Double
    @State private var task: Task<Void, Never>?
    @State private var failure: String?

    init(lengths: [Double], player: PreviewPlayer, send: @escaping (PartChoice) -> Void) {
        self.lengths = lengths
        self.player = player
        self.send = send
        _length = State(initialValue: lengths.contains(15) ? 15 : lengths.first ?? 10)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Listen to this part").font(.headline)
            Text("Play the song out loud in your music app, at the part you want, and hold your phone near the speaker. Soundalike records a few seconds with the microphone and matches just that. Headphones won't work: the microphone can't hear them.")
                .font(.subheadline).foregroundStyle(.secondary)
            if recorder.isRecording {
                VStack(spacing: 12) {
                    Text("\(recorder.remaining)").font(.system(size: 44, weight: .bold).monospacedDigit())
                        .contentTransition(.numericText(countsDown: true))
                        .animation(.default, value: recorder.remaining)
                    LevelMeter(level: recorder.level)
                    Text("Recording…").foregroundStyle(.secondary)
                    Button("Cancel", role: .cancel) { task?.cancel() }.buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Recording, \(recorder.remaining) seconds left")
            } else {
                Picker("Length", selection: $length) {
                    ForEach(lengths, id: \.self) { Text("\(Int($0)) s").tag($0) }
                }
                .pickerStyle(.segmented)
                Button("Record \(Int(length)) seconds", systemImage: "mic.fill") { record() }
                    .buttonStyle(.borderedProminent)
                    .frame(maxWidth: .infinity)
            }
            if let failure {
                Text(failure).font(.subheadline).foregroundStyle(.red)
            }
            Text("The recording is sent once to Soundalike to find soundalikes and isn't kept, on your phone or on the server.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .onDisappear { task?.cancel() }
    }

    private func record() {
        player.stop()
        failure = nil
        let seconds = length
        task = Task {
            let started = Date()
            do {
                let samples = try await recorder.record(seconds: seconds)
                EventLog.record("clip-record", ["ms": EventLog.ms(since: started), "ok": true, "seconds": Int(seconds)])
                let wav = WAV.encode(AudioClip.normalized(samples))
                send(.clip(wav: wav, info: ClipInfo(source: .recording, start: 0, end: Double(samples.count) / Double(AudioClip.sampleRate))))
            } catch is CancellationError {
                EventLog.record("clip-record", ["ms": EventLog.ms(since: started), "ok": false, "error": "cancelled"])
            } catch {
                EventLog.record("clip-record", ["ms": EventLog.ms(since: started), "ok": false, "error": "\(error)"])
                failure = error.localizedDescription
            }
        }
    }
}

struct LevelMeter: View {
    let level: Float

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.2))
                Capsule().fill(Color.green).frame(width: geo.size.width * CGFloat(min(max(level, 0), 1)))
                    .animation(.linear(duration: 0.1), value: level)
            }
        }
        .frame(height: 8)
        .accessibilityHidden(true)
    }
}

/// "From a file": an audio file of the song, decoded on the phone. Only the chosen part is sent, once.
struct FileSource: View {
    let lengths: [Double]
    @ObservedObject var player: PreviewPlayer
    let send: (PartChoice) -> Void
    @State private var picking = false
    @State private var file: URL?
    @State private var duration: Double = 0
    @State private var span = PartSpan(start: 0, end: 15)
    @State private var failure: String?

    private static let playerID = "match-a-part-file"

    private var usable: [Double] { lengths.filter { $0 <= duration } }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("From a file").font(.headline)
            Text("Pick an audio file of the song. It's read on your phone; only the part you choose is sent, once, and it isn't kept.")
                .font(.subheadline).foregroundStyle(.secondary)
            Button(file == nil ? "Choose a file" : "Choose another file", systemImage: "doc.badge.plus") { picking = true }
                .buttonStyle(.bordered)
            if let file, !usable.isEmpty {
                Text(file.lastPathComponent).font(.subheadline.weight(.semibold)).lineLimit(1)
                let playing = player.isActive(Self.playerID)
                SpanBar(total: duration, span: $span, step: 1, lengths: usable, label: timeLabel, playhead: playing ? player.position : nil)
                Text("\(timeLabel(span.start))–\(timeLabel(span.end)) of your file").font(.subheadline.monospacedDigit().weight(.semibold))
                HStack {
                    Button(playing ? "Stop" : "Play this part", systemImage: playing ? "stop.fill" : "play.fill") {
                        if playing {
                            player.stop()
                        } else {
                            Task { await player.play(id: Self.playerID, url: file, deezerID: nil, span: span.start...span.end) }
                        }
                    }
                    .buttonStyle(.bordered)
                    Spacer()
                    Button("Apply") { apply(file) }.buttonStyle(.borderedProminent)
                }
            }
            if let failure {
                Text(failure).font(.subheadline).foregroundStyle(.red)
            }
        }
        .fileImporter(isPresented: $picking, allowedContentTypes: [.audio]) { result in
            switch result {
            case .success(let url): open(url)
            case .failure(let error): failure = error.localizedDescription
            }
        }
        .onDisappear { close() }
    }

    private func open(_ url: URL) {
        close()
        failure = nil
        guard url.startAccessingSecurityScopedResource() else {
            failure = "Soundalike couldn't open that file."
            return
        }
        do {
            let seconds = try AudioClip.duration(of: url)
            guard let shortest = lengths.min(), seconds >= shortest else {
                url.stopAccessingSecurityScopedResource()
                failure = "That file is shorter than \(Int(lengths.min() ?? 10)) seconds."
                return
            }
            file = url
            duration = seconds
            let length = usable.contains(15) ? 15 : usable.first ?? shortest
            let start = min(max(0, (seconds / 2 - length / 2).rounded()), seconds - length)
            span = PartSpan(start: start, end: start + length)
            EventLog.record("clip-file", ["ok": true, "seconds": Int(seconds)])
        } catch {
            url.stopAccessingSecurityScopedResource()
            EventLog.record("clip-file", ["ok": false, "error": "\(error)"])
            failure = error.localizedDescription
        }
    }

    private func apply(_ url: URL) {
        player.stop()
        failure = nil
        do {
            let samples = try AudioClip.read(url, from: span.start, to: span.end)
            guard !AudioClip.isSilent(samples) else {
                failure = "That part of the file is silent. Pick another part."
                return
            }
            send(.clip(wav: WAV.encode(AudioClip.normalized(samples)), info: ClipInfo(source: .file, start: span.start, end: span.end)))
        } catch {
            failure = error.localizedDescription
        }
    }

    private func close() {
        player.stop()
        file?.stopAccessingSecurityScopedResource()
        file = nil
    }
}

/// Records a clip with the microphone, while the music app keeps playing (the session mixes with others).
@MainActor
final class ClipRecorder: ObservableObject {
    enum Failure: LocalizedError {
        case permission
        case privateOutput(String)
        case silent
        case unavailable

        var errorDescription: String? {
            switch self {
            case .permission: "Soundalike needs the microphone to record a part. Allow it in Settings › Privacy & Security › Microphone."
            case .privateOutput(let name): "Music is playing through \(name), so the microphone can't hear it. Play it out loud on the phone's speaker or a speaker in the room, then try again."
            case .silent: "Soundalike heard almost nothing. Play the song out loud, turn it up, and try again."
            case .unavailable: "The microphone isn't available right now. Try again in a moment."
            }
        }
    }

    @Published private(set) var isRecording = false
    @Published private(set) var remaining = 0
    @Published private(set) var level: Float = 0

    /// One channel at `AudioClip.sampleRate`.
    func record(seconds: Double) async throws -> [Float] {
        guard await AVAudioApplication.requestRecordPermission() else { throw Failure.permission }
        let session = AVAudioSession.sharedInstance()
        if let output = Self.privateOutput(session.currentRoute) { throw Failure.privateOutput(output) }
        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.mixWithOthers, .defaultToSpeaker])
            try session.setActive(true)
        } catch {
            throw Failure.unavailable
        }
        let engine = AVAudioEngine()
        let collector = Collector()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        defer {
            engine.stop()
            input.removeTap(onBus: 0)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            isRecording = false
            level = 0
        }
        guard format.sampleRate > 0, format.channelCount > 0 else { throw Failure.unavailable }
        input.installTap(onBus: 0, bufferSize: 4096, format: format, block: Self.tap(collector))
        do { try engine.start() } catch { throw Failure.unavailable }
        isRecording = true
        let started = Date()
        while true {
            let elapsed = Date().timeIntervalSince(started)
            if elapsed >= seconds { break }
            remaining = Int((seconds - elapsed).rounded(.up))
            level = min(collector.level * 4, 1)
            try await Task.sleep(for: .milliseconds(100))
        }
        let (samples, rate) = collector.take()
        var clip = try AudioClip.resample(samples, from: rate ?? format.sampleRate)
        clip = Array(clip.prefix(Int(seconds) * AudioClip.sampleRate))
        guard !AudioClip.isSilent(clip) else { throw Failure.silent }
        return clip
    }

    /// The name of an output the microphone can't hear (headphones, AirPlay, a car), if one is in use.
    static func privateOutput(_ route: AVAudioSessionRouteDescription) -> String? {
        let ports: [AVAudioSession.Port: String] = [
            .headphones: "headphones", .bluetoothA2DP: "Bluetooth headphones", .bluetoothHFP: "Bluetooth headphones",
            .bluetoothLE: "Bluetooth headphones", .airPlay: "AirPlay", .carAudio: "your car", .usbAudio: "a USB device",
            .HDMI: "HDMI", .lineOut: "a cable",
        ]
        return route.outputs.lazy.compactMap { ports[$0.portType] }.first
    }

    /// Made outside the main actor: the tap runs on the audio thread.
    private nonisolated static func tap(_ collector: Collector) -> AVAudioNodeTapBlock {
        { buffer, _ in collector.append(buffer) }
    }
}

/// Samples from the audio thread, read by the recorder.
private final class Collector: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var rate: Double?
    private var lastLevel: Float = 0

    var level: Float { lock.withLock { lastLevel } }

    func append(_ buffer: AVAudioPCMBuffer) {
        guard let data = buffer.floatChannelData else { return }
        let count = Int(buffer.frameLength)
        let chunk = UnsafeBufferPointer(start: data[0], count: count)
        let level = AudioClip.rms(Array(chunk))
        lock.withLock {
            samples.append(contentsOf: chunk)
            rate = buffer.format.sampleRate
            lastLevel = level
        }
    }

    func take() -> ([Float], Double?) {
        lock.withLock { (samples, rate) }
    }
}
#endif
