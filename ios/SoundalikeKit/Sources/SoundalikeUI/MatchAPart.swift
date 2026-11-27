import SoundalikeKit
import SwiftUI
import UniformTypeIdentifiers

/// "Match a part" under the seed: shown only when the answer on screen has `query.parts`. The lists then follow
/// a part of the 30-second preview (`part=`), or, when the server also takes clips (`query.parts.excerpt`) and
/// the host allows it, a clip of the listener's own audio from the microphone or a file.
struct PartRow: View {
    let results: SoundalikeResults
    let parts: PartLimits
    @ObservedObject var player: PreviewPlayer
    let allowsClips: Bool
    let match: (PartChoice?) async throws -> Void
    @State private var showsPanel = false
    @State private var resetting = false
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button { showsPanel = true } label: {
                HStack(spacing: 6) {
                    Label("Match a part", systemImage: "waveform.path.ecg")
                        .font(.subheadline.weight(.semibold))
                    BetaBadge()
                }
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.capsule)
            if let chip {
                HStack(spacing: 8) {
                    Text(chip).font(.footnote.weight(.semibold)).lineLimit(2)
                    Spacer(minLength: 4)
                    if resetting {
                        ProgressView().controlSize(.small)
                    } else {
                        Button("Whole preview") { wholePreview() }
                            .font(.footnote.weight(.semibold))
                            .buttonStyle(.borderless)
                            .accessibilityHint("Match the whole preview again")
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(Color.accentColor.opacity(0.12), in: Capsule())
            }
            if let failure {
                Text(failure).font(.footnote).foregroundStyle(.red)
            }
        }
        .sheet(isPresented: $showsPanel) {
            PartPanel(results: results, parts: parts, player: player, allowsClips: allowsClips, match: match)
        }
    }

    /// "Matching 0:58–1:13 of the song", "… of the preview", or "Matching your clip (0:00–0:20 of your file)".
    var chip: String? { Self.chip(for: results) }

    nonisolated static func chip(for results: SoundalikeResults) -> String? {
        if let clip = results.clip {
            return "Matching your clip (\(timeLabel(clip.start))–\(timeLabel(clip.end)) of your \(clip.source == .file ? "file" : "recording"))"
        }
        guard let part = results.part else { return nil }
        return "Matching " + PartTimes(parts: results.parts).span(part)
    }

    private func wholePreview() {
        resetting = true
        failure = nil
        player.stop()
        Task {
            do { try await match(nil) } catch { failure = error.localizedDescription }
            resetting = false
        }
    }
}

/// Times on the preview's bar: song times when the server says where the preview starts, else preview times.
/// The song's length isn't known here, so the usual 0:48 is never assumed.
struct PartTimes {
    let offset: Double?

    init(parts: PartLimits?) {
        offset = parts?.songOffset
    }

    func label(_ seconds: Double) -> String { timeLabel(seconds + (offset ?? 0)) }

    func span(_ part: PartSpan) -> String {
        "\(label(part.start))–\(label(part.end)) of the \(offset == nil ? "preview" : "song")"
    }
}

struct BetaBadge: View {
    var body: some View {
        Text("BETA").font(.caption2.weight(.heavy))
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(.orange.opacity(0.2), in: Capsule())
            .foregroundStyle(.orange)
            .accessibilityLabel("beta")
    }
}

struct PartPanel: View {
    enum Source: Hashable { case preview, listen, file }

    let results: SoundalikeResults
    let parts: PartLimits
    @ObservedObject var player: PreviewPlayer
    let allowsClips: Bool
    let match: (PartChoice?) async throws -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var source: Source = .preview
    @State private var span: PartSpan
    @State private var working = false
    @State private var failure: String?

    init(results: SoundalikeResults, parts: PartLimits, player: PreviewPlayer, allowsClips: Bool, match: @escaping (PartChoice?) async throws -> Void) {
        self.results = results
        self.parts = parts
        self.player = player
        self.allowsClips = allowsClips
        self.match = match
        let lengths = Self.previewLengths(parts)
        let length = lengths.contains(10) ? 10 : lengths.first ?? parts.minSeconds
        _span = State(initialValue: results.part ?? PartSpan(start: min(10, parts.previewSeconds - length), end: min(10, parts.previewSeconds - length) + length))
    }

    nonisolated static func previewLengths(_ parts: PartLimits) -> [Double] {
        let lengths = [10.0, 15, 20].filter { $0 >= parts.minSeconds && $0 <= min(parts.maxSeconds, parts.previewSeconds) }
        return lengths.isEmpty ? [parts.minSeconds] : lengths
    }

    /// The clip lengths the server takes, or none when it takes no clips.
    private var clipLengths: [Double] { Self.clipLengths(parts, allowsClips: allowsClips) }

    /// Clips are offered only when the answer has `parts.excerpt` and the host can record or read files.
    nonisolated static func clipLengths(_ parts: PartLimits, allowsClips: Bool) -> [Double] {
        guard allowsClips, let excerpt = parts.excerpt else { return [] }
        return AudioClip.lengths(for: excerpt)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if !clipLengths.isEmpty {
                        Picker("Source", selection: $source) {
                            Text("Preview").tag(Source.preview)
                            Text("Listen").tag(Source.listen)
                            Text("File").tag(Source.file)
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: source) { player.stop(); failure = nil }
                    }
                    switch source {
                    case .preview: previewSource
                    #if os(iOS)
                    case .listen: ListenSource(lengths: clipLengths, player: player, send: send)
                    case .file: FileSource(lengths: clipLengths, player: player, send: send)
                    #else
                    case .listen, .file: EmptyView()
                    #endif
                    }
                    if working {
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Finding soundalikes for this part…").foregroundStyle(.secondary)
                        }
                    }
                    if let failure {
                        Text(failure).font(.subheadline).foregroundStyle(.red)
                    }
                }
                .padding()
            }
            .disabled(working)
            .navigationTitle("Match a part")
            .navigationBarTitleDisplayModeInline()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 6) { Text("Match a part").font(.headline); BetaBadge() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .onDisappear { player.stop() }
    }

    private var previewSource: some View {
        let times = PartTimes(parts: parts)
        let playing = player.isActive(Self.playerID)
        return VStack(alignment: .leading, spacing: 14) {
            Text("The 30-second preview").font(.headline)
            Text("Pick the part of the preview the soundalikes should follow. Drag the part or its edges, or tap the bar.")
                .font(.subheadline).foregroundStyle(.secondary)
            SpanBar(
                total: parts.previewSeconds, span: $span, step: 5, lengths: Self.previewLengths(parts),
                label: times.label, playhead: playing ? player.position : nil
            )
            Text(times.span(span).prefix(1).uppercased() + times.span(span).dropFirst())
                .font(.subheadline.monospacedDigit().weight(.semibold))
            HStack {
                Button(playing ? "Stop" : "Play this part", systemImage: playing ? "stop.fill" : "play.fill") {
                    if playing {
                        player.stop()
                    } else {
                        Task { await player.play(id: Self.playerID, url: nil, deezerID: results.seed.deezerID, span: span.start...span.end) }
                    }
                }
                .buttonStyle(.bordered)
                .disabled(results.seed.deezerID == nil)
                Spacer()
                Button("Apply") { send(.preview(span)) }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    static let playerID = "match-a-part"

    private func send(_ choice: PartChoice) {
        player.stop()
        working = true
        failure = nil
        Task {
            do {
                try await match(choice)
                working = false
                dismiss()
            } catch {
                working = false
                failure = error.localizedDescription
            }
        }
    }
}

/// A bar of the preview, recording or file with the part on it: drag the part, drag an edge, or tap elsewhere
/// to move it there. The part stays on a grid of `step` seconds and is one of `lengths` long, or in between
/// when an edge is dragged.
struct SpanBar: View {
    let total: Double
    @Binding var span: PartSpan
    let step: Double
    let lengths: [Double]
    let label: (Double) -> String
    let playhead: Double?
    @State private var mode: Mode?
    @State private var origin = PartSpan(start: 0, end: 0)

    enum Mode { case move, start, end, place }

    private var minLength: Double { lengths.min() ?? 10 }
    private var maxLength: Double { min(lengths.max() ?? 20, total) }

    var body: some View {
        VStack(spacing: 6) {
            GeometryReader { geo in
                let width = geo.size.width
                let x = { (seconds: Double) in CGFloat(seconds / max(total, 1)) * width }
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.18))
                    RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.3))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.accentColor, lineWidth: 2))
                        .frame(width: max(x(span.end) - x(span.start), 12))
                        .offset(x: x(span.start))
                    Capsule().fill(Color.accentColor).frame(width: 5, height: 26).offset(x: x(span.start) + 4)
                    Capsule().fill(Color.accentColor).frame(width: 5, height: 26).offset(x: x(span.end) - 9)
                    if let playhead {
                        Rectangle().fill(Color.primary).frame(width: 2).offset(x: x(min(max(playhead, 0), total)) - 1)
                    }
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in changed(value, width: width) }
                        .onEnded { value in
                            if mode == .place { place(at: Double(value.location.x / width) * total) }
                            mode = nil
                        }
                )
            }
            .frame(height: 48)
            .accessibilityElement()
            .accessibilityLabel("Part to match")
            .accessibilityValue("\(label(span.start)) to \(label(span.end))")
            .accessibilityAdjustableAction { direction in
                move(by: direction == .increment ? step : -step)
            }
            HStack {
                Text(label(0))
                Spacer()
                Text(label(total))
            }
            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button("−5 s") { move(by: -5) }.accessibilityLabel("Earlier by 5 seconds")
                Button("+5 s") { move(by: 5) }.accessibilityLabel("Later by 5 seconds")
                Spacer()
                ForEach(lengths, id: \.self) { length in
                    Button("\(Int(length)) s") { setLength(length) }
                        .tint(abs(span.length - length) < 0.01 ? .accentColor : .secondary)
                        .accessibilityLabel("\(Int(length)) seconds long")
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    private func snap(_ value: Double) -> Double { (value / step).rounded() * step }

    private func changed(_ value: DragGesture.Value, width: CGFloat) {
        let seconds = { (points: CGFloat) in Double(points / width) * total }
        if mode == nil {
            origin = span
            let at = value.startLocation.x
            let startX = CGFloat(span.start / total) * width, endX = CGFloat(span.end / total) * width
            if abs(at - startX) < 22 { mode = .start }
            else if abs(at - endX) < 22 { mode = .end }
            else if at > startX, at < endX { mode = .move }
            else { mode = .place }
        }
        let delta = seconds(value.translation.width)
        switch mode {
        case .move:
            let start = min(max(snap(origin.start + delta), 0), total - origin.length)
            span = PartSpan(start: start, end: start + origin.length)
        case .start:
            let start = min(max(snap(origin.start + delta), max(0, origin.end - maxLength)), origin.end - minLength)
            span = PartSpan(start: start, end: origin.end)
        case .end:
            let end = max(min(snap(origin.end + delta), min(total, origin.start + maxLength)), origin.start + minLength)
            span = PartSpan(start: origin.start, end: end)
        case .place:
            if abs(value.translation.width) > 8 { mode = .move; origin = span }
        case nil:
            break
        }
    }

    private func place(at seconds: Double) {
        let length = span.length
        let start = min(max(snap(seconds - length / 2), 0), total - length)
        span = PartSpan(start: start, end: start + length)
    }

    private func move(by seconds: Double) {
        let length = span.length
        let start = min(max(span.start + seconds, 0), total - length)
        span = PartSpan(start: start, end: start + length)
    }

    private func setLength(_ length: Double) {
        let start = min(span.start, max(total - length, 0))
        span = PartSpan(start: start, end: min(start + length, total))
    }
}

extension View {
    func navigationBarTitleDisplayModeInline() -> some View {
        #if os(iOS)
        navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }
}
