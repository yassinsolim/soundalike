import Foundation

/// Local, on-device record of what each route did and how long it took
/// (share payloads, open attempts, recognition time). Nothing leaves the phone;
/// a developer reads it with `devicectl device copy from --domain-type appGroupDataContainer`
/// (which only sees `Library/`).
public enum EventLog {
    private static let queue = DispatchQueue(label: "app.yassin.soundalike.eventlog")
    private static let limit = 512 * 1024

    static var fileURL: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: SharedSettings.appGroup)?
            .appendingPathComponent("Library/events.jsonl")
    }

    public static func record(_ event: String, _ fields: [String: Any] = [:]) {
        var entry = fields
        entry["event"] = event
        entry["at"] = ISO8601DateFormatter().string(from: Date())
        entry["process"] = Bundle.main.bundleIdentifier ?? "?"
        guard let data = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
        queue.async {
            guard let url = fileURL else { return }
            if let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int, size > limit {
                try? FileManager.default.removeItem(at: url)
            }
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            guard let handle = try? FileHandle(forWritingTo: url) else { return }
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data + Data("\n".utf8))
        }
    }

    /// Milliseconds since `start`, for timing fields.
    public static func ms(since start: Date) -> Int {
        Int(Date().timeIntervalSince(start) * 1000)
    }
}
