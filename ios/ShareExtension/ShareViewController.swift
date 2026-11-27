import SoundalikeKit
import SoundalikeUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// "Find soundalikes" in the share sheet. Shows results over the music app;
/// never bounces into the main app.
final class ShareViewController: UIViewController {
    private let model = MainActor.assumeIsolated { ResultsModel() }

    override func viewDidLoad() {
        super.viewDidLoad()
        let root = ShareRoot(model: model, actions: actions, close: { [weak self] in self?.close() })
        let host = UIHostingController(rootView: root)
        addChild(host)
        host.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
        host.didMove(toParent: self)

        Task { @MainActor in
            let types = (extensionContext?.inputItems as? [NSExtensionItem] ?? []).flatMap { $0.attachments ?? [] }.map(\.registeredTypeIdentifiers)
            let text = await sharedText()
            EventLog.record("share-received", ["types": types.description, "payload": text ?? ""])
            if let text {
                model.load(shared: text)
            } else {
                model.state = .failed(SoundalikeError.unsupportedLink.localizedDescription)
            }
        }
    }

    /// Spotify shares a URL (sometimes text with "Title by Artist"); Music shares a URL.
    private func sharedText() async -> String? {
        let providers = (extensionContext?.inputItems as? [NSExtensionItem] ?? []).flatMap { $0.attachments ?? [] }
        var found: [String] = []
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.url.identifier),
               let url = try? await provider.loadItem(forTypeIdentifier: UTType.url.identifier) as? URL {
                found.append(url.absoluteString)
            } else if provider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier),
                      let text = try? await provider.loadItem(forTypeIdentifier: UTType.plainText.identifier) as? String {
                found.append(text)
            }
        }
        let joined = found.joined(separator: "\n")
        return joined.isEmpty ? nil : joined
    }

    /// iOS doesn't let share extensions open other apps (`extensionContext.open`
    /// returned false from Spotify on iOS 26.6, 2026-11-20), so a result is copied
    /// as "Title Artist", ready to paste into the music app's search.
    private var actions: ResultActions {
        MainActor.assumeIsolated {
            ResultActions(
                open: { [weak self] match, target in
                    UIPasteboard.general.string = "\(match.title) \(match.artist)"
                    EventLog.record("copy", ["from": "share", "target": target.rawValue])
                    self?.model.notice = "Copied \"\(match.title)\". Tap Done, then paste it into \(target.displayName) search."
                },
                openSymbol: "doc.on.doc.fill",
                openLabel: { "Copy for \($0.displayName) search" }
            )
        }
    }

    private func close() {
        extensionContext?.completeRequest(returningItems: nil)
    }
}

private struct ShareRoot: View {
    @ObservedObject var model: ResultsModel
    let actions: ResultActions
    let close: () -> Void

    var body: some View {
        NavigationStack {
            ResultsView(state: model.state, target: $model.target, actions: actions, matchPart: { try await model.match($0) }, retry: { model.retry() })
                .navigationTitle("Soundalikes")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done", action: close) } }
                .safeAreaInset(edge: .bottom) {
                    if let notice = model.notice {
                        Text(notice).font(.footnote).multilineTextAlignment(.center)
                            .padding(12).frame(maxWidth: .infinity).background(.thinMaterial)
                    }
                }
        }
    }
}
