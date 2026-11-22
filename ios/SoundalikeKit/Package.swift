// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SoundalikeKit",
    platforms: [.iOS(.v26), .macOS(.v15)],
    products: [
        .library(name: "SoundalikeKit", targets: ["SoundalikeKit"]),
        .library(name: "SoundalikeUI", targets: ["SoundalikeUI"]),
    ],
    targets: [
        .target(name: "SoundalikeKit", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "SoundalikeUI", dependencies: ["SoundalikeKit"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(
            name: "SoundalikeKitTests",
            dependencies: ["SoundalikeKit", "SoundalikeUI"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
