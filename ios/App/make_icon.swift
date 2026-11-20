// Renders the Soundalike app icon in the iOS system style: one bold SF Symbol glyph
// on a smooth vertical gradient. A music note and its softer echo, side by side:
// two songs that sound alike.
//
//   swift make_icon.swift   ->   Assets.xcassets/AppIcon.appiconset/icon.png (1024 px)
import AppKit

let size: CGFloat = 1024
let output = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent("Assets.xcassets/AppIcon.appiconset/icon.png")

let image = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
    // Background: violet at the top to deep indigo at the bottom, like Apple's own icons.
    NSGradient(colors: [
        NSColor(srgbRed: 0.62, green: 0.36, blue: 1.00, alpha: 1),
        NSColor(srgbRed: 0.33, green: 0.16, blue: 0.86, alpha: 1),
    ])!.draw(in: rect, angle: -90)

    func note(alpha: CGFloat, offset: CGPoint, pointSize: CGFloat) {
        let config = NSImage.SymbolConfiguration(pointSize: pointSize, weight: .semibold)
            .applying(NSImage.SymbolConfiguration(paletteColors: [NSColor.white.withAlphaComponent(alpha)]))
        guard let symbol = NSImage(systemSymbolName: "music.note", accessibilityDescription: nil)?
            .withSymbolConfiguration(config) else { return }
        let glyph = symbol.size
        let origin = CGPoint(x: (size - glyph.width) / 2 + offset.x, y: (size - glyph.height) / 2 + offset.y)
        symbol.draw(in: CGRect(origin: origin, size: glyph))
    }

    let shadow = NSShadow()
    shadow.shadowColor = NSColor(srgbRed: 0.12, green: 0.02, blue: 0.35, alpha: 0.35)
    shadow.shadowOffset = NSSize(width: 0, height: -14)
    shadow.shadowBlurRadius = 36

    // The echo, behind and to the left.
    note(alpha: 0.38, offset: CGPoint(x: -118, y: 34), pointSize: 470)
    // The note itself, in front.
    shadow.set()
    note(alpha: 1, offset: CGPoint(x: 92, y: -22), pointSize: 470)
    return true
}

let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size), bitsPerSample: 8,
    samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
image.draw(in: NSRect(x: 0, y: 0, width: size, height: size))
NSGraphicsContext.restoreGraphicsState()
// App icons must be opaque: re-encode without alpha.
let opaque = NSBitmapImageRep(cgImage: rep.cgImage!.copy(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)!)
let cg = opaque.cgImage!
let context = CGContext(data: nil, width: cg.width, height: cg.height, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
context.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
try NSBitmapImageRep(cgImage: context.makeImage()!).representation(using: .png, properties: [:])!.write(to: output)
print("wrote", output.path)
