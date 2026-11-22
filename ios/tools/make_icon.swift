// Renders the Soundalike app icon: the original nested diamonds on the green-to-violet
// gradient, refined with softer corners, a richer gradient and a little depth.
//
//   swift tools/make_icon.swift   ->   App/Assets.xcassets/AppIcon.appiconset/icon.png (1024 px)
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let size = 1024
let s = CGFloat(size)
let output = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("App/Assets.xcassets/AppIcon.appiconset/icon.png")

let space = CGColorSpace(name: CGColorSpace.displayP3)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                    space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: space, components: [r, g, b, a])!
}

// Diagonal gradient, top left to bottom right: green, teal, violet.
let background = CGGradient(colorsSpace: space, colors: [
    color(0.11, 0.80, 0.42), color(0.20, 0.62, 0.66), color(0.53, 0.40, 0.98),
] as CFArray, locations: [0, 0.48, 1])!
ctx.drawLinearGradient(background, start: CGPoint(x: 0, y: s), end: CGPoint(x: s, y: 0), options: [])

// Soft light from the top and a gentle darkening at the bottom for depth.
let light = CGGradient(colorsSpace: space, colors: [color(1, 1, 1, 0.16), color(1, 1, 1, 0)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(light, startCenter: CGPoint(x: s * 0.3, y: s * 0.95), startRadius: 0,
                       endCenter: CGPoint(x: s * 0.3, y: s * 0.95), endRadius: s * 0.9, options: [])
let shade = CGGradient(colorsSpace: space, colors: [color(0, 0, 0, 0), color(0.05, 0.02, 0.2, 0.18)] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(shade, start: CGPoint(x: s / 2, y: s * 0.55), end: CGPoint(x: s / 2, y: 0), options: [])

/// A square with rounded corners, turned 45° and centred.
func diamond(halfDiagonal: CGFloat, cornerRadius: CGFloat) -> CGPath {
    let side = halfDiagonal * 2 / sqrt(2)
    let square = CGRect(x: -side / 2, y: -side / 2, width: side, height: side)
    var transform = CGAffineTransform(translationX: s / 2, y: s / 2).rotated(by: .pi / 4)
    return CGPath(roundedRect: square, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: &transform)
}

let ink = color(0.04, 0.05, 0.08)
ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 28, color: color(0.02, 0.05, 0.12, 0.35))

// Outer ring.
ctx.addPath(diamond(halfDiagonal: 262, cornerRadius: 34))
ctx.setStrokeColor(ink)
ctx.setLineWidth(46)
ctx.setLineJoin(.round)
ctx.strokePath()

// Inner diamond.
ctx.addPath(diamond(halfDiagonal: 112, cornerRadius: 22))
ctx.setFillColor(ink)
ctx.fillPath()

let destination = CGImageDestinationCreateWithURL(output as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(destination, ctx.makeImage()!, nil)
CGImageDestinationFinalize(destination)
print("wrote", output.path)
