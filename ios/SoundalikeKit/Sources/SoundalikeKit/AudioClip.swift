import AVFoundation
import Foundation

/// A clip of the listener's own audio, made ready for `POST /api/anysong/excerpt`: one channel at 22,050 Hz,
/// sent as 16-bit PCM WAV. Nothing here touches the disk.
public enum AudioClip {
    public static let sampleRate = 22_050

    /// The clip lengths offered (10, 15 and 20 s) that the server takes, also within its size limit.
    public static func lengths(for limits: ExcerptLimits, offered: [Double] = [10, 15, 20]) -> [Double] {
        offered.filter { length in
            length >= limits.minSeconds && length <= limits.maxSeconds
                && WAV.size(frames: Int(length) * sampleRate) <= limits.maxBytes
        }
    }

    /// Root mean square of the samples, 0 for none.
    public static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for sample in samples { sum += sample * sample }
        return (sum / Float(samples.count)).squareRoot()
    }

    /// Too quiet to be music: what the microphone hears when nothing plays out loud.
    public static func isSilent(_ samples: [Float]) -> Bool {
        rms(samples) < 0.002
    }

    /// Brought to full scale (the server does this too; doing it first keeps a quiet recording's detail in 16 bits).
    public static func normalized(_ samples: [Float]) -> [Float] {
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        guard peak > 0 else { return samples }
        let gain = 0.98 / peak
        return samples.map { $0 * gain }
    }

    /// The first channels of `buffer` mixed to one and converted to `sampleRate`.
    public static func mono(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        let input: AVAudioPCMBuffer
        if buffer.format.commonFormat == .pcmFormatFloat32, buffer.format.channelCount > 1 {
            input = mixDown(buffer)
        } else {
            input = buffer
        }
        if input.format == target { return samples(input) }
        guard let converter = AVAudioConverter(from: input.format, to: target) else { throw SoundalikeError.clip("Soundalike can't read that kind of audio.") }
        let capacity = AVAudioFrameCount(Double(input.frameLength) * Double(sampleRate) / input.format.sampleRate) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { throw SoundalikeError.clip("That part is too long.") }
        var fed = false
        var failure: NSError?
        let status = converter.convert(to: output, error: &failure) { _, state in
            if fed {
                state.pointee = .endOfStream
                return nil
            }
            fed = true
            state.pointee = .haveData
            return input
        }
        if status == .error { throw SoundalikeError.clip("Soundalike couldn't read that audio.") }
        return samples(output)
    }

    /// Samples at `rate`, mono, converted to `sampleRate`.
    public static func resample(_ samples: [Float], from rate: Double) throws -> [Float] {
        guard !samples.isEmpty else { return [] }
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return [] }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        return try mono(buffer)
    }

    /// Seconds `start` to `end` of an audio file, decoded on the phone.
    public static func read(_ url: URL, from start: Double, to end: Double) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw SoundalikeError.clip("Soundalike can't read that file. Pick an audio file such as MP3, M4A or WAV.") }
        let rate = file.processingFormat.sampleRate
        let first = AVAudioFramePosition(max(0, start) * rate)
        let count = AVAudioFrameCount(max(0, min(Double(file.length - first), (end - start) * rate)))
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count) else {
            throw SoundalikeError.clip("That part of the file is empty.")
        }
        file.framePosition = first
        do { try file.read(into: buffer, frameCount: count) } catch { throw SoundalikeError.clip("Soundalike couldn't read that file.") }
        return try mono(buffer)
    }

    /// The length of an audio file in seconds.
    public static func duration(of url: URL) throws -> Double {
        do {
            let file = try AVAudioFile(forReading: url)
            return Double(file.length) / file.processingFormat.sampleRate
        } catch {
            throw SoundalikeError.clip("Soundalike can't read that file. Pick an audio file such as MP3, M4A or WAV.")
        }
    }

    private static func mixDown(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: buffer.format.sampleRate, channels: 1, interleaved: false)!
        let mono = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buffer.frameLength)!
        mono.frameLength = buffer.frameLength
        let channels = Int(buffer.format.channelCount), frames = Int(buffer.frameLength)
        let out = mono.floatChannelData![0]
        let stride = buffer.stride
        let interleaved = buffer.format.isInterleaved
        for frame in 0..<frames {
            var sum: Float = 0
            for channel in 0..<channels {
                sum += interleaved
                    ? buffer.floatChannelData![0][frame * stride + channel]
                    : buffer.floatChannelData![channel][frame]
            }
            out[frame] = sum / Float(channels)
        }
        return mono
    }

    private static func samples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let data = buffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(buffer.frameLength)))
    }
}

/// A RIFF WAVE file of 16-bit PCM, one channel.
public enum WAV {
    public static let headerSize = 44

    /// Bytes of a file holding `frames` samples.
    public static func size(frames: Int) -> Int {
        headerSize + frames * 2
    }

    public static func encode(_ samples: [Float], sampleRate: Int = AudioClip.sampleRate) -> Data {
        let dataBytes = samples.count * 2
        var data = Data(capacity: headerSize + dataBytes)
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + dataBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        append(UInt32(16)) // fmt chunk size
        append(UInt16(1)) // PCM
        append(UInt16(1)) // channels
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2)) // bytes per second
        append(UInt16(2)) // block align
        append(UInt16(16)) // bits per sample
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(dataBytes))
        let pcm = samples.map { Int16((max(-1, min(1, $0)) * Float(Int16.max)).rounded()).littleEndian }
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }
}
