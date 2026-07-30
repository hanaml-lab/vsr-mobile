import CoreML
import CoreVideo
import Foundation

/// Runs VSRLite.mlpackage (python -m export.coreml) frame by frame, carrying the hidden state.
/// Call `reset()` at a scene cut; the model's own gate handles fast motion.
final class Upscaler {
    static let stateChannels = 16

    private let model: MLModel
    private var state: MLMultiArray
    let width: Int
    let height: Int

    init(width: Int = 480, height: Int = 270) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "VSRLite", withExtension: "mlmodelc") else {
            throw NSError(domain: "VSRDemo", code: 1, userInfo: [NSLocalizedDescriptionKey: "VSRLite.mlpackage is not in the app bundle"])
        }
        model = try MLModel(contentsOf: url, configuration: config)
        self.width = width
        self.height = height
        state = try MLMultiArray(shape: [1, NSNumber(value: Upscaler.stateChannels), NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        reset()
    }

    func reset() {
        let n = state.count
        state.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress!.update(repeating: 0, count: n)
        }
    }

    /// `frame` is a 32BGRA pixel buffer of width x height; returns the x4 frame as a planar RGB array.
    func upscale(_ frame: CVPixelBuffer) throws -> MLMultiArray {
        let input = try planarRGB(frame)
        let features = try MLDictionaryFeatureProvider(dictionary: ["frame": input, "state": state])
        let out = try model.prediction(from: features)
        guard let up = out.featureValue(for: "upscaled")?.multiArrayValue,
              let next = out.featureValue(for: "state_out")?.multiArrayValue else {
            throw NSError(domain: "VSRDemo", code: 2, userInfo: [NSLocalizedDescriptionKey: "unexpected model outputs"])
        }
        state = next
        return up
    }

    private func planarRGB(_ buffer: CVPixelBuffer) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard CVPixelBufferGetWidth(buffer) == width, CVPixelBufferGetHeight(buffer) == height,
              let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NSError(domain: "VSRDemo", code: 3, userInfo: [NSLocalizedDescriptionKey: "frame must be \(width)x\(height) BGRA"])
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let plane = width * height
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            for y in 0..<height {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
                for x in 0..<width {
                    let p = row.advanced(by: x * 4)
                    let i = y * width + x
                    dst[i] = Float(p[2]) / 255
                    dst[plane + i] = Float(p[1]) / 255
                    dst[2 * plane + i] = Float(p[0]) / 255
                }
            }
        }
        return array
    }
}
