import AVFoundation
import SwiftUI

/// Plays a bundled 270p clip through the upscaler and shows the sustained frame time.
struct ContentView: View {
    @State private var status = "Tap Run to upscale demo.mp4"
    @State private var running = false

    var body: some View {
        VStack(spacing: 16) {
            Text("vsr-lite, x4 on device").font(.headline)
            Text(status).font(.system(.body, design: .monospaced))
            Button(running ? "Running" : "Run") { Task { await run() } }
                .disabled(running)
        }
        .padding()
    }

    private func run() async {
        running = true
        defer { running = false }
        do {
            guard let url = Bundle.main.url(forResource: "demo", withExtension: "mp4") else {
                status = "demo.mp4 is not in the app bundle"
                return
            }
            let upscaler = try Upscaler()
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { return }
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: upscaler.width,
                kCVPixelBufferHeightKey as String: upscaler.height,
            ])
            reader.add(output)
            reader.startReading()
            var times: [Double] = []
            while let sample = output.copyNextSampleBuffer(), let frame = CMSampleBufferGetImageBuffer(sample) {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try upscaler.upscale(frame)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            let tail = times.suffix(300)
            let mean = tail.isEmpty ? 0 : tail.reduce(0, +) / Double(tail.count)
            status = String(format: "%d frames, sustained %.1f ms/frame", times.count, mean)
        } catch {
            status = error.localizedDescription
        }
    }
}
