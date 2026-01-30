import SwiftUI

struct TimelineDebugView: View {
    let timeline: StepTimeline
    @Binding var selectedSnapshotIndex: Double
    
    var body: some View {
        List {
            ForEach(Array(timeline.snapshots.enumerated()), id: \.element.id) { index, snap in
                TimelineSnapshotRow(index: index, snap: snap, selectedIndex: $selectedSnapshotIndex)
            }
        }
        .navigationTitle("Timeline Debug")
    }
}

struct TimelineSnapshotRow: View {
    let index: Int
    let snap: StepSnapshot
    @Binding var selectedIndex: Double
    
    var body: some View {
        Button(action: {
            selectedIndex = Double(index)
        }) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Frame \(index): \(snap.phase.title)")
                        .font(.headline)
                    Text("t: \(String(format: "%.2f", snap.t))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                debugInfo
                    .font(.caption.monospaced())
            }
            .padding(.vertical, 4)
            .background(Int(selectedIndex) == index ? Color.blue.opacity(0.1) : Color.clear)
        }
    }
    
    @ViewBuilder
    var debugInfo: some View {
        VStack(alignment: .trailing) {
            switch snap.phase {
            case .forward:
                Text("yHat: \(String(format: "%.3f", snap.cache.yHat.first ?? 0))")
            case .loss:
                Text("Loss: \(String(format: "%.3f", snap.loss))")
            case .backward:
                if let grads = snap.grads {
                    Text("dW1[0]: \(String(format: "%.3f", grads.dW1[0][0]))")
                }
            case .update:
                // Show a weight changing
                if let w = weightValue {
                    Text("W1_00: \(String(format: "%.3f", w))")
                }
            }
        }
    }
    
    var weightValue: Double? {
        let eid = EdgeID(from: NodeID(layer: 0, index: 0), to: NodeID(layer: 1, index: 0))
        return snap.edgeValues[eid]?.w
    }
}
