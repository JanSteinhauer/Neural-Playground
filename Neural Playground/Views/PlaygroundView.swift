import SwiftUI

struct PlaygroundView: View {
    @StateObject private var viewModel = TrainingViewModel()
    @State private var showDebug = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("Neural Playground")
                    .font(.title).bold()
                Spacer()
                Button("Reset") {
                    viewModel.reset()
                }
                
                Button(action: { showDebug = true }) {
                    Image(systemName: "list.bullet.rectangle")
                }
            }
            .padding()
            .sheet(isPresented: $showDebug) {
                if let timeline = viewModel.timeline {
                    NavigationView {
                        TimelineDebugView(timeline: timeline, selectedSnapshotIndex: $viewModel.scrubberValue)
                    }
                } else {
                    Text("No timeline")
                }
            }
            
            // Configuration Controls
            HStack {
                VStack(alignment: .leading) {
                    Text("Learning Rate: \(String(format: "%.2f", viewModel.lr))")
                    Slider(value: $viewModel.lr, in: 0.01...1.0)
                }
                
                Picker("Sample", selection: $viewModel.selectedSampleIndex) {
                    ForEach(0..<viewModel.samples.count, id: \.self) { idx in
                        let s = viewModel.samples[idx]
                        Text("[\(Int(s.0[0])), \(Int(s.0[1]))] → \(Int(s.1[0]))")
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: viewModel.selectedSampleIndex) { _ in
                    viewModel.refreshTimeline()
                }
            }
            .padding(.horizontal)
            
            // Visualization
            if let snapshot = viewModel.currentSnapshot {
                NetworkCanvasView(
                    snapshot: snapshot,
                    selectedNode: viewModel.selectedNode,
                    onSelectNode: { id in
                        viewModel.selectedNode = id
                    }
                )
                .background(Color(white: 0.95))
                .cornerRadius(12)
                .padding()
                
                // Timeline Controls
                VStack {
                    ZStack {
                        // Progress bar with phase regions could go here
                        Text(snapshot.phase.title)
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                    
                    Slider(value: $viewModel.scrubberValue, in: 0...viewModel.maxScrubberValue)
                        .accentColor(.blue)
                    
                    HStack {
                        Text("Snapshot: \(Int(viewModel.scrubberValue))")
                        Spacer()
                        if let loss = snapshot.loss as Double? {
                            Text("Loss: \(String(format: "%.4f", loss))")
                        }
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            } else {
                Text("No Timeline Generated")
            }
            
            // Loss History (Simple Chart)
            if !viewModel.lossHistory.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Loss History")
                        .font(.caption).bold()
                        .foregroundColor(.secondary)
                    
                    Path { path in
                        let w = 300.0
                        let h = 40.0
                        let maxLoss = (viewModel.lossHistory.max() ?? 1.0)
                        let count = viewModel.lossHistory.count
                        
                        // Draw line
                        if count > 1 {
                             for (i, loss) in viewModel.lossHistory.enumerated() {
                                 let x = Double(i) / Double(count - 1) * w
                                 // Scale y so 0 is bottom, maxLoss is top
                                 // Actually let's clamp max visual loss to 1.0 for stability or use actual max
                                 let normalizedLoss = loss / (maxLoss > 0.001 ? maxLoss : 1.0)
                                 let y = h * (1.0 - normalizedLoss)
                                 
                                 if i == 0 {
                                     path.move(to: CGPoint(x: x, y: y))
                                 } else {
                                     path.addLine(to: CGPoint(x: x, y: y))
                                 }
                             }
                        }
                    }
                    .stroke(Color.red, lineWidth: 2)
                    .frame(height: 40)
                    .background(Color.green.opacity(0.1)) // simple bg
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                }
                .padding(.horizontal)
            }
            
            Spacer()
            
            // Action
            Button(action: {
                withAnimation {
                    viewModel.trainOneStep()
                }
            }) {
                Text("Train One Step")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .padding()
        }
        .sheet(item: $viewModel.selectedNode) { nodeID in
            if let snap = viewModel.currentSnapshot, let val = snap.nodeValues[nodeID] {
                 NodeInspectorSheet(nodeID: nodeID, visual: val)
            } else {
                Text("No data for node")
            }
        }
    }
}

// Make NodeID conform to Identifiable for sheet item binding
extension NodeID: Identifiable {
    public var id: String { "\(layer)-\(index)" }
}
