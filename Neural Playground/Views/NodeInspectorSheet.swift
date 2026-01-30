import SwiftUI

struct NodeInspectorSheet: View {
    let nodeID: NodeID
    let visual: NodeVisualValues
    
    // We could pass in edges if we want to list incoming/outgoing weights.
    // For now, let's keep it simple as per prompt: "selected node values"
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Node Details")
                .font(.headline)
            
            HStack {
                Text("Layer \(nodeID.layer), Neuron \(nodeID.index)")
                    .font(.title3).bold()
                Spacer()
            }
            
            Divider()
            
            // Grid of values
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                ValueCard(title: "Pre-activation (z)", value: visual.z)
                ValueCard(title: "Activation (a)", value: visual.a)
                ValueCard(title: "Gradient (grad)", value: visual.grad)
            }
            
            Spacer()
        }
        .padding()
        .presentationDetents([.fraction(0.3), .medium])
    }
}

struct ValueCard: View {
    let title: String
    let value: Double?
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            if let v = value {
                Text(String(format: "%.4f", v))
                    .font(.system(.body, design: .monospaced))
                    .padding(4)
                    .background(Color.secondary.opacity(0.1))
                    .cornerRadius(4)
            } else {
                Text("-")
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
