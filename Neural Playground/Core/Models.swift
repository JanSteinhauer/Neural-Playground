import Foundation

// MARK: - Core Identifiers

struct NodeID: Hashable {
    let layer: Int
    let index: Int
}

struct EdgeID: Hashable {
    let from: NodeID
    let to: NodeID
}

enum Phase: Int, CaseIterable {
    case forward = 0
    case loss
    case backward
    case update
    
    var title: String {
        switch self {
        case .forward: return "Forward Pass"
        case .loss: return "Loss Computation"
        case .backward: return "Backward Pass"
        case .update: return "Weight Update"
        }
    }
}

// MARK: - Network Parameters

struct NetworkParams {
    // Layer 1: Hidden (2 neurons, 2 inputs)
    var W1: [[Double]] // 2x2
    var b1: [Double]   // 2

    // Layer 2: Output (1 neuron, 2 inputs from hidden)
    var W2: [[Double]] // 1x2
    var b2: [Double]   // 1
    
    static func random() -> NetworkParams {
        func rand() -> Double { Double.random(in: -1...1) }
        func resultRow(count: Int) -> [Double] { (0..<count).map { _ in rand() } }
        
        return NetworkParams(
            W1: [resultRow(count: 2), resultRow(count: 2)],
            b1: resultRow(count: 2),
            W2: [resultRow(count: 2)],
            b2: [rand()] // output bias usually scalar but modeled as array for consistency
        )
    }
}

// MARK: - Runtime Values

struct ForwardCache {
    var x: [Double]     // Input
    var z1: [Double]    // Hidden pre-activation
    var a1: [Double]    // Hidden activation
    var z2: [Double]    // Output pre-activation
    var yHat: [Double]  // Output activation (prediction)
}

struct Gradients {
    var dW1: [[Double]]
    var db1: [Double]
    var dW2: [[Double]]
    var db2: [Double]

    // Local gradients for visualization
    var dZ1: [Double]
    var dA1: [Double]
    var dZ2: [Double]
}

// MARK: - Timeline / Visualization Snapshots

struct StepSnapshot: Identifiable {
    let id: UUID
    let phase: Phase
    let t: Double // 0...1 within phase for interpolation

    let params: NetworkParams
    let cache: ForwardCache
    let loss: Double

    let grads: Gradients?

    // Derived dictionaries for easy UI lookup
    let nodeValues: [NodeID: NodeVisualValues]
    let edgeValues: [EdgeID: EdgeVisualValues]
}

struct NodeVisualValues {
    var z: Double?
    var a: Double?
    var grad: Double?  // e.g. dZ or dA
}

struct EdgeVisualValues {
    var w: Double
    var grad: Double?  // dW
    var delta: Double? // -eta * dW (weight update amount)
}
