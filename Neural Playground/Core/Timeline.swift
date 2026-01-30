import Foundation

struct StepTimeline {
    // A sequence of snapshots representing one training step
    let snapshots: [StepSnapshot]
}

struct TimelineGenerator {
    
    /// Generates a full timeline of snapshots for one training step (Forward -> Loss -> Backward -> Update)
    static func makeTimeline(
        params: NetworkParams,
        x: [Double],
        y: [Double],
        lr: Double
    ) -> (StepTimeline, NetworkParams) {
        
        // 1. Compute Step Data (Single Source of Truth)
        let cache = BackpropEngine.forward(params: params, x: x)
        let lossValue = BackpropEngine.loss(yHat: cache.yHat, y: y)
        let grads = BackpropEngine.backward(params: params, cache: cache, y: y)
        let newParams = BackpropEngine.apply(params: params, grads: grads, lr: lr)
        
        var snapshots = [StepSnapshot]()
        
        // --- Phase 1: Forward Pass (Visualizing signal flow) ---
        // We'll generate N frames to show input -> hidden -> output
        let forwardFrames = 10
        for i in 0..<forwardFrames {
            let t = Double(i) / Double(forwardFrames - 1)
            
            // Progressive disclosure of values
            var nodeValues = [NodeID: NodeVisualValues]()
            
            // Inputs always visible
            for (idx, val) in x.enumerated() {
                nodeValues[NodeID(layer: 0, index: idx)] = NodeVisualValues(a: val)
            }
            
            // Hidden layer reveals around t=0.3
            if t > 0.3 {
                for (idx, val) in cache.a1.enumerated() {
                    nodeValues[NodeID(layer: 1, index: idx)] = NodeVisualValues(z: cache.z1[idx], a: val)
                }
            }
            
            // Output layer reveals around t=0.7
            if t > 0.7 {
                for (idx, val) in cache.yHat.enumerated() {
                    nodeValues[NodeID(layer: 2, index: idx)] = NodeVisualValues(z: cache.z2[idx], a: val)
                }
            }
            
            // Edges: static params during forward
            var edgeValues = [EdgeID: EdgeVisualValues]()
            // Fill edges (helper function later?)
            // For now, simple loop
            // Input->Hidden
            for r in 0..<params.W1.count {
                for c in 0..<params.W1[r].count {
                    let eid = EdgeID(from: NodeID(layer: 0, index: c), to: NodeID(layer: 1, index: r))
                    edgeValues[eid] = EdgeVisualValues(w: params.W1[r][c])
                }
            }
            // Hidden->Output
            for r in 0..<params.W2.count {
                for c in 0..<params.W2[r].count {
                    let eid = EdgeID(from: NodeID(layer: 1, index: c), to: NodeID(layer: 2, index: r))
                    edgeValues[eid] = EdgeVisualValues(w: params.W2[r][c])
                }
            }

            snapshots.append(StepSnapshot(
                id: UUID(),
                phase: .forward,
                t: t,
                params: params,
                cache: cache,
                loss: (t > 0.9) ? lossValue : 0, // reveal loss at end
                grads: nil,
                nodeValues: nodeValues,
                edgeValues: edgeValues
            ))
        }
        
        // --- Phase 2: Loss (Pause to read loss) ---
        let lossFrames = 5
        // Re-use last forward state but mark phase as loss
        if let lastForward = snapshots.last {
            for i in 0..<lossFrames {
                let t = Double(i) / Double(lossFrames - 1)
                snapshots.append(StepSnapshot(
                    id: UUID(),
                    phase: .loss,
                    t: t,
                    params: params,
                    cache: cache,
                    loss: lossValue,
                    grads: nil,
                    nodeValues: lastForward.nodeValues,
                    edgeValues: lastForward.edgeValues
                ))
            }
        }
        
        // --- Phase 3: Backward Pass (Visualizing flows back) ---
        // Reveal gradients: Output -> Hidden -> Input weights
        let backwardFrames = 10
        for i in 0..<backwardFrames {
            let t = Double(i) / Double(backwardFrames - 1)
            
            var nodeValues = [NodeID: NodeVisualValues]()
            var edgeValues = [EdgeID: EdgeVisualValues]()
            
            // Re-populate basic values (activations stay visible)
            // Inputs
            for (idx, val) in x.enumerated() {
                nodeValues[NodeID(layer: 0, index: idx)] = NodeVisualValues(a: val)
            }
            // Hidden
            for (idx, val) in cache.a1.enumerated() {
                // Show dZ1 or dA1 if t big enough
                let grad = (t > 0.5) ? grads.dZ1[idx] : nil
                nodeValues[NodeID(layer: 1, index: idx)] = NodeVisualValues(z: cache.z1[idx], a: val, grad: grad)
            }
            // Output
            for (idx, val) in cache.yHat.enumerated() {
                // Show dZ2 immediately
                let grad = grads.dZ2[idx]
                nodeValues[NodeID(layer: 2, index: idx)] = NodeVisualValues(z: cache.z2[idx], a: val, grad: grad)
            }
            
            // Edges (W2) - show gradients early
            for r in 0..<params.W2.count {
                for c in 0..<params.W2[r].count {
                    let eid = EdgeID(from: NodeID(layer: 1, index: c), to: NodeID(layer: 2, index: r))
                    let grad = grads.dW2[r][c]
                    edgeValues[eid] = EdgeVisualValues(w: params.W2[r][c], grad: grad)
                }
            }
            
            // Edges (W1) - show gradients late (t > 0.5)
            for r in 0..<params.W1.count {
                for c in 0..<params.W1[r].count {
                    let eid = EdgeID(from: NodeID(layer: 0, index: c), to: NodeID(layer: 1, index: r))
                    let grad = (t > 0.5) ? grads.dW1[r][c] : nil
                    edgeValues[eid] = EdgeVisualValues(w: params.W1[r][c], grad: grad)
                }
            }

            snapshots.append(StepSnapshot(
                id: UUID(),
                phase: .backward,
                t: t,
                params: params,
                cache: cache,
                loss: lossValue,
                grads: grads,
                nodeValues: nodeValues,
                edgeValues: edgeValues
            ))
        }
        
        // --- Phase 4: Update (Interpolate weights) ---
        // Ensure grads match params
        guard grads.dW1.count == params.W1.count,
              !grads.dW1.isEmpty && grads.dW1[0].count == params.W1[0].count,
              grads.dW2.count == params.W2.count else {
            // Return incomplete timeline if grads mismatch to avoid crash
            return (StepTimeline(snapshots: snapshots), params)
        }
        
        let updateFrames = 5
        for i in 0..<updateFrames {
            let t = Double(i) / Double(updateFrames - 1)
            
            // Interpolate weights: w(t) = wOld + t * (wNew - wOld)
            // Actually it's cleaner: w(t) = wOld - t * lr * grad
            
            var edgeValues = [EdgeID: EdgeVisualValues]()
            
            // W1
            for r in 0..<params.W1.count {
                for c in 0..<params.W1[r].count {
                    let wOld = params.W1[r][c]
                    let g = grads.dW1[r][c]
                    let wNow = wOld - t * lr * g
                    let eid = EdgeID(from: NodeID(layer: 0, index: c), to: NodeID(layer: 1, index: r))
                    edgeValues[eid] = EdgeVisualValues(w: wNow, grad: g, delta: -lr * g)
                }
            }
            
            // W2
            for r in 0..<params.W2.count {
                for c in 0..<params.W2[r].count {
                    let wOld = params.W2[r][c]
                    let g = grads.dW2[r][c]
                    let wNow = wOld - t * lr * g
                    let eid = EdgeID(from: NodeID(layer: 1, index: c), to: NodeID(layer: 2, index: r))
                    edgeValues[eid] = EdgeVisualValues(w: wNow, grad: g, delta: -lr * g)
                }
            }
            
            // Nodes don't change much during update, keep showing last state
            let lastNodes = snapshots.last?.nodeValues ?? [:]

            snapshots.append(StepSnapshot(
                id: UUID(),
                phase: .update,
                t: t,
                params: params, // technically params are shifting, but snapshot params usually store "base" or "current"? 
                                // Let's store interpolating params if we want to be strict, but for now `params` is the START of step.
                                // The renderer uses edgeValues.w for drawing.
                cache: cache,
                loss: lossValue,
                grads: grads,
                nodeValues: lastNodes,
                edgeValues: edgeValues
            ))
        }
        
        return (StepTimeline(snapshots: snapshots), newParams)
    }
}
