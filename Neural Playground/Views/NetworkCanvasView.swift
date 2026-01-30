import SwiftUI

struct NetworkCanvasView: View {
    let snapshot: StepSnapshot
    let selectedNode: NodeID?
    let onSelectNode: (NodeID) -> Void
    
    // Use the static layout for standard topology
    private let layout = GraphLayout.standard
    
    var body: some View {
        Canvas { context, size in
            // Draw Edges first (behind nodes)
            drawEdges(context: context)
            
            // Draw Nodes
            drawNodes(context: context)
        }
        .gesture(
            DragGesture(minimumDistance: 0).onEnded { value in
                let location = value.location
                if let hit = hitTest(location) {
                    onSelectNode(hit)
                }
            }
        )
        .frame(width: 350, height: 300) // Fixed size to match layout for POC
    }
    
    private func drawEdges(context: GraphicsContext) {
        for (edgeID, path) in layout.edgePaths {
            // Get visual state for this edge
            if let visual = snapshot.edgeValues[edgeID] {
                // 1. Base Stroke: Thickness ∝ abs(weight)
                // Range typical: 0...2 -> 1...6 width
                let wAbs = abs(visual.w)
                let lineWidth = 1.0 + wAbs * 4.0
                let color: Color = (visual.w >= 0) ? .blue : .red
                let opacity = 0.4 + wAbs * 0.4 // Faint for small weights
                
                context.stroke(
                    path,
                    with: .color(color.opacity(opacity)),
                    lineWidth: lineWidth
                )
                
                // 2. Gradient Overlay (if dW is present & significant)
                if let grad = visual.grad, abs(grad) > 0.001 {
                    // Highlight "active learning" edges in yellow
                    context.stroke(
                        path,
                        with: .color(.yellow.opacity(0.8)),
                        lineWidth: lineWidth + 2
                    )
                }
                
                // 3. Update Delta Overlay (if weights are changing)
                if let delta = visual.delta, abs(delta) > 0.0001 {
                    // Show flow? Just a white dash or glow
                    context.stroke(
                        path,
                        with: .color(.white.opacity(0.6)),
                        style: StrokeStyle(lineWidth: 1, dash: [4, 4])
                    )
                }
            }
        }
    }
    
    private func drawNodes(context: GraphicsContext) {
        for (id, pos) in layout.nodePositions {
            let rect = CGRect(x: pos.x - 15, y: pos.y - 15, width: 30, height: 30)
            let circle = Path(ellipseIn: rect)
            
            // Visual State
            let visual = snapshot.nodeValues[id]
            let activation = visual?.a ?? 0.0
            
            // Fill: 0=White -> 1=Black/Blue? 
            // Or typically: White -> Black (darker = higher activation)
            // Or White -> Blue
            let intensity = max(0, min(1, activation))
            let fillColor = Color(white: 1.0 - intensity * 0.9) // White -> Dark Grey
            
            // Check for gradients (Backward pass)
            if let grad = visual?.grad, abs(grad) > 0.001 {
                // Glow yellow/red if high gradient
                let glowColor = Color.orange.opacity(0.6)
                context.fill(circle, with: .color(glowColor))
                context.fill(Path(ellipseIn: rect.insetBy(dx: 3, dy: 3)), with: .color(fillColor))
            } else {
                context.fill(circle, with: .color(fillColor))
            }
            
            // Stroke
            var strokeColor = Color.black
            var strokeWidth = 2.0
            
            if selectedNode == id {
                strokeColor = .green
                strokeWidth = 3.0
            }
            
            context.stroke(circle, with: .color(strokeColor), lineWidth: strokeWidth)
        }
    }
    
    private func hitTest(_ point: CGPoint) -> NodeID? {
        // Find closest node within radius
        for (id, pos) in layout.nodePositions {
            let dist = hypot(point.x - pos.x, point.y - pos.y)
            if dist < 25 { // slightly larger than radius 15 for easier tap
                return id
            }
        }
        return nil
    }
}
