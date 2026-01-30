import SwiftUI

struct GraphLayout {
    let nodePositions: [NodeID: CGPoint]
    let edgePaths: [EdgeID: Path]
    
    // Topology: 2 inputs -> 2 hidden -> 1 output
    static let standard = GraphLayout(
        layers: [2, 2, 1],
        size: CGSize(width: 350, height: 300)
    )
    
    init(layers: [Int], size: CGSize) {
        var positions = [NodeID: CGPoint]()
        var paths = [EdgeID: Path]()
        
        // 1. Compute Node Positions
        // We'll distribute layers horizontally
        // Layer 0 is at x=50, Layer N is at x=size.width-50
        // Or slightly more centered?
        
        let marginX: CGFloat = 60
        let availableWidth = size.width - 2 * marginX
        let layerSpacing = availableWidth / CGFloat(max(1, layers.count - 1))
        
        let neuronSpacing: CGFloat = 80
        let centerY = size.height / 2
        
        for (l, count) in layers.enumerated() {
            let x = marginX + CGFloat(l) * layerSpacing
            
            // Center neurons vertically
            let totalHeight = CGFloat(count - 1) * neuronSpacing
            let startY = centerY - totalHeight / 2
            
            for i in 0..<count {
                let id = NodeID(layer: l, index: i)
                let p = CGPoint(x: x, y: startY + CGFloat(i) * neuronSpacing)
                positions[id] = p
            }
        }
        
        self.nodePositions = positions
        
        // 2. Compute Edge Paths (Curved Beziers)
        // Edges exist between Layer L and L+1 (fully connected)
        for l in 0..<(layers.count - 1) {
            let currentCount = layers[l]
            let nextCount = layers[l+1]
            
            for i in 0..<currentCount {
                for j in 0..<nextCount {
                    let fromID = NodeID(layer: l, index: i)
                    let toID = NodeID(layer: l+1, index: j)
                    
                    if let p1 = positions[fromID], let p2 = positions[toID] {
                        let eid = EdgeID(from: fromID, to: toID)
                        
                        var path = Path()
                        path.move(to: p1)
                        // Control points for S-curve
                        let cp1 = CGPoint(x: p1.x + layerSpacing * 0.5, y: p1.y)
                        let cp2 = CGPoint(x: p2.x - layerSpacing * 0.5, y: p2.y)
                        path.addCurve(to: p2, control1: cp1, control2: cp2)
                        
                        paths[eid] = path
                    }
                }
            }
        }
        
        self.edgePaths = paths
    }
    
    func position(for id: NodeID) -> CGPoint {
        return nodePositions[id] ?? .zero
    }
}
