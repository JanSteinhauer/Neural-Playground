import Foundation

struct BackpropEngine {
    
    // MARK: - Math Helpers
    static func sigmoid(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }
    
    static func dSigmoidFromActivation(_ a: Double) -> Double {
        return a * (1.0 - a)
    }
    
    // MARK: - Forward Pass
    static func forward(params: NetworkParams, x: [Double]) -> ForwardCache {
        // Validation
        guard !params.W1.isEmpty else { return ForwardCache(x: x, z1: [], a1: [], z2: [], yHat: []) }
        // Ensure input matches W1 input dim
        guard x.count == params.W1[0].count else {
            print("BackpropEngine Error: Input size \(x.count) != W1 columns \(params.W1[0].count)")
            return ForwardCache(x: x, z1: [], a1: [], z2: [], yHat: [])
        }
        
        // Layer 1: Hidden
        // z1[i] = dot(W1[i], x) + b1[i]
        var z1 = [Double]()
        var a1 = [Double]()
        
        for i in 0..<params.W1.count {
            let row = params.W1[i]
            let dot = zip(row, x).map(*).reduce(0, +)
            let z = dot + params.b1[i]
            z1.append(z)
            a1.append(sigmoid(z))
        }
        
        // Layer 2: Output
        // z2[j] = dot(W2[j], a1) + b2[j]
        var z2 = [Double]()
        var yHat = [Double]()
        
        for j in 0..<params.W2.count {
            let row = params.W2[j]
            let dot = zip(row, a1).map(*).reduce(0, +)
            let z = dot + params.b2[j]
            z2.append(z)
            yHat.append(sigmoid(z))
        }
        
        return ForwardCache(x: x, z1: z1, a1: a1, z2: z2, yHat: yHat)
    }
    
    // MARK: - Loss
    // MSE: 0.5 * (yHat - y)^2
    static func loss(yHat: [Double], y: [Double]) -> Double {
        var totalLoss: Double = 0.0
        for i in 0..<yHat.count {
            let diff = yHat[i] - y[i]
            totalLoss += 0.5 * diff * diff
        }
        return totalLoss
    }
    
    // MARK: - Backward Pass
    static func backward(params: NetworkParams, cache: ForwardCache, y: [Double]) -> Gradients {
        guard cache.yHat.count == y.count else { return Gradients(dW1: [], db1: [], dW2: [], db2: [], dZ1: [], dA1: [], dZ2: []) }
        guard !cache.a1.isEmpty else { return Gradients(dW1: [], db1: [], dW2: [], db2: [], dZ1: [], dA1: [], dZ2: []) }
        guard !cache.x.isEmpty else { return Gradients(dW1: [], db1: [], dW2: [], db2: [], dZ1: [], dA1: [], dZ2: []) }
        
        let yHat = cache.yHat
        let a1 = cache.a1
        let x = cache.x
        
        // 1. Output Layer Gradients
        // dL/dyHat = (yHat - y)
        // dyHat/dz2 = yHat * (1 - yHat)
        // dZ2 = (yHat - y) * yHat * (1 - yHat)
        
        var dZ2 = [Double]()
        for j in 0..<yHat.count {
            let error = yHat[j] - y[j] // dL/dyHat
            let derivative = dSigmoidFromActivation(yHat[j])
            dZ2.append(error * derivative)
        }
        
        // dW2 = dZ2 * a1^T
        // db2 = dZ2
        var dW2 = [[Double]]()
        var db2 = [Double]()
        
        // Since we have single sample, gradients are outer product
        for j in 0..<dZ2.count {
            var row = [Double]()
            for k in 0..<a1.count {
                row.append(dZ2[j] * a1[k])
            }
            dW2.append(row)
            db2.append(dZ2[j])
        }
        
        // 2. Hidden Layer Gradients
        // dA1 = W2^T * dZ2
        var dA1 = [Double](repeating: 0.0, count: a1.count)
        // W2 is [output_dim x hidden_dim] -> [1 x 2]
        // W2^T is [hidden_dim x output_dim] -> [2 x 1]
        
        for k in 0..<a1.count {
            var sum = 0.0
            for j in 0..<dZ2.count {
                sum += params.W2[j][k] * dZ2[j]
            }
            dA1[k] = sum
        }
        
        // dZ1 = dA1 * a1 * (1 - a1)
        var dZ1 = [Double]()
        for k in 0..<a1.count {
            dZ1.append(dA1[k] * dSigmoidFromActivation(a1[k]))
        }
        
        // dW1 = dZ1 * x^T
        // db1 = dZ1
        var dW1 = [[Double]]()
        var db1 = [Double]()
        
        for k in 0..<dZ1.count {
            var row = [Double]()
            for i in 0..<x.count {
                row.append(dZ1[k] * x[i])
            }
            dW1.append(row)
            db1.append(dZ1[k])
        }
        
        return Gradients(dW1: dW1, db1: db1, dW2: dW2, db2: db2, dZ1: dZ1, dA1: dA1, dZ2: dZ2)
    }
    
    // MARK: - Update
    static func apply(params: NetworkParams, grads: Gradients, lr: Double) -> NetworkParams {
        var newParams = params
        
        // Update W1, b1
        for r in 0..<params.W1.count {
            for c in 0..<params.W1[r].count {
                newParams.W1[r][c] -= lr * grads.dW1[r][c]
            }
            newParams.b1[r] -= lr * grads.db1[r]
        }
        
        // Update W2, b2
        for r in 0..<params.W2.count {
            for c in 0..<params.W2[r].count {
                newParams.W2[r][c] -= lr * grads.dW2[r][c]
            }
            newParams.b2[r] -= lr * grads.db2[r]
        }
        
        return newParams
    }
}
