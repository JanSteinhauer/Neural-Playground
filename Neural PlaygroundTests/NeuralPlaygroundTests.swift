import XCTest
@testable import Neural_Playground

@MainActor
final class NeuralPlaygroundTests: XCTestCase {

    // MARK: - Shape Tests
    func testForwardShapes() {
        print("Starting testForwardShapes")
        // Arrange
        let params = NetworkParams.random()
        let x = [0.0, 1.0]

        // Act
        let cache = BackpropEngine.forward(params: params, x: x)

        // Assert
        // Inputs
        XCTAssertEqual(cache.x.count, 2, "Input shape should be 2")
        // Layer 1 (Hidden): 2 neurons
        XCTAssertEqual(cache.z1.count, 2, "z1 shape should be 2")
        XCTAssertEqual(cache.a1.count, 2, "a1 shape should be 2")
        // Layer 2 (Output): 1 neuron
        XCTAssertEqual(cache.z2.count, 1, "z2 shape should be 1")
        XCTAssertEqual(cache.yHat.count, 1, "yHat shape should be 1")
    }

    // MARK: - Gradient Checking
    func testBackwardFiniteDifference() {
        // Verify dL/dW numerically for a single weight
        var params = NetworkParams.random()
        let x = [0.5, -0.5] // Arbitrary input
        let y = [1.0]       // Arbitrary target
        
        // 1. Compute Analytic Gradient
        let cache = BackpropEngine.forward(params: params, x: x)
        let grads = BackpropEngine.backward(params: params, cache: cache, y: y)
        
        // Let's check W1[0][0]
        let analyticGrad = grads.dW1[0][0]
        
        // 2. Compute Numerical Gradient
        let epsilon = 1e-5
        
        // f(w + epsilon)
        var paramsPlus = params
        paramsPlus.W1[0][0] += epsilon
        let cachePlus = BackpropEngine.forward(params: paramsPlus, x: x)
        let lossPlus = BackpropEngine.loss(yHat: cachePlus.yHat, y: y)
        
        // f(w - epsilon)
        var paramsMinus = params
        paramsMinus.W1[0][0] -= epsilon
        let cacheMinus = BackpropEngine.forward(params: paramsMinus, x: x)
        let lossMinus = BackpropEngine.loss(yHat: cacheMinus.yHat, y: y)
        
        // Numerical Grad = (f(x+h) - f(x-h)) / 2h
        let numericGrad = (lossPlus - lossMinus) / (2 * epsilon)
        
        // Assert
        let diff = abs(analyticGrad - numericGrad)
        XCTAssertLessThan(diff, 1e-4, "Analytic gradient \(analyticGrad) differs from numeric \(numericGrad)")
    }

    // MARK: - Update Logic
    func testApplyMovesOppositeGradient() {
        var params = NetworkParams.random()
        let x = [0.5, 0.5]
        let y = [0.0]
        let lr = 0.1
        
        // Initial Loss
        let cache0 = BackpropEngine.forward(params: params, x: x)
        let loss0 = BackpropEngine.loss(yHat: cache0.yHat, y: y)
        
        // Compute Update
        let grads = BackpropEngine.backward(params: params, cache: cache0, y: y)
        let newParams = BackpropEngine.apply(params: params, grads: grads, lr: lr)
        
        // New Loss
        let cache1 = BackpropEngine.forward(params: newParams, x: x)
        let loss1 = BackpropEngine.loss(yHat: cache1.yHat, y: y)
        
        XCTAssertLessThan(loss1, loss0, "Loss should decrease after a gradient step (for sufficiently small lr)")
    }
    
    // MARK: - Loss History
    @MainActor
    func testLossHistory() {
        let vm = TrainingViewModel()
        
        XCTAssertTrue(vm.lossHistory.isEmpty, "History starts empty")
        
        // Train 1 step
        vm.trainOneStep()
        
        XCTAssertEqual(vm.lossHistory.count, 1, "History has 1 item after 1 step")
        
        // Train 100 steps
        for _ in 0..<100 {
            vm.trainOneStep()
        }
        
        XCTAssertLessThanOrEqual(vm.lossHistory.count, 50, "History should be capped at 50")
        XCTAssertEqual(vm.lossHistory.count, 50, "History should be full at 50")
    }
}
