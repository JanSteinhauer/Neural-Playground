import Foundation
import SwiftUI
import Combine

@MainActor
class TrainingViewModel: ObservableObject {
    
    // MARK: - State
    @Published var params: NetworkParams
    @Published var timeline: StepTimeline?
    @Published var lossHistory: [Double] = []
    
    // Playback / scrubbing control
    // Double for slider smoothness, but we map to Int index
    @Published var scrubberValue: Double = 0.0
    
    @Published var lr: Double = 0.5
    @Published var selectedSampleIndex: Int = 0
    
    @Published var selectedNode: NodeID? = nil
    
    // XOR Data
    let samples: [([Double], [Double])] = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    // MARK: - Integration
    
    init() {
        self.params = NetworkParams.random()
        // Build initial timeline for the first sample
        self.refreshTimeline()
    }
    
    // MARK: - Computed Properties
    
    var currentSnapshot: StepSnapshot? {
        guard let timeline = timeline else { return nil }
        guard !timeline.snapshots.isEmpty else { return nil }
        let idx = Int(scrubberValue)
        if idx >= 0 && idx < timeline.snapshots.count {
            return timeline.snapshots[idx]
        }
        return timeline.snapshots.last
    }
    
    var maxScrubberValue: Double {
        guard let timeline = timeline else { return 0 }
        return Double(timeline.snapshots.count - 1)
    }
    
    // MARK: - Actions
    
    func trainOneStep() {
        guard let timeline = timeline, let lastSnapshot = timeline.snapshots.last else { return }
        
        // Use the params from the END of the current timeline (which are updated)
        // Wait, current timeline generation returns (Timeline, NewParams).
        // My makeTimeline returns (StepTimeline, NetworkParams).
        // So I should have stored the 'next' params somewhere or applying them now.
        
        // Actually, `makeTimeline` computes the update but returns it. 
        // If we want to commit the update, we set self.params = newParam (returned from makeTimeline).
        
        // BUT, if I am just VIWEING the step, I haven't committed it yet?
        // Usually: "Train Step" -> Commits the update to `params`, then generates a NEW timeline for the *next* step.
        // The CURRENT timeline shows the transition from params_old to params_new.
        
        // So:
        // 1. We are viewing step N. we have params_N inside the snapshot?
        //    Actually, `Timeline.swift` computes `newParams` but we discard it in the current code?
        //    I need to fix `TimelineGenerator` calls or store the result.
        
        // Let's re-run makeTimeline to get the new params (deterministic if we use same inputs).
        let (newTimeline, nextParams) = TimelineGenerator.makeTimeline(
            params: self.params,
            x: samples[selectedSampleIndex].0,
            y: samples[selectedSampleIndex].1,
            lr: lr
        )
        
        // Commit the update
        self.params = nextParams
        
        // Record loss from this step (it's in the snapshot? or from the generator?)
        // TimelineGenerator computes forward/loss/backward.
        // The loss in the timeline corresponds to the loss BEFORE the update (usually).
        // Let's grab the loss from the new timeline (Phase .loss snapshots).
        if let lossSnap = newTimeline.snapshots.first(where: { $0.phase == .loss }) {
            self.lossHistory.append(lossSnap.loss)
            if self.lossHistory.count > 50 { self.lossHistory.removeFirst() }
        }
        
        self.timeline = newTimeline
        self.scrubberValue = 0 // Reset to start of step
    }
    
    func reset() {
        self.params = NetworkParams.random()
        self.lossHistory = []
        refreshTimeline()
    }
    
    func refreshTimeline() {
        // Just generate a timeline for the current state without updating (pseudo-step? or just forward pass?)
        // If we haven't trained, we can just show a Forward pass.
        // But `makeTimeline` does the full update. 
        // Let's just generate a full step visualization but NOT commit the params yet?
        // No, `trainOneStep` implies moving forward.
        
        // If I just want to see "Current State", I essentially want a timeline that just does Forward?
        // For POC, let's just always compute a full "Proposed Step" timeline.
        // i.e. "Here is what happens if you train on this sample".
        
        let (t, _) = TimelineGenerator.makeTimeline(
            params: self.params,
            x: samples[selectedSampleIndex].0,
            y: samples[selectedSampleIndex].1,
            lr: lr
        )
        self.timeline = t
        self.scrubberValue = 0
    }
    
    func selectSample(index: Int) {
        self.selectedSampleIndex = index
        refreshTimeline()
    }
}
