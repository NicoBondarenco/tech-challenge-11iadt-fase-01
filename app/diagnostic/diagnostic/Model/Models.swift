import Foundation

struct PredictionResponse: Codable {
    let model: String
    let prediction: String
    let probability: Int
}
