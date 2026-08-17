import Foundation

final class NetworkManager {
    static let shared = NetworkManager()
    private init() {}

    var baseURL = "http://127.0.0.1:8000"

    func getPrediction(endpoint: String = "/predict/knn", query: [String: String], completion: @escaping (Result<PredictionResponse, Error>) -> Void) {
        guard var components = URLComponents(string: baseURL + endpoint) else {
            completion(.failure(URLError(.badURL)))
            return
        }

        components.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }

        guard let url = components.url else {
            completion(.failure(URLError(.badURL)))
            return
        }

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }

            guard let data = data else {
                completion(.failure(URLError(.badServerResponse)))
                return
            }

            do {
                let decoder = JSONDecoder()
                let resp = try decoder.decode(PredictionResponse.self, from: data)
                completion(.success(resp))
            } catch {
                completion(.failure(error))
            }
        }

        task.resume()
    }
}
