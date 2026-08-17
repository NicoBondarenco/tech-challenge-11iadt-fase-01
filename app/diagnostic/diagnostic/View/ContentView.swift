import SwiftUI

struct ContentView: View {
    let featureKeys: [String] = [
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ]

    @State private var inputs: [String: String] = [:]
    @State private var isLoading = false
    @State private var response: PredictionResponse?
    @State private var showAlert = false
    @State private var errorMessage = ""
    
    let modelOptions: [(label: String, endpoint: String)] = [
        ("KNN", "/predict/knn"),
        ("Random Forest", "/predict/random_forest"),
        ("Decision Tree PCA", "/predict/decision_tree_pca"),
        ("KNN PCA", "/predict/knn_pca"),
        ("Logistic Regression", "/predict/logistic_regression"),
        ("Logistic Regression PCA", "/predict/logistic_regression_pca"),
        ("Random Forest PCA", "/predict/random_forest_pca"),
        ("SVM PCA", "/predict/svm_pca")
    ]

    @State private var selectedModelEndpoint: String = "/predict/knn"

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Modelo")) {
                    Picker("Modelo", selection: $selectedModelEndpoint) {
                        ForEach(modelOptions, id: \.endpoint) { opt in
                            Text(opt.label).tag(opt.endpoint)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }

                Section(header: Text("Features")) {
                    ForEach(featureKeys, id: \ .self) { key in
                        HStack {
                            Text(key.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.subheadline)
                            Spacer()
                            TextField("0", text: Binding(
                                get: { inputs[key] ?? "" },
                                set: { inputs[key] = $0 }
                            ))
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 120)
                        }
                    }
                }

                Section {
                    Button(action: send) {
                        if isLoading { ProgressView() } else { Text("Analisar") }
                    }
                    .disabled(isLoading || !isValid())
                }

                if let resp = response {
                    Section(header: Text("Resultado")) {
                        Text("Modelo: \(resp.model)")
                        Text("Diagnóstico: \(resp.prediction)")
                            .font(.headline)
                        Text("Confiança: \(resp.probability)%")
                    }
                }
            }
            .navigationTitle("Predição Câncer")
            .alert(isPresented: $showAlert) {
                Alert(title: Text("Erro"), message: Text(errorMessage), dismissButton: .default(Text("OK")))
            }
            .onAppear {
                // inicializar inputs vazios
                for key in featureKeys { inputs[key] = "" }
            }
        }
    }

    func isValid() -> Bool {
        for key in featureKeys {
            guard let v = inputs[key], !v.isEmpty, Double(v) != nil else { return false }
        }
        return true
    }

    func send() {
        isLoading = true
        response = nil

        // Converter para query string usando valores atuais
        var query: [String: String] = [:]
        for key in featureKeys {
            if let v = inputs[key] { query[key] = v }
        }

        // Usar o endpoint selecionado pelo usuário
        NetworkManager.shared.getPrediction(endpoint: selectedModelEndpoint, query: query) { result in
            DispatchQueue.main.async {
                isLoading = false
                switch result {
                case .success(let resp):
                    self.response = resp
                case .failure(let err):
                    self.errorMessage = err.localizedDescription
                    self.showAlert = true
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
