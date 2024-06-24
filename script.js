let model;
const fileInput = document.getElementById('file-input');
const selectedImage = document.getElementById('selected-image');
const predictButton = document.getElementById('predict-button');
const predictionResult = document.getElementById('prediction-result');

// Cargar el modelo
async function loadModel() {
    model = await tf.loadLayersModel('carpeta_salida/model.json');
    console.log("Modelo cargado");
}

loadModel();

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
        selectedImage.src = e.target.result;
        selectedImage.style.display = 'block';
    };

    reader.readAsDataURL(file);
});

predictButton.addEventListener('click', async () => {
    if (!model) {
        predictionResult.innerText = "Modelo no cargado todavía.";
        return;
    }

    const image = document.getElementById('selected-image');
    const tensor = preprocessImage(image);
    const prediction = await model.predict(tensor).data();
    const result = prediction[0] > 0.5 ? "Tumor detectado" : "No hay tumor";
    predictionResult.innerText = result;
});

function preprocessImage(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([100, 100])
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat()
        .div(tf.scalar(255.0));
    return tensor;
}