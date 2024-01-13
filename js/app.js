document.addEventListener('DOMContentLoaded', async function () {
    const inputText = document.getElementById('inputText');
    const outputDiv = document.getElementById('output');

    // Load the GPT-2 model
    const model = await tf.loadLayersModel('../model/de_3meister_LSTM_model.json');

    // Event listener for input changes
    inputText.addEventListener('input', function () {
        const input = inputText.value.trim().toLowerCase();

        if (input.length > 0) {
            // Tokenize the input and predict the next word
            const tokenizedInput = input.split(' ');
            const inputTensor = tf.tensor2d([tokenizedInput.map(word => word.charCodeAt(0))]);
            const prediction = model.predict(inputTensor);

            // Get the index of the predicted word
            const predictedIndex = tf.argMax(prediction, axis=1).dataSync()[0];

            // Get the predicted word
            const predictedWord = String.fromCharCode(predictedIndex);

            // Display the predicted word
            outputDiv.textContent = `Predicted Next Word: ${predictedWord}`;
        } else {
            outputDiv.textContent = '';
        }
    });
});