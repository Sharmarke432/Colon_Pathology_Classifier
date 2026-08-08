const imageInput = document.querySelector('#image-input');
const dropZone = document.querySelector('#drop-zone');
const previewWrapper = document.querySelector('#preview-wrapper');
const imagePreview = document.querySelector('#image-preview');
const removeImageButton = document.querySelector('#remove-image');
const predictButton = document.querySelector('#predict-button');
const buttonLabel = document.querySelector('#button-label');
const spinner = document.querySelector('#spinner');
const errorMessage = document.querySelector('#error-message');
const emptyState = document.querySelector('#empty-state');
const predictionContent = document.querySelector('#prediction-content');
const probabilityList = document.querySelector('#probability-list');

let selectedFile = null;

function setError(message = '') {
  errorMessage.textContent = message;
  errorMessage.classList.toggle('hidden', !message);
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading || !selectedFile;
  buttonLabel.textContent = isLoading ? 'Classifying…' : 'Run prediction';
  spinner.classList.toggle('hidden', !isLoading);
}

function setSelectedFile(file) {
  if (!file) return;

  if (!file.type.startsWith('image/')) {
    setError('Please select a PNG, JPG, or WebP image.');
    return;
  }

  selectedFile = file;
  setError();
  imagePreview.src = URL.createObjectURL(file);
  imagePreview.alt = `Preview of ${file.name}`;
  previewWrapper.classList.remove('hidden');
  dropZone.classList.add('hidden');
  predictButton.disabled = false;
}

function clearSelectedFile() {
  selectedFile = null;
  imageInput.value = '';
  imagePreview.removeAttribute('src');
  previewWrapper.classList.add('hidden');
  dropZone.classList.remove('hidden');
  predictButton.disabled = true;
}

function formatPercentage(value) {
  const percentage = Number(value) <= 1 ? Number(value) * 100 : Number(value);
  return `${percentage.toFixed(1)}%`;
}

function getProbabilityEntries(probabilities) {
  if (Array.isArray(probabilities)) {
    return probabilities.map((item, index) => ({
      className: item.class_name ?? item.class_id ?? item.class ?? index,
      probability: item.probability ?? item.value ?? item.score ?? 0,
    }));
  }

  return Object.entries(probabilities ?? {}).map(([className, probability]) => ({
    className,
    probability,
  }));
}

function renderPrediction(data) {
  const predictedClass = data.predicted_class ?? data.class_id ?? data.prediction ?? '—';
  const predictedLabel = data.label ?? data.predicted_label ?? predictedClass;
  const confidence = data.confidence ?? 0;
  const probabilities = getProbabilityEntries(data.probabilities);

  document.querySelector('#predicted-label').textContent = predictedLabel;
  document.querySelector('#predicted-class').textContent = `Class ${predictedClass}`;
  document.querySelector('#confidence').textContent = formatPercentage(confidence);

  probabilityList.innerHTML = probabilities
    .sort((a, b) => Number(b.probability) - Number(a.probability))
    .map(({ className, probability }) => {
      const percentage = Number(probability) <= 1 ? Number(probability) * 100 : Number(probability);
      return `
        <div class="probability-row">
          <span>Class ${className}</span>
          <div class="probability-track" role="progressbar" aria-valuenow="${percentage.toFixed(1)}" aria-valuemin="0" aria-valuemax="100">
            <div class="probability-bar" style="width: ${Math.min(Math.max(percentage, 0), 100)}%"></div>
          </div>
          <span class="probability-value">${formatPercentage(probability)}</span>
        </div>`;
    })
    .join('');

  emptyState.classList.add('hidden');
  predictionContent.classList.remove('hidden');
}

async function runPrediction() {
  if (!selectedFile) return;

  setError();
  setLoading(true);
  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('The API returned an invalid response.');
    }

    if (!response.ok) {
      throw new Error(data.detail ?? 'Prediction failed. Please try another image.');
    }

    renderPrediction(data);
  } catch (error) {
    setError(error.message || 'Unable to reach the prediction API.');
  } finally {
    setLoading(false);
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch('/model-info');
    if (!response.ok) return;
    const info = await response.json();

    if (info.model_name) document.querySelector('#model-name').textContent = info.model_name;
    if (info.num_classes) document.querySelector('#class-count').textContent = info.num_classes;
    if (info.image_size) {
      const size = Array.isArray(info.image_size) ? info.image_size.join('×') : info.image_size;
      document.querySelector('#image-size').textContent = size;
    }
  } catch {
    // The static fallback metadata remains visible if model-info is unavailable.
  }
}

imageInput.addEventListener('change', (event) => setSelectedFile(event.target.files[0]));
removeImageButton.addEventListener('click', clearSelectedFile);
predictButton.addEventListener('click', runPrediction);

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  setSelectedFile(event.dataTransfer.files[0]);
});

loadModelInfo();
