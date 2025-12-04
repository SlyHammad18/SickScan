document.addEventListener('DOMContentLoaded', () => {
    const symptomInput = document.getElementById('symptom-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const symptomsSection = document.getElementById('symptoms-section');
    const detectedSymptomsList = document.getElementById('detected-symptoms-list');
    const symptomCount = document.getElementById('symptom-count');
    const addSymptomInput = document.getElementById('add-symptom-input');
    const addSymptomBtn = document.getElementById('add-symptom-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultsSection = document.getElementById('results-section');
    const predictionsList = document.getElementById('predictions-list');
    const resetBtn = document.getElementById('reset-btn');
    const allSymptomsDatalist = document.getElementById('all-symptoms-list');

    let currentSymptoms = new Set(); // Stores IDs of currently selected symptoms
    let allSymptomsMap = new Map(); // Stores ID -> Name mapping

    // Fetch all symptoms for autocomplete
    fetch('/all_symptoms')
        .then(response => response.json())
        .then(data => {
            data.forEach(s => {
                const option = document.createElement('option');
                option.value = s.name;
                option.dataset.id = s.id; // Store ID in dataset (though datalist doesn't directly support this, we use map)
                allSymptomsDatalist.appendChild(option);
                allSymptomsMap.set(s.name.toLowerCase(), s.id);
                allSymptomsMap.set(s.id, s.name); // Store both ways for easy lookup
            });
        });

    // Analyze Text
    analyzeBtn.addEventListener('click', () => {
        const text = symptomInput.value.trim();
        if (!text) return;

        analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        analyzeBtn.disabled = true;

        fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            // Add detected symptoms
            data.symptoms.forEach(s => {
                addSymptomTag(s.id, s.name);
            });

            symptomsSection.classList.remove('hidden');
            symptomsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(err => console.error(err))
        .finally(() => {
            analyzeBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Analyze Symptoms';
            analyzeBtn.disabled = false;
        });
    });

    // Add Manual Symptom
    addSymptomBtn.addEventListener('click', () => {
        const val = addSymptomInput.value.trim();
        if (!val) return;

        const lowerVal = val.toLowerCase();
        // Try to find ID from map
        let id = null;
        if (allSymptomsMap.has(lowerVal)) {
            id = allSymptomsMap.get(lowerVal);
        } else {
            // Fallback: try to find exact match in datalist options if map lookup fails (unlikely if map is built correctly)
             // For now, if not in map, we assume it's not a valid symptom in our DB.
             // But let's check if the user typed an ID directly?
             if (allSymptomsMap.has(val)) { // Check if they typed an ID
                 id = val; 
             }
        }

        if (id) {
            addSymptomTag(id, val); // Use user's casing for display, or map's? Let's use map's name if available
            const officialName = allSymptomsMap.get(id);
            // If we found an ID, update the tag with the official name
            if (officialName && officialName !== id) {
                 // Remove the old tag if we added it with raw input (we haven't yet)
                 // Actually, addSymptomTag handles duplicates.
                 addSymptomTag(id, officialName);
            } else {
                addSymptomTag(id, val);
            }
            addSymptomInput.value = '';
        } else {
            alert("Symptom not recognized in database. Please select from the list.");
        }
    });

    // Get Diagnosis
    predictBtn.addEventListener('click', () => {
        if (currentSymptoms.size === 0) {
            alert("Please add at least one symptom.");
            return;
        }

        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Diagnosing...';
        predictBtn.disabled = true;

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: Array.from(currentSymptoms) })
        })
        .then(response => response.json())
        .then(data => {
            renderPredictions(data.predictions);
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(err => console.error(err))
        .finally(() => {
            predictBtn.innerHTML = '<i class="fa-solid fa-user-doctor"></i> Get Diagnosis';
            predictBtn.disabled = false;
        });
    });

    // Reset
    resetBtn.addEventListener('click', () => {
        currentSymptoms.clear();
        detectedSymptomsList.innerHTML = '';
        updateCount();
        symptomInput.value = '';
        resultsSection.classList.add('hidden');
        symptomsSection.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Helper: Add Tag
    function addSymptomTag(id, name) {
        if (currentSymptoms.has(id)) return;

        currentSymptoms.add(id);
        updateCount();

        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `
            <span>${name}</span>
            <i class="fa-solid fa-xmark" onclick="removeSymptom('${id}', this)"></i>
        `;
        detectedSymptomsList.appendChild(tag);
    }

    // Helper: Remove Tag (exposed to window for onclick)
    window.removeSymptom = function(id, element) {
        currentSymptoms.delete(id);
        updateCount();
        element.parentElement.remove();
        
        // Hide results if modified? Maybe not, let user re-click diagnose.
    };

    function updateCount() {
        symptomCount.textContent = currentSymptoms.size;
    }

    function renderPredictions(predictions) {
        predictionsList.innerHTML = '';
        predictions.forEach((p, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.1}s`;
            
            const symptomsStr = p.common_symptoms.length > 0 ? p.common_symptoms.join(', ') : 'N/A';

            card.innerHTML = `
                <div class="result-header">
                    <span class="disease-name">${p.disease}</span>
                    <span class="confidence">${p.confidence}%</span>
                </div>
                <div class="common-symptoms">
                    <strong>Common Symptoms:</strong> ${symptomsStr}
                </div>
            `;
            predictionsList.appendChild(card);
        });
    }
});
