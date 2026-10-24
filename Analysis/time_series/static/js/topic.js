async function startAnalysis() {

    const btn = document.getElementById('run-btn');
    const statusSection = document.getElementById('status-section');
    const statusText = document.getElementById('status-text');
    const statusItems = document.getElementById('status-items');
    const resultContainer = document.getElementById('result-container');

    statusItems.innerHTML = '';
    resultContainer.style.display = 'none';
    statusSection.style.display = 'block';
    btn.disabled = true;

    const payload = {
        days: document.getElementById('days').value,
        min_score: document.getElementById('min_score').value,
        max_items: document.getElementById('max_items').value,
        refresh: document.getElementById('refresh').checked,
        type: 'story'
    };

    try {
        const response = await fetch('/analyse_topics', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error);
        }
        finishAnalysis(data.images);
    } catch (err) {
        addStatusItem(`Error: ${err.message}`, "error");
        btn.disabled = false;
    }
}

function finishAnalysis(images) {
    const btn = document.getElementById('run-btn');
    const resultContainer = document.getElementById('result-container');
    const spinner = document.getElementById('main-spinner');
    
    const ts = new Date().getTime();
    document.getElementById('cluster-img').src = images[1] + "?v=" + ts;
    document.getElementById('dist-img').src = images[0] + "?v=" + ts;

    spinner.style.display = 'none';
    resultContainer.style.display = 'block';
    btn.disabled = false;
    addStatusItem("Analysis complete. Visualizations rendered.", "success");
}

function addStatusItem(text, className) {
    const container = document.getElementById('status-items');
    const div = document.createElement('div');
    div.className = `status-item ${className}`;
    div.innerText = `> ${text}`;
    container.prepend(div);
}
