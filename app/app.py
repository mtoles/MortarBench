"""
Flask app to compare model accuracy between GPT-5 and other models.

Shows examples where at least one model is wrong, with collapsible divs.
"""

import json
from flask import Flask, render_template_string, redirect
from pathlib import Path

app = Flask(__name__)

# Allow all origins for port forwarding
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Hardcoded paths to results
GPT5_RESULTS_PATH = Path("app/results/2026-01-16_13-15-59/gpt-5_results.jsonl")
SOLO_RESULTS_PATH = Path("app/results/2025-12-31_12-14-29/solo_results.jsonl")

def load_model_results(path):
    """Load results from a JSONL file."""
    results = []
    with open(path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def compare_models(gpt5_results, other_results, other_model_name):
    """Compare GPT-5 with another model and return discrepancies."""
    comparisons = []
    
    for idx, (gpt5, other) in enumerate(zip(gpt5_results, other_results)):
        # Skip PII questions (matching stats calculation)
        if gpt5.get('pii') is True:
            continue
        
        # Verify they're the same question
        assert gpt5['question'] == other['question'], f"Question mismatch at index {idx}"
        assert gpt5['loan_id'] == other['loan_id'], f"Loan ID mismatch at index {idx}"
        
        # Get correctness - handle list format
        gpt5_correct = gpt5['exact_match']
        other_correct = other['exact_match']
        
        if isinstance(gpt5_correct, list):
            gpt5_correct = gpt5_correct[0] if len(gpt5_correct) > 0 else None
        if isinstance(other_correct, list):
            other_correct = other_correct[0] if len(other_correct) > 0 else None
        
        # Skip if exact_match is None (not evaluated)
        if gpt5_correct is None or other_correct is None:
            continue
        
        # Get F1 scores
        gpt5_f1 = gpt5.get('f1_score')
        other_f1 = other.get('f1_score')
        
        if isinstance(gpt5_f1, list):
            gpt5_f1 = gpt5_f1[0] if len(gpt5_f1) > 0 else None
        if isinstance(other_f1, list):
            other_f1 = other_f1[0] if len(other_f1) > 0 else None
        
        # Get predictions
        gpt5_pred = gpt5['pred']
        other_pred = other['pred']
        
        if isinstance(gpt5_pred, list) and len(gpt5_pred) > 0:
            gpt5_pred = gpt5_pred[0]
        if isinstance(other_pred, list) and len(other_pred) > 0:
            other_pred = other_pred[0]
        
        # Skip if both are correct
        if gpt5_correct and other_correct:
            continue
        
        # Add to comparisons
        comparison = {
            'index': idx,
            'loan_id': gpt5['loan_id'],
            'question': gpt5['question'],
            'true_answer': gpt5['answer'],
            'answer_type': gpt5['answer_type'],
            'gpt5_pred': gpt5_pred,
            'gpt5_correct': gpt5_correct,
            'gpt5_f1': gpt5_f1,
            f'{other_model_name}_pred': other_pred,
            f'{other_model_name}_correct': other_correct,
            f'{other_model_name}_f1': other_f1,
        }
        
        # Add raw_answer from GPT-5 if it exists
        if 'raw_answer' in gpt5:
            gpt5_raw = gpt5['raw_answer']
            if isinstance(gpt5_raw, list) and len(gpt5_raw) > 0:
                gpt5_raw = gpt5_raw[0]
            comparison['gpt5_raw_answer'] = gpt5_raw
        
        comparisons.append(comparison)
    
    return comparisons

def calculate_stats(gpt5_results, other_results):
    """Calculate accuracy statistics.
    
    Matches the markdown summary calculation which:
    1. Sums all trials (if multiple trials exist)
    2. Filters to PII==False subset (matching markdown's "Overall" metric)
    """
    gpt5_correct = 0
    other_correct = 0
    both_correct = 0
    both_wrong = 0
    total_trials_gpt5 = 0
    total_trials_other = 0
    
    for gpt5, other in zip(gpt5_results, other_results):
        # Filter to PII==False (matching markdown calculation)
        if gpt5.get('pii') is True:
            continue
        
        # Get exact_match values - handle lists (multiple trials)
        gpt5_em = gpt5.get('exact_match')
        other_em = other.get('exact_match')
        
        # Sum across all trials (matching markdown calculation)
        if isinstance(gpt5_em, list):
            gpt5_correct_count = sum(1 for x in gpt5_em if x)
            total_trials_gpt5 += len(gpt5_em)
            gpt5_c = gpt5_em[0] if len(gpt5_em) > 0 else False
        else:
            gpt5_correct_count = 1 if gpt5_em else 0
            total_trials_gpt5 += 1
            gpt5_c = gpt5_em if gpt5_em is not None else False
        
        if isinstance(other_em, list):
            other_correct_count = sum(1 for x in other_em if x)
            total_trials_other += len(other_em)
            other_c = other_em[0] if len(other_em) > 0 else False
        else:
            other_correct_count = 1 if other_em else 0
            total_trials_other += 1
            other_c = other_em if other_em is not None else False
        
        # Accumulate correct counts (sum across trials)
        gpt5_correct += gpt5_correct_count
        other_correct += other_correct_count
        
        # For both_correct/both_wrong, use first trial result
        if gpt5_c and other_c:
            both_correct += 1
        if not gpt5_c and not other_c:
            both_wrong += 1
    
    # Calculate accuracy matching markdown: sum(correct) / sum(trials)
    total = len([r for r in gpt5_results if r.get('pii') is not True])
    
    return {
        'total': total,
        'gpt5_correct': gpt5_correct,
        'other_correct': other_correct,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'gpt5_accuracy': (gpt5_correct / total_trials_gpt5 * 100) if total_trials_gpt5 > 0 else 0,
        'other_accuracy': (other_correct / total_trials_other * 100) if total_trials_other > 0 else 0,
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison: GPT-5 vs {{ model_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .nav {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav a {
            margin-right: 15px;
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        .nav a:hover {
            text-decoration: underline;
        }
        .stats {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .comparison-item {
            background: white;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .comparison-header {
            padding: 15px 20px;
            cursor: pointer;
            background: #fff;
            border-left: 5px solid #e74c3c;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }
        .comparison-header:hover {
            background: #f8f9fa;
        }
        .comparison-header.both-wrong {
            border-left-color: #e74c3c;
        }
        .comparison-header.gpt5-only-wrong {
            border-left-color: #f39c12;
        }
        .comparison-header.other-only-wrong {
            border-left-color: #9b59b6;
        }
        .header-title {
            flex: 1;
            font-weight: 500;
            color: #2c3e50;
        }
        .header-indicators {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .indicator {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .correct {
            background: #d4edda;
            color: #155724;
        }
        .wrong {
            background: #f8d7da;
            color: #721c24;
        }
        .comparison-content {
            display: none;
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        .comparison-content.active {
            display: block;
        }
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metadata-item {
            padding: 12px;
            background: white;
            border-radius: 6px;
            border-left: 3px solid #3498db;
        }
        .metadata-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        .metadata-value {
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.95em;
            color: #2c3e50;
            word-break: break-all;
        }
        .models-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .model-result {
            padding: 15px;
            background: white;
            border-radius: 6px;
        }
        .model-name {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .result-field {
            margin-bottom: 12px;
        }
        .result-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 4px;
        }
        .result-value {
            font-family: 'Monaco', 'Courier New', monospace;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .solo-answer {
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 3px solid #9b59b6;
        }
        .solo-answer-label {
            font-weight: bold;
            margin-bottom: 10px;
            color: #9b59b6;
        }
        .solo-answer-content {
            font-family: 'Monaco', 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 0.85em;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
        }
        .toggle-icon {
            transition: transform 0.2s;
        }
        .toggle-icon.active {
            transform: rotate(90deg);
        }
    </style>
</head>
<body>
    <h1>Model Comparison: GPT-5 vs {{ model_name }}</h1>
    
    <div class="stats">
        <h2>Statistics</h2>
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-label">Total Examples</div>
                <div class="stat-value">{{ stats.total }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">GPT-5 Accuracy</div>
                <div class="stat-value">{{ "%.1f"|format(stats.gpt5_accuracy) }}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">{{ model_name }} Accuracy</div>
                <div class="stat-value">{{ "%.1f"|format(stats.other_accuracy) }}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Both Correct</div>
                <div class="stat-value">{{ stats.both_correct }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Both Wrong</div>
                <div class="stat-value">{{ stats.both_wrong }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Discrepancies</div>
                <div class="stat-value">{{ comparisons|length }}</div>
            </div>
        </div>
    </div>
    
    <h2>Discrepancies ({{ comparisons|length }} examples where at least one model is wrong)</h2>
    
    {% for comp in comparisons %}
    <div class="comparison-item">
        <div class="comparison-header {% if not comp.gpt5_correct and not comp[model_name + '_correct'] %}both-wrong{% elif not comp.gpt5_correct %}gpt5-only-wrong{% else %}other-only-wrong{% endif %}" 
             onclick="toggleComparison({{ loop.index0 }})">
            <div class="header-title">
                #{{ comp.index }} - {{ comp.question[:100] }}{% if comp.question|length > 100 %}...{% endif %}
            </div>
            <div class="header-indicators">
                <span class="indicator {% if comp.gpt5_correct %}correct{% else %}wrong{% endif %}">
                    GPT-5: {% if comp.gpt5_correct %}✓{% else %}✗{% endif %} (F1: {% if comp.gpt5_f1 is not none %}{{ "%.3f"|format(comp.gpt5_f1) }}{% else %}N/A{% endif %})
                </span>
                <span class="indicator {% if comp[model_name + '_correct'] %}correct{% else %}wrong{% endif %}">
                    {{ model_name }}: {% if comp[model_name + '_correct'] %}✓{% else %}✗{% endif %} (F1: {% if comp[model_name + '_f1'] is not none %}{{ "%.3f"|format(comp[model_name + '_f1']) }}{% else %}N/A{% endif %})
                </span>
                <span class="toggle-icon" id="icon-{{ loop.index0 }}">▶</span>
            </div>
        </div>
        <div class="comparison-content" id="content-{{ loop.index0 }}">
            <div class="metadata">
                <div class="metadata-item">
                    <div class="metadata-label">Loan ID</div>
                    <div class="metadata-value">{{ comp.loan_id }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Answer Type</div>
                    <div class="metadata-value">{{ comp.answer_type }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">True Answer</div>
                    <div class="metadata-value">{{ comp.true_answer }}</div>
                </div>
            </div>
            
            <div class="metadata-item" style="grid-column: 1/-1;">
                <div class="metadata-label">Question</div>
                <div class="metadata-value">{{ comp.question }}</div>
            </div>
            
            <div class="models-comparison">
                <div class="model-result">
                    <div class="model-name">GPT-5</div>
                    <div class="result-field">
                        <div class="result-label">Prediction</div>
                        <div class="result-value">{{ comp.gpt5_pred }}</div>
                    </div>
                    <div class="result-field">
                        <div class="result-label">Correct</div>
                        <div class="result-value">{{ comp.gpt5_correct }}</div>
                    </div>
                    <div class="result-field">
                        <div class="result-label">F1 Score</div>
                        <div class="result-value">{% if comp.gpt5_f1 is not none %}{{ "%.3f"|format(comp.gpt5_f1) }}{% else %}N/A{% endif %}</div>
                    </div>
                </div>
                
                <div class="model-result">
                    <div class="model-name">{{ model_name }}</div>
                    <div class="result-field">
                        <div class="result-label">Prediction</div>
                        <div class="result-value">{{ comp[model_name + '_pred'] }}</div>
                    </div>
                    <div class="result-field">
                        <div class="result-label">Correct</div>
                        <div class="result-value">{{ comp[model_name + '_correct'] }}</div>
                    </div>
                    <div class="result-field">
                        <div class="result-label">F1 Score</div>
                        <div class="result-value">{% if comp[model_name + '_f1'] is not none %}{{ "%.3f"|format(comp[model_name + '_f1']) }}{% else %}N/A{% endif %}</div>
                    </div>
                </div>
            </div>
            
            {% if 'gpt5_raw_answer' in comp %}
            <div class="solo-answer">
                <div class="solo-answer-label">Raw GPT-5 Output (before cleaning)</div>
                <div class="solo-answer-content">{{ comp.gpt5_raw_answer }}</div>
            </div>
            {% endif %}
        </div>
    </div>
    {% endfor %}
    
    <script>
        function toggleComparison(index) {
            const content = document.getElementById('content-' + index);
            const icon = document.getElementById('icon-' + index);
            content.classList.toggle('active');
            icon.classList.toggle('active');
        }
    </script>
</body>
</html>
"""

@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'message': 'Flask app is running'}, 200

@app.route('/')
def index():
    """Redirect to comparison page."""
    return redirect('/compare/solo')

@app.route('/compare/<model_name>')
def compare(model_name):
    """Compare GPT-5 with specified model."""
    
    # Load results
    if not GPT5_RESULTS_PATH.exists():
        return f"GPT-5 results not found at {GPT5_RESULTS_PATH}", 404
    
    if not SOLO_RESULTS_PATH.exists():
        return f"Solo results not found at {SOLO_RESULTS_PATH}", 404
    
    gpt5_results = load_model_results(GPT5_RESULTS_PATH)
    solo_results = load_model_results(SOLO_RESULTS_PATH)
    
    # Verify same length
    if len(gpt5_results) != len(solo_results):
        return f"Results length mismatch: GPT-5 has {len(gpt5_results)}, solo has {len(solo_results)}", 400
    
    # Get comparisons and stats
    comparisons = compare_models(gpt5_results, solo_results, model_name)
    stats = calculate_stats(gpt5_results, solo_results)
    
    return render_template_string(
        HTML_TEMPLATE,
        model_name=model_name,
        comparisons=comparisons,
        stats=stats
    )

if __name__ == '__main__':
    # Check that result files exist
    print("Checking result files...")
    if GPT5_RESULTS_PATH.exists():
        print(f"✓ GPT-5: {GPT5_RESULTS_PATH}")
    else:
        print(f"✗ GPT-5: {GPT5_RESULTS_PATH} (NOT FOUND)")
    
    if SOLO_RESULTS_PATH.exists():
        print(f"✓ Solo: {SOLO_RESULTS_PATH}")
    else:
        print(f"✗ Solo: {SOLO_RESULTS_PATH} (NOT FOUND)")
    
    print("\nStarting Flask server...")
    print("Navigate to http://localhost:5002")
    print("For port forwarding, use: http://<your-ip>:5002")
    # Run with host 0.0.0.0 to accept connections from all interfaces
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)

