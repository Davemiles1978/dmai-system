#!/usr/bin/env python3
"""
Visualize learning trends from self-assessment
"""
import json
from pathlib import Path

def plot_learning_curve():
    assessment_path = Path("shared_data/agi_evolution/assessment")
    if not assessment_path.exists():
        print("❌ No assessment data found")
        return
    
    reports = list(assessment_path.glob("assessment_*.json"))
    if len(reports) < 2:
        print("Not enough data for trend visualization (need at least 2 reports)")
        return
    
    generations = []
    learning_rates = []
    improvement_rates = []
    
    for report in sorted(reports)[-10:]:  # Last 10 reports
        with open(report, 'r') as f:
            data = json.load(f)
            if 'learning_metrics' in data:
                generations.append(data['learning_metrics'].get('generation', 0))
                learning_rates.append(data['learning_metrics'].get('learning_rate', 0))
                improvement_rates.append(data['learning_metrics'].get('improvement_rate', 0))
    
    # Print as ASCII chart since we can't guarantee matplotlib
    print("\n📈 LEARNING TREND VISUALIZATION")
    print("=" * 50)
    print("Gen | Learning Rate | Improvement Rate")
    print("-" * 50)
    for i in range(len(generations)):
        gen = generations[i]
        lr = learning_rates[i]
        ir = improvement_rates[i]
        lr_bar = "█" * int(lr * 20)
        ir_bar = "█" * int(ir * 20)
        print(f"{gen:3} | {lr_bar:<20} {lr:.2f} | {ir_bar:<20} {ir:.2f}")
    
    # Save to file
    output = "static/img/learning_trend.txt"
    Path("static/img").mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(f"Learning Trend Data - {len(generations)} generations\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]},{learning_rates[i]},{improvement_rates[i]}\n")
    print(f"\n✅ Trend data saved to {output}")

if __name__ == "__main__":
    plot_learning_curve()
