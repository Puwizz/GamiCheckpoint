import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def generate_eda(sessions_dir, output_file):
    sessions = [d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]
    steps = [f"Step{i}" for i in range(1, 7)]
    
    data = {session: {step: 0 for step in steps} for session in sessions}
    
    for session in sessions:
        session_path = os.path.join(sessions_dir, session)
        for step in steps:
            step_path = os.path.join(session_path, step)
            if os.path.exists(step_path):
                count = len([f for f in os.listdir(step_path) if f.endswith(('.jpg', '.png'))])
                data[session][step] = count

    # Plotting
    x = np.arange(len(steps))
    width = 0.8 / len(sessions)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, session in enumerate(sessions):
        counts = [data[session][step] for step in steps]
        offset = (i - len(sessions)/2) * width + width/2
        bars = ax.bar(x + offset, counts, width, label=session)
        ax.bar_label(bars, padding=3, rotation=90)

    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Distribution by Session and Step')
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.legend(title="Sessions", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add target lines
    ax.axhline(y=35, color='r', linestyle='--', alpha=0.5, label='Min Target (35)')
    ax.axhline(y=50, color='g', linestyle='--', alpha=0.5, label='Max Target (50)')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] EDA plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Exploratory Data Analysis (EDA) plot for the dataset.")
    parser.add_argument("--sessions-dir", default="data/sessions", help="Path to sessions directory")
    parser.add_argument("--output", default="outputs/plots/dataset_distribution.png", help="Output plot path")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sessions_path = os.path.join(base_dir, args.sessions_dir)
    output_path = os.path.join(base_dir, args.output)

    if not os.path.exists(sessions_path):
        print(f"[ERROR] Directory not found: {sessions_path}")
        return

    generate_eda(sessions_path, output_path)

if __name__ == "__main__":
    main()
