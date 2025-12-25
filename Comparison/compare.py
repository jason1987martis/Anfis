import os
import sys
import glob
import re
from pathlib import Path

def parse_log(log_path):
    """
    Parses a training log file to extract training duration and epoch count.
    """
    data = {
        "duration": None,
        "epochs": 0,
        "device": "unknown"
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    # Extract Duration
    m_duration = re.search(r"Duration: (\d+\.\d+)s", content)
    if m_duration:
        data["duration"] = float(m_duration.group(1))
        
    # Extract Epoch Count
    # We count lines starting with "Epoch "
    epochs = len(re.findall(r"^Epoch \d+", content, re.MULTILINE))
    data["epochs"] = epochs
        
    if "gpu" in str(log_path).lower():
        data["device"] = "GPU"
    else:
        data["device"] = "CPU"
        
    return data

def main():
    base_dir = Path(__file__).resolve().parent.parent
    gpu_logs = list((base_dir / "gpu_version" / "logs").glob("*.log"))
    cpu_logs = list((base_dir / "cpu_version" / "logs").glob("*.log"))
    
    all_logs = gpu_logs + cpu_logs
    
    results = []
    
    for log in all_logs:
        info = parse_log(log)
        info["filename"] = log.name
        results.append(info)
        
    # Sort by device for cleaner output
    results.sort(key=lambda x: x["device"])
    
    print("\n" + "="*60)
    print(f"{'DEVICE':<10} | {'EPOCHS':<10} | {'DURATION (s)':<15} | {'FILENAME'}")
    print("-" * 60)
    
    for res in results:
        duration_str = f"{res['duration']:.2f}" if res['duration'] is not None else "N/A"
        print(f"{res['device']:<10} | {res['epochs']:<10} | {duration_str:<15} | {res['filename']}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
