import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import shutil

def load_config(config_path):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                if ":" in line and not line.strip().startswith("#"):
                    key, val = line.split(":", 1)
                    config[key.strip()] = val.strip()
    return config

def main():
    parser = argparse.ArgumentParser(description="Wrapper for CPU training")
    parser.add_argument("--script", type=str, required=True, help="Script to run (relative to code/ dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="cpu_outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="run_config.yml", help="Config file path")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
            
    base_dir = Path(__file__).resolve().parent
    code_dir = base_dir / "code"
    logs_dir = base_dir / "logs"
    models_dir = base_dir / "models"
    logs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Set Environment Variables to DISABLE CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" 
    env["PYTHONUNBUFFERED"] = "1"
    
    script_path = code_dir / args.script
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)
        
    print(f"Running {args.script} on CPU...")
    
    start_time = time.time()
    log_file = logs_dir / f"{script_path.stem}_{int(time.time())}.log"
    
    with open(log_file, "w") as log_f:
        def write_log(msg):
            print(msg)
            log_f.write(msg + "\n")
            
        write_log(f"STARTING CPU RUN: {script_path}")
        write_log(f"Timestamp: {time.ctime()}")
        
        try:
            cmd = [sys.executable, str(script_path)]
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            while True:
                line = p.stdout.readline()
                if not line and p.poll() is not None:
                    break
                if line:
                    print(line, end='')
                    log_f.write(line)
                    
            ret_code = p.poll()
            if ret_code != 0:
                write_log(f"ERROR: Script failed with code {ret_code}")
                
        except Exception as e:
            write_log(f"EXCEPTION: {e}")
            
        end_time = time.time()
        duration = end_time - start_time
        write_log(f"FINISHED. Duration: {duration:.2f}s")

if __name__ == "__main__":
    main()
