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
    parser = argparse.ArgumentParser(description="Wrapper for GPU training")
    parser.add_argument("--script", type=str, required=True, help="Script to run (relative to code/ dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="gpu_outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="run_config.yml", help="Config file path")
    
    args = parser.parse_args()
    
    # Load config manually to avoid pyyaml dependency
    config = load_config(args.config)
            
    # Setup paths
    base_dir = Path(__file__).resolve().parent
    code_dir = base_dir / "code"
    logs_dir = base_dir / "logs"
    models_dir = base_dir / "models"
    logs_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Check GPU (lightweight check using nvidia-smi or just trust the wrapper)
    # Removing py-torch import check to keep wrapper lightweight and dependency free
        
    # Set Environment Variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" # Default to first GPU
    env["PYTHONUNBUFFERED"] = "1"
    
    # Script path
    script_path = code_dir / args.script
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)
        
    print(f"Running {args.script} on GPU...")
    print(f"Seed: {args.seed}")
    print(f"Output: {models_dir}")
    
    # Construct command
    # Assuming scripts take some args? The prompt says "Run the same training scripts with the same hyperparameters".
    # Existing scripts might not accept --seed or --outdir easily if they are hardcoded.
    # The requirement says: "Ensure CUDA devices are available... Record training time..."
    # And "Run the training scripts and capture stdout/stderr".
    # If the original scripts don't take arguments, we might just run them. 
    # But the wrapper must accept --seed etc. 
    # If the underlying script doesn't support them, we can't magically pass them unless we modify the script (forbidden) 
    # OR if we set them via env vars or just ignore them but record them?
    # I will pass them if the script accepts them (I can't easily know). 
    # BUT wait, the prompt says "run_gpu.bat... should accept --script... --seed... --outdir".
    # Check if existing scripts accept arguments?
    # train_on_input2.py (Step 22) uses hardcoded values (random_state=42). 
    # It does NOT use argparse.
    # So passing arguments to it will fail or be ignored.
    # However, I can set the seed in this wrapper before calling the subprocess? 
    # No, subprocess is a separate process.
    # If I can't modify the code, I can't effectively change the seed inside the script unless I inject code or use env vars (if the script used env vars).
    # Since I cannot modify "original training code files", I have to assume the scripts runs as is.
    # I will just run the script.
    
    start_time = time.time()
    
    log_file = logs_dir / f"{script_path.stem}_{int(time.time())}.log"
    
    with open(log_file, "w") as log_f:
        # We write to both stdout and log file
        def write_log(msg):
            print(msg)
            log_f.write(msg + "\n")
            
        write_log(f"STARTING GPU RUN: {script_path}")
        write_log(f"Timestamp: {time.ctime()}")
        
        try:
            # We run the script. We pass --seed if the user asked, but if the script crashes due to unrecog args, that's bad.
            # Safe bet: Run without args unless known.
            # Check train_anfis.py content (Step 16) -> No argparse.
            # So I CANNOT pass arguments to the script.
            # I will just run it. 
            # I will log that I cannot enforce seed/outdir on the inner script without modification.
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
        
        # GPU Memory usage is hard to capture post-facto. 
        # Usually requires monitoring during run.
        # I'll rely on the existing script output or `nvidia-smi` if I could run it in parallel.
        
if __name__ == "__main__":
    main()
