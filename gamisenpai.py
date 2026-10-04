import sys
import argparse
import subprocess
import os

def run_script(script_path, args):
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found.")
        sys.exit(1)
    
    cmd = [sys.executable, script_path] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)

def main():
    parser = argparse.ArgumentParser(
        description="GamiSenpai Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  data          Dataset curation and preparation tools
  train         Model training tools
  eval          Evaluation and benchmarking tools
  utils         Utility and visualization scripts

Example:
  python gamisenpai.py data prepare_v3_bg_dataset --help
        """
    )
    
    parser.add_argument("module", choices=["data", "train", "eval", "utils"], help="The module to run")
    parser.add_argument("script", help="The name of the script to run (without .py)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the script")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Map module to folder
    folder_map = {
        "data": "dataset",
        "train": "training",
        "eval": "evaluation",
        "utils": "utils"
    }
    
    folder = folder_map[args.module]
    script_file = f"{args.script}.py"
    if not script_file.endswith(".py"):
        script_file += ".py"
        
    script_path = os.path.join("scripts", folder, script_file)
    run_script(script_path, args.args)

if __name__ == "__main__":
    main()
