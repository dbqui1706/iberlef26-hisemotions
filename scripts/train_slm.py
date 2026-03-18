import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="Train SLM for HISEMOTIONS")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loaded config for experiment: {config['experiment_name']}")
    # Full implementation will wire up the HuggingFace Trainer here.

if __name__ == "__main__":
    main()
