import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description='MLOps Pipeline Runner')
    parser.add_argument('--step', type=str, required=True, 
                        choices=['generate', 'train', 'tune', 'app', 'all'], 
                        help='Pipeline step to run: generate, train, tune, app, or all')
    
    args = parser.parse_args()
    
    if args.step == 'generate' or args.step == 'all':
        print("\n=== Generating Data ===")
        from src.data_utils import generate_data
        generate_data()
    
    if args.step == 'train' or args.step == 'all':
        print("\n=== Training Model ===")
        from src.train import train
        train()
    
    if args.step == 'tune' or args.step == 'all':
        print("\n=== Tuning Hyperparameters ===")
        from src.tune import tune
        tune()
    
    if args.step == 'app' or args.step == 'all':
        print("\n=== Starting Flask App ===")
        os.chdir('app')
        os.system("python app.py")

if __name__ == '__main__':
    main() 