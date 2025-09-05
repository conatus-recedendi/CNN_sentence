#!/usr/bin/env python
"""
Simple runner script for CNN sentence classification with PyTorch
GPU support and automatic backward pass
"""

import os
import sys
import subprocess


def run_data_processing():
    """Run data processing if needed"""
    if not os.path.exists("mr.p"):
        print("Processing data...")
        subprocess.run([sys.executable, "process_data_pytorch.py"])
    else:
        print("Data already processed (mr.p exists)")


def run_training():
    """Run training with different configurations"""
    configurations = [
        ("CNN-rand", ["-nonstatic", "-rand"]),
        ("CNN-static", ["-static", "-word2vec"]),
        ("CNN-non-static", ["-nonstatic", "-word2vec"]),
    ]

    for name, args in configurations:
        print(f"\n{'='*60}")
        print(f"Running {name}")
        print(f"{'='*60}")

        cmd = (
            [sys.executable, "conv_net_sentence_pytorch.py", "mr.p"]
            + args
            + [
                "--single-fold",
                "0",  # Run single fold for quick testing
                "--epochs",
                "3",  # Reduce epochs for quick testing
            ]
        )

        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)


def main():
    print("CNN Sentence Classification with PyTorch")
    print("Features: GPU support, automatic backward pass")
    print()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run data processing
    run_data_processing()

    # Run training
    run_training()


if __name__ == "__main__":
    main()
