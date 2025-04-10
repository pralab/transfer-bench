# Instruction README for Using the `trbench` Script

The `trbench` script is a command-line tool designed for managing and running tasks related to benchmarking. This script is intended for **internal usage only** and requires specific prerequisites to function correctly.

## Prerequisites

1. **Weights and Biases Account**:  
    You must have an active [Weights and Biases](https://wandb.ai/) account. This is required for logging and tracking the benchmarking experiments. Follow the instruction at your first run

2. **Transfer-Bench Team Membership**:  
    Ensure that you are part of the **Transfer-Bench team** on Weights and Biases. Access to this team is mandatory to use the script effectively.

## Usage

The `trbench` script provides two main functionalities: retrieving information about runs and executing specific tasks. Below is a detailed explanation of how to use the script.

### 1. Display Information About Runs

To display information about the runs, use the `info` subcommand. Below is an example:

```bash
./trbench info --help
```

This command retrieves and displays a list of all available runs.
