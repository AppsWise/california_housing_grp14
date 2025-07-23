# Build and Deploy Instructions

This document provides instructions on how to build and deploy the Docker image for this project.

## Prerequisites

- Docker must be installed and running on your system.

## Build and Run Scripts

### For Linux and macOS

A `build.sh` script is provided to automate the build and deployment process.

#### Instructions

1. **Make the script executable:**
   ```bash
   chmod +x build.sh
   ```

2. **Run the script:**
   ```bash
   ./build.sh
   ```

### For Windows

A `build.ps1` script is provided for deploying on Windows.

#### Instructions

1. **Open PowerShell:**
   Open a PowerShell terminal.

2. **Run the script:**
   ```powershell
   .\build.ps1
   ```

This will build the Docker image with the tag `mlops-assignment` and run it in a container, mapping port 5000 of the container to port 5000 on your local machine.