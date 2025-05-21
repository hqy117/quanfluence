# Quanfluence QUBO Workflow Example

This guide explains how to create, upload, and execute QUBO using the Quanfluence platform.

## Workflow Overview

1. Create a QUBO problem file
2. Upload the QUBO file to the Quanfluence platform
3. Execute the uploaded QUBO problem
4. Re-run the QUBO problem without re-uploading

## Example Files

- `create_qubo.py`: Creates a QUBO file for a Max-Cut problem
- `upload_example.py`: Uploads and executes a QUBO file
- `run_example.py`: Executes a previously uploaded QUBO file
- `max_cut_triangle.qubo`: Example QUBO file (created by `create_qubo.py`)
- `bqm_example.py`: Additional example for working with Binary Quadratic Models

## Step 1: Create a QUBO Problem File

I create a  `create_qubo.py` file to generate a QUBO file (You can create your own code to create the qubo problem you want):

```bash
python create_qubo.py
```

This script creates a simple Max-Cut problem on a triangle graph and saves it as `max_cut_triangle.qubo`.

## Step 2: Upload and Execute the QUBO File

Use `upload_example.py` to upload your QUBO file to the Quanfluence platform and execute it:

```bash
python upload_example.py
```

**Important**: 
- Note the filename returned after uploading (it will be printed in the console). You'll need this filename for subsequent runs.
- `upload_example.py` should only be run once to upload a QUBO file or to update parameters.
- For multiple executions of the same problem, use `run_example.py` to avoid unnecessary uploads as following Step 3.
- The device ID and user credentials are preconfigured and should not be changed.

## Step 3: Re-run the QUBO Problem

For subsequent runs of your QUBO problem (without re-uploading), use `run_example.py`:

1. First, edit `run_example.py` to specify the qubo filename path you received from the upload step.

2. Then run the script:

```bash
python run_example.py
```

## Notes

For more detailed information, please refer to `README_quickstart.pdf`.