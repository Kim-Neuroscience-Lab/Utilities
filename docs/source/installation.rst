Installation
============

This guide will help you set up the Kim Lab Utilities package on your system. The package is designed to work across all major operating systems (Linux, Windows, macOS).

System Requirements
-----------------

- Python 3.12 or higher
- Operating System:
    * Linux (any modern distribution)
    * Windows 10/11
    * macOS 10.15 or higher
- Hardware:
    * CPU: Any x86_64 or ARM64 processor
    * RAM: 8GB minimum (16GB recommended)
    * Storage: 1GB free disk space
- GPU Support (optional):
    * NVIDIA GPU with CUDA support
    * Apple Silicon with MPS support
    * AMD ROCm support (experimental)

Dependencies
-----------

Core Dependencies:
    - NumPy >= 2.2.0
    - Pandas >= 2.0.0
    - SciPy >= 1.12.0
    - PyTorch >= 2.2.0 (for GPU acceleration)

Development Dependencies:
    - pytest
    - sphinx
    - sphinx-rtd-theme
    - black
    - flake8

Installation Steps
----------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/Kim-Neuroscience-Lab/Utilities.git
    cd Utilities

2. Set Up Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

We strongly recommend using a virtual environment. Choose the appropriate commands for your operating system:

Linux/macOS:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate

Windows:

.. code-block:: batch

    python -m venv venv
    venv\Scripts\activate.bat

3. Install Dependencies
~~~~~~~~~~~~~~~~~~~~

Basic Installation:

.. code-block:: bash

    pip install -r requirements.txt

For Development:

.. code-block:: bash

    pip install -r requirements-dev.txt

4. GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate installation based on your GPU:

NVIDIA GPUs (Linux/Windows):

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Apple Silicon (macOS):

.. code-block:: bash

    # MPS support is included in the base PyTorch installation
    pip install torch torchvision

AMD GPUs (Linux):

.. code-block:: bash

    # ROCm support is experimental
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

5. Verify Installation
~~~~~~~~~~~~~~~~~~~

Run the test suite to verify the installation:

.. code-block:: bash

    # Linux/macOS
    pytest tests/

.. code-block:: batch

    :: Windows
    python -m pytest tests/

Environment Variables
------------------

Set these environment variables if needed. The method varies by operating system:

Linux/macOS:

.. code-block:: bash

    # Add the project root to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/path/to/Utilities

    # For GPU memory management (optional)
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

Windows:

.. code-block:: batch

    :: Add the project root to PYTHONPATH
    set PYTHONPATH=%PYTHONPATH%;C:\path\to\Utilities

    :: For GPU memory management (optional)
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. GPU Not Detected
******************

For NVIDIA GPUs:

.. code-block:: bash

    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"

    # Check CUDA version
    nvidia-smi
    python -c "import torch; print(torch.version.cuda)"

For Apple Silicon:

.. code-block:: bash

    # Check MPS availability
    python -c "import torch; print(torch.backends.mps.is_available())"

For AMD GPUs:

.. code-block:: bash

    # Check ROCm availability
    python -c "import torch; print(torch.version.hip)"

2. Import Errors
*************

- Ensure PYTHONPATH is set correctly for your OS
- Verify all dependencies are installed:

Linux/macOS:

.. code-block:: bash

    pip list | grep -E "numpy|pandas|scipy|torch"

Windows:

.. code-block:: batch

    pip list | findstr /I "numpy pandas scipy torch"

3. Memory Issues
*************

- Reduce batch size using ``--batch-size`` argument
- Adjust GPU memory allocation:

.. code-block:: bash

    # Linux/macOS
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

.. code-block:: batch

    :: Windows
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

Getting Help
----------

If you encounter any issues:

1. Check the :doc:`FAQ <faq>` section
2. Search existing GitHub issues
3. Create a new issue with:
   - Operating system and version
   - Python version
   - GPU details (if applicable)
   - Error message
   - Steps to reproduce
   - Expected vs actual behavior 