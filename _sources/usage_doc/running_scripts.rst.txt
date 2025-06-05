Running Python Scripts with Blender
=====================================

Using the Blender Python Interpreter
---------------------------------------

To run a script using the Blender Python interpreter, you need to use the Blender executable and pass the script as an argument.

Assume the Blender executable is located at:

.. code-block:: console

    ~/blender/blender

And your project is structured as follows:

.. code-block:: console

    .
    ├── blender
    │   └── blender
    ├── Documents
    │   └── My_project
    │       └── scripts
    │           └── script1.py

You are currently in the ``~/Documents/My_project`` directory and want to run the script ``script1.py``, located at:

- Absolute path: ``~/Documents/My_project/scripts/script1.py``
- Relative path: ``scripts/script1.py``

To run the script using the Blender Python interpreter and launch the Blender GUI, use:

.. code-block:: console

    ~/blender/blender --python scripts/script1.py

If you want to run the script without opening the GUI (in background mode), use:

.. code-block:: console

    ~/blender/blender --background --python scripts/script1.py

This will execute the script using Blender's Python interpreter without showing the Blender interface.


Using the Blender GUI Scripting Console
---------------------------------------

You can also run scripts directly in the Blender GUI using the scripting console. 

Assume the Blender executable is located at:

.. code-block:: console

    ~/blender/blender

To open the Blender GUI, simply run the Blender executable:

.. code-block:: console

    ~/blender/blender

On the top menu, navigate to the "Scripting" tab. Here, you can create a new text file or open an existing script file.
You can write your script directly in the text editor or load an existing script by clicking on "Open" and selecting your script file.
Then simply click on the "Run Script" button to execute the script.

All the ``print`` statements in your script will output to the open terminal where you launched Blender, allowing you to see the results of your script execution.


