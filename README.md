# Autonomous-Car-UE5
This project was created as a part of my bachelor thesis on Creation of computer game and its solution using artificial intelligence. You can read more about how this project works in the paper attached in the repository.

## How to install
Download needed software:
1. Download Unreal Engine (project is based on version 5.3).
2. Download PyCharm by Jetbrains.

Setup software:
1. When you import the project, the Unreal Engine asks to install *TCPSocketConnection*. Install it there or from Marketplace.
2. In the PyCharm setup the python version. (project is written on version 3.12)
3. In the PyCharm, download packages: *PyTorch*, *pandas*

How to run it:
1. In Unreal Engine open map called *MainMenuMap*. After you can run the game.
2. Open as PyCharm project the folder *PythonAPP*.

### Player Mode
There you can play the game if you want to.

### Watch AI
Before playing, run the python app called *AiDriver*. Wait for the console to output *listening on port..*. Then you can run the game and watch AI.

### Train AI
Before playing choose if you want to add data to existing (*received_messages.csv*) or make a own data set. It has to be named same.

Run in PyCharm app called *Training.py*. Then you can run the game mode Train AI.

Before watching AI with your train data. You need to run program *ModelTrainer.py*. It creates a model of neural network called *trained_model.pt*.
