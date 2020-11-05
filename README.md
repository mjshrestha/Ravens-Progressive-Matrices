# Ravens Progressive Matrices Test
The agent solves the non-verbal intelligence test similar to [Raven's Progressive Matrices -RPM](https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices) using visual approach with pixel intensity values as the comparison metrics. The problems include 4 sets from B to E. Set B has 12 - 2x2 matrix problems with 6 answer choices and sets C, D, and E have 12 - 3x3 matrix problems with 8 answer choices and the problems complexity increase in ascending order.
The agent has the accuracy of 84.3% with breakdown as follows:

| Problem Set              |  Correct | Incorrect | Skipped |
|--------------------------|----------|-----------|---------|
| Basic Problems B         |    12    |     0     |     0   |
| Test Problems B (Unseen) |    10    |     2     |     0   |
| Basic Problems C         |    12    |     0     |     0   |
| Test Problems C (Unseen) |     9    |     3     |     0   |
| Basic Problems D         |     9    |     3     |     0   |
| Test Problems D (Unseen) |    10    |     2     |     0   |
| Basic Problems E         |    12    |     0     |     0   |
| Test Problems E (Unseen) |     7    |     5     |     0   |
| **Total (96)**           |  **81**  |  **15**   |  **0**  |

## Required Steps for installation
1. Install Pycharm community edition
2. Download and install Anaconda3
3. Open the Anaconda3 prompt
4. Run the commands in terminal:
```
conda create -n RPM python=3.4.3 anaconda pillow=4.0.0 numpy=1.10.4
conda install requests==2.14.2 future==0.16.0
python -m pip install numpy==1.12.0
```
5. Activate the environment:
```conda activate RPM```
6. Open Pycharm and use import existing project pointing to RPM

***Note:*** For full path of env, use the below command in terminal:
```conda info --envs```


## Instructions to run:
1. Clone the repository.
2. Run RavensProject.py
3. The correct, incorrect and skipped problems summary can be seen in **SetResults.csv** and the individual problem details can be found in **ProblemResults.csv** in the folder.

