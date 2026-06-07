# Bảng So sánh Metrics (NN vs PCN)

## Bảng 1: Model Bias và Tổng quan Metrics

| Model | Mode | Accuracy | F1 | LogLoss | Brier | Class1_Rate | Elo_Reliance | Upset_Acc | Rank_Reliance |
|---|---|---|---|---|---|---|---|---|---|
| NN | static | 65.71 | 0.6458 | 0.6026 | 0.2086 | 49.75 | 86.72 | 20.66 | 81.01 |
| NN | finetune | 65.88 | 0.6518 | 0.6033 | 0.2092 | 50.92 | 87.23 | 20.19 | 81.18 |
| NN | retrain | 66.22 | 0.6576 | 0.604 | 0.2095 | 51.6 | 86.89 | 21.13 | 82.18 |
| NN | online | 66.55 | 0.6527 | 0.5979 | 0.2067 | 49.24 | 85.88 | 23.0 | 79.83 |
| NN | ultimate_streaming | 67.56 | 0.662 | 0.5999 | 0.2076 | 48.91 | 87.56 | 22.07 | 81.85 |
| PCN | static | 63.7 | 0.6015 | 1.0546 | 0.2439 | 44.03 | 74.96 | 34.27 | 76.3 |
| PCN | finetune | 63.36 | 0.6007 | 1.0328 | 0.2439 | 44.71 | 73.61 | 35.68 | 75.63 |
| PCN | retrain | 63.19 | 0.5967 | 1.0537 | 0.2444 | 44.2 | 74.12 | 34.74 | 75.8 |
| PCN | online | 63.19 | 0.5982 | 1.053 | 0.2438 | 44.54 | 74.45 | 34.27 | 76.13 |
| PCN | ultimate_streaming | 63.19 | 0.5982 | 1.053 | 0.2439 | 44.54 | 74.45 | 34.27 | 76.13 |


## Bảng 2: Player Metrics (Độ chính xác theo người chơi)

| Player | Mode | Matches | NN_Acc | NN_Prec | NN_Rec | PCN_Acc | PCN_Prec | PCN_Rec |
|---|---|---|---|---|---|---|---|---|
| Alexander Zverev | static | 26 | 92.31 | 90.91 | 90.91 | 84.62 | 88.89 | 72.73 |
| Alexander Zverev | finetune | 26 | 92.31 | 90.91 | 90.91 | 84.62 | 88.89 | 72.73 |
| Alexander Zverev | retrain | 26 | 92.31 | 90.91 | 90.91 | 80.77 | 87.5 | 63.64 |
| Alexander Zverev | online | 26 | 92.31 | 90.91 | 90.91 | 84.62 | 88.89 | 72.73 |
| Alexander Zverev | ultimate_streaming | 26 | 92.31 | 90.91 | 90.91 | 84.62 | 88.89 | 72.73 |
| Novak Djokovic | static | 4 | 50.0 | 50.0 | 100.0 | 25.0 | 33.33 | 50.0 |
| Novak Djokovic | finetune | 4 | 50.0 | 50.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| Novak Djokovic | retrain | 4 | 50.0 | 50.0 | 100.0 | 25.0 | 33.33 | 50.0 |
| Novak Djokovic | online | 4 | 50.0 | 50.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| Novak Djokovic | ultimate_streaming | 4 | 50.0 | 50.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| Daniil Medvedev | static | 17 | 70.59 | 71.43 | 62.5 | 64.71 | 66.67 | 50.0 |
| Daniil Medvedev | finetune | 17 | 70.59 | 71.43 | 62.5 | 64.71 | 66.67 | 50.0 |
| Daniil Medvedev | retrain | 17 | 76.47 | 75.0 | 75.0 | 64.71 | 66.67 | 50.0 |
| Daniil Medvedev | online | 17 | 70.59 | 71.43 | 62.5 | 64.71 | 66.67 | 50.0 |
| Daniil Medvedev | ultimate_streaming | 17 | 70.59 | 71.43 | 62.5 | 64.71 | 66.67 | 50.0 |
| Grigor Dimitrov | static | 4 | 50.0 | 50.0 | 50.0 | 75.0 | 100.0 | 50.0 |
| Grigor Dimitrov | finetune | 4 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| Grigor Dimitrov | retrain | 4 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| Grigor Dimitrov | online | 4 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| Grigor Dimitrov | ultimate_streaming | 4 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| Andrey Rublev | static | 13 | 69.23 | 62.5 | 83.33 | 76.92 | 71.43 | 83.33 |
| Andrey Rublev | finetune | 13 | 76.92 | 66.67 | 100.0 | 76.92 | 71.43 | 83.33 |
| Andrey Rublev | retrain | 13 | 76.92 | 66.67 | 100.0 | 76.92 | 71.43 | 83.33 |
| Andrey Rublev | online | 13 | 69.23 | 62.5 | 83.33 | 76.92 | 71.43 | 83.33 |
| Andrey Rublev | ultimate_streaming | 13 | 76.92 | 66.67 | 100.0 | 76.92 | 71.43 | 83.33 |
| Taylor Fritz | static | 6 | 50.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| Taylor Fritz | finetune | 6 | 50.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| Taylor Fritz | retrain | 6 | 50.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| Taylor Fritz | online | 6 | 50.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| Taylor Fritz | ultimate_streaming | 6 | 50.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| Stefanos Tsitsipas | static | 12 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 33.33 |
| Stefanos Tsitsipas | finetune | 12 | 58.33 | 60.0 | 50.0 | 50.0 | 50.0 | 33.33 |
| Stefanos Tsitsipas | retrain | 12 | 58.33 | 60.0 | 50.0 | 50.0 | 50.0 | 33.33 |
| Stefanos Tsitsipas | online | 12 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 33.33 |
| Stefanos Tsitsipas | ultimate_streaming | 12 | 58.33 | 60.0 | 50.0 | 50.0 | 50.0 | 33.33 |
| Karen Khachanov | static | 12 | 66.67 | 83.33 | 62.5 | 33.33 | 50.0 | 25.0 |
| Karen Khachanov | finetune | 12 | 66.67 | 83.33 | 62.5 | 33.33 | 50.0 | 25.0 |
| Karen Khachanov | retrain | 12 | 66.67 | 83.33 | 62.5 | 33.33 | 50.0 | 25.0 |
| Karen Khachanov | online | 12 | 58.33 | 80.0 | 50.0 | 33.33 | 50.0 | 25.0 |
| Karen Khachanov | ultimate_streaming | 12 | 66.67 | 83.33 | 62.5 | 33.33 | 50.0 | 25.0 |
| Alex De Minaur | static | 14 | 42.86 | 42.86 | 42.86 | 57.14 | 60.0 | 42.86 |
| Alex De Minaur | finetune | 14 | 50.0 | 50.0 | 42.86 | 57.14 | 60.0 | 42.86 |
| Alex De Minaur | retrain | 14 | 42.86 | 42.86 | 42.86 | 57.14 | 60.0 | 42.86 |
| Alex De Minaur | online | 14 | 42.86 | 42.86 | 42.86 | 57.14 | 60.0 | 42.86 |
| Alex De Minaur | ultimate_streaming | 14 | 50.0 | 50.0 | 42.86 | 57.14 | 60.0 | 42.86 |
| Marin Cilic | static | 11 | 45.45 | 20.0 | 33.33 | 45.45 | 20.0 | 33.33 |
| Marin Cilic | finetune | 11 | 54.55 | 25.0 | 33.33 | 45.45 | 20.0 | 33.33 |
| Marin Cilic | retrain | 11 | 54.55 | 25.0 | 33.33 | 45.45 | 20.0 | 33.33 |
| Marin Cilic | online | 11 | 54.55 | 25.0 | 33.33 | 45.45 | 20.0 | 33.33 |
| Marin Cilic | ultimate_streaming | 11 | 54.55 | 25.0 | 33.33 | 45.45 | 20.0 | 33.33 |
| Luca Nardi | static | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Luca Nardi | finetune | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Luca Nardi | retrain | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Luca Nardi | online | 1 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Luca Nardi | ultimate_streaming | 1 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Adam Walton | static | 5 | 80.0 | 50.0 | 100.0 | 80.0 | 50.0 | 100.0 |
| Adam Walton | finetune | 5 | 80.0 | 50.0 | 100.0 | 80.0 | 50.0 | 100.0 |
| Adam Walton | retrain | 5 | 80.0 | 50.0 | 100.0 | 80.0 | 50.0 | 100.0 |
| Adam Walton | online | 5 | 80.0 | 50.0 | 100.0 | 80.0 | 50.0 | 100.0 |
| Adam Walton | ultimate_streaming | 5 | 80.0 | 50.0 | 100.0 | 80.0 | 50.0 | 100.0 |
| Sumit Nagal | static | 1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Sumit Nagal | finetune | 1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Sumit Nagal | retrain | 1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Sumit Nagal | online | 1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Sumit Nagal | ultimate_streaming | 1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Emilio Nava | static | 6 | 66.67 | 100.0 | 60.0 | 66.67 | 100.0 | 60.0 |
| Emilio Nava | finetune | 6 | 66.67 | 100.0 | 60.0 | 66.67 | 100.0 | 60.0 |
| Emilio Nava | retrain | 6 | 66.67 | 100.0 | 60.0 | 66.67 | 100.0 | 60.0 |
| Emilio Nava | online | 6 | 66.67 | 100.0 | 60.0 | 66.67 | 100.0 | 60.0 |
| Emilio Nava | ultimate_streaming | 6 | 66.67 | 100.0 | 60.0 | 66.67 | 100.0 | 60.0 |
| Zachary Svajda | static | 5 | 20.0 | 50.0 | 25.0 | 20.0 | 50.0 | 25.0 |
| Zachary Svajda | finetune | 5 | 20.0 | 50.0 | 25.0 | 20.0 | 50.0 | 25.0 |
| Zachary Svajda | retrain | 5 | 20.0 | 50.0 | 25.0 | 20.0 | 50.0 | 25.0 |
| Zachary Svajda | online | 5 | 20.0 | 50.0 | 25.0 | 20.0 | 50.0 | 25.0 |
| Zachary Svajda | ultimate_streaming | 5 | 20.0 | 50.0 | 25.0 | 20.0 | 50.0 | 25.0 |
| Nikoloz Basilashvili | static | 7 | 57.14 | 60.0 | 75.0 | 57.14 | 60.0 | 75.0 |
| Nikoloz Basilashvili | finetune | 7 | 57.14 | 60.0 | 75.0 | 57.14 | 60.0 | 75.0 |
| Nikoloz Basilashvili | retrain | 7 | 57.14 | 60.0 | 75.0 | 57.14 | 60.0 | 75.0 |
| Nikoloz Basilashvili | online | 7 | 57.14 | 60.0 | 75.0 | 57.14 | 60.0 | 75.0 |
| Nikoloz Basilashvili | ultimate_streaming | 7 | 57.14 | 60.0 | 75.0 | 57.14 | 60.0 | 75.0 |
| Joao Fonseca | static | 14 | 85.71 | 71.43 | 100.0 | 78.57 | 66.67 | 80.0 |
| Joao Fonseca | finetune | 14 | 71.43 | 55.56 | 100.0 | 78.57 | 66.67 | 80.0 |
| Joao Fonseca | retrain | 14 | 71.43 | 55.56 | 100.0 | 71.43 | 60.0 | 60.0 |
| Joao Fonseca | online | 14 | 85.71 | 71.43 | 100.0 | 78.57 | 66.67 | 80.0 |
| Joao Fonseca | ultimate_streaming | 14 | 85.71 | 71.43 | 100.0 | 78.57 | 66.67 | 80.0 |
| Learner Tien | static | 14 | 57.14 | 25.0 | 25.0 | 57.14 | 33.33 | 50.0 |
| Learner Tien | finetune | 14 | 64.29 | 33.33 | 25.0 | 57.14 | 33.33 | 50.0 |
| Learner Tien | retrain | 14 | 64.29 | 33.33 | 25.0 | 57.14 | 33.33 | 50.0 |
| Learner Tien | online | 14 | 57.14 | 25.0 | 25.0 | 57.14 | 33.33 | 50.0 |
| Learner Tien | ultimate_streaming | 14 | 64.29 | 33.33 | 25.0 | 57.14 | 33.33 | 50.0 |
| Jannik Sinner | static | 28 | 96.43 | 93.75 | 100.0 | 96.43 | 93.75 | 100.0 |
| Jannik Sinner | finetune | 28 | 96.43 | 93.75 | 100.0 | 92.86 | 88.24 | 100.0 |
| Jannik Sinner | retrain | 28 | 96.43 | 93.75 | 100.0 | 92.86 | 88.24 | 100.0 |
| Jannik Sinner | online | 28 | 96.43 | 93.75 | 100.0 | 92.86 | 88.24 | 100.0 |
| Jannik Sinner | ultimate_streaming | 28 | 96.43 | 93.75 | 100.0 | 92.86 | 88.24 | 100.0 |
| Carlos Alcaraz | static | 13 | 76.92 | 83.33 | 71.43 | 76.92 | 83.33 | 71.43 |
| Carlos Alcaraz | finetune | 13 | 76.92 | 83.33 | 71.43 | 76.92 | 83.33 | 71.43 |
| Carlos Alcaraz | retrain | 13 | 76.92 | 83.33 | 71.43 | 76.92 | 83.33 | 71.43 |
| Carlos Alcaraz | online | 13 | 76.92 | 83.33 | 71.43 | 76.92 | 83.33 | 71.43 |
| Carlos Alcaraz | ultimate_streaming | 13 | 76.92 | 83.33 | 71.43 | 76.92 | 83.33 | 71.43 |
| Stan Wawrinka | static | 4 | 75.0 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| Stan Wawrinka | finetune | 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Stan Wawrinka | retrain | 4 | 75.0 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| Stan Wawrinka | online | 4 | 75.0 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| Stan Wawrinka | ultimate_streaming | 4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
