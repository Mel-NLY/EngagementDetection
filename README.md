# EngagementDetection Model
Goal of this project was to create a model to detect the Engagement level of students during lesson time. There had been a few months of research before embarking on the development of the model. Resulting in state-of-the-art algorithms like CNN-LSTM and Bi-directional models to be focused on.

After the research, theories that had been proposed in the model had to be tested out. For clarification, only the fundamentals had been managed to be tested, such as the creation of the CNN-LSTM model, implementing a Learning Rate Test, using the SWA optimizer, and the bugs faced/solutions during the experimental development of the model. CNNLSTMDraft3.py and model_4.h5 had given the best results of 55.54% Test Accuracy and 0.832 Test Loss.

Originally, the model had been run on laptops with 16GB RAM without GPU, then on NP's desktop monitor   with GPU and Google Cloud with 52GB RAM without CPU. All of them were 	unable to handle the large dataset   of video frames that DAiSEE had. Hence, this could explain why the derived 55.54% Test Accuracy had been   lower than the expected baseline 	of 57.1%.
