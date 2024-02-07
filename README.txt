

files:

1. mdp_algms.py - contains all the functions for algorithms that find the optimal policy in MDPs, based on dynamic programming

2. task_structure.py - contains functions for constructing reward/ effort functions based on various reward schedules and convex/ linear cost functions
and also tranisiton functions based on the transition structure in the different models

3. plotter.py - contains plotting code for:
 sausage plots which show mean trajectory (and std as shaded region) given trajectories
 specified number of example trajectories 

4. constants.py - define few shared constants over all the models (states, actions, horizon, effort, shirk reward)

5. FollowUpStudymatrixDf_finalpaper.csv - data from Zhang and Ma 2023

6. data_relevant_clustered.csv - k-means cluster labels for the data 

7. data_clusters.py - script for preprocessing (remove dropped-out subjects), clustering (output is file 6.), 
plotting patterns of working in clusters (as in Fig 1 of paper)

the following files (6-9) implement each model type from the paper; they call modules 1-4 for defining task structure and model params, solving MDP and plotting;
outputs plots and saves svgs to 'plots/vectors/' folder:
(please note that no random seed was set, so example trajectories might differ from those in the paper)

8. basic_model.py - implements the model with delayed rewards and common exponential discount factor; reproduces plots in Figure 2 (A-D)

9. immediate_reward_model.py - implements model with immediate (at threshold rewards); reproduces Figure 3 (A,B)

10. defection_model.py - implements model with different discount factors; reproduces Figure 4 (A,B)

11. no_commitment.py - implements model with uncertain interest rewards; reproduces Figure 5 (A,B)

12. .gitignore - tell git to ignore some local files, please change this based on your local repo

13. requirements.txt - python packages required to run these files

14. plots/ - folder containing svg and png files of plots/ figures in paper
