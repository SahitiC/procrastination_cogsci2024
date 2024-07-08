# Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task

This repository contains code and data for [Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task](https://escholarship.org/uc/item/2mg517js) accepted at the Cognitive Science Society 2024 meeting. 

Authors: [Sahiti Chebolu](https://www.kyb.tuebingen.mpg.de/person/107410/2549) and [Peter Dayan](https://www.mpg.de/12309357/biologische-kybernetik-dayan)

## Abstract
Procrastination is a universal phenomenon, with a significantproportion of the population reporting interference and evenharm from such delays.  Why do people put off tasks despitewhat are apparently their best intentions, and why do they de-liberately defer in the face of prospective failure? Past researchshows that procrastination is a heterogeneous construct withpossibly diverse causes. To grapple with the complexity of thetopic, we construct a taxonomy of different types of procrasti-nation and potential sources for each type.  We simulate com-pletion patterns from three broad model types: exponential orinconsistent temporal discounting, and waiting for interestingtasks;  and provide some preliminary evidence, through com-parisons with real-world data,  of the plausibility of multipletypes of, and pathways for, procrastination

## Description and usage

1. FollowUpStudymatrixDf_finalpaper.csv - data from [Zhang and Ma 2024](https://www.nature.com/articles/s41598-024-65110-4). Consists of data from 193 students in a pyshology course. For our paper, the columns of interest are 'delta progress' and 'cumulative progress' that contain data about how many hours of experiments each student did per day in the semester

2. plots/ - folder containing all plots as vector images and final figures in the paper

3. data_clusters.py - first run this script for preprocessing (remove dropped-out subjects) and k-means clustering (clustered data is in data_relevant_clustered.csv), plotting patterns of working in clusters (as in Fig 1 of paper)

4. data_relevant_clustered.csv - contains data with cluster labels

modules containing some helper functions for further steps:
5. mdp_algms.py - functions for algorithms that find the optimal policy in MDPs, based on dynamic programming 

6. task_structure.py - functions for constructing reward/ effort functions based on various reward schedules and convex/ linear cost functions
and also tranisiton functions based on the transition structure in the different models

7. plotter.py - code for plotting average and example trajectories reported in the paper

8. constants.py - define few shared constants over all the models (states, actions, horizon, effort, shirk reward)

9. compute_distance.py - functions to compute (Euclidean) distance between simulated trajectories from models and data clusters: gives an idea of how well a model 
(with a specific parameter configuration) 'fits' the data cluster

Next run the following scripts (10-13) to implement each model type from the paper; they call modules 1-4 for defining task structure and model params, solving MDP and plotting; output plots and save svgs to 'plots/vectors/' folder: \
(please note that no random seed was set, so example trajectories might differ from those in the paper but avg plots will be more or less the same)

10. basic_model.py - implements the model with delayed rewards and common exponential discount factor; reproduces plots in Figure 2 (A-D)

11. immediate_reward_model.py - implements model with immediate rewards; reproduces Figure 3 (A,B)

12. defection_model.py - implements model with different discount factors; reproduces Figure 4 (A,B)

13. no_commitment.py - implements model with uncertain interest rewards; reproduces Figure 5 (A,B) 

14. .gitignore - tell git to ignore some local files, please change this based on your local repo

15. requirements.txt - python packages required to run these files

16. CheboluDayan2024.pdf - final pdf of the paper

## Installation

1. clone repository \
   for https: `git clone https://github.com/SahitiC/procrastination_cogsci2024.git` or \
   for ssh: `git clone git@github.com:SahitiC/procrastination_cogsci2024.git`
2. create and activate python virtual environment using your favourite environment manager (pip, conda etc)
3. install packages in requirments.txt: \
   for pip: `pip install -r requirements.txt`\
   for conda: `conda config --add channels conda-forge` \
              `conda config --set channel-priority strict`
              `conda install --yes --file requirements.txt`

## Citation

If you found this code or paper helpful, please cite us as: Chebolu, S., & Dayan, P. (2024). Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task. Proceedings of the Annual Meeting of the Cognitive Science Society, 46. Retrieved from <https://escholarship.org/uc/item/2mg517js> 

<code>
 @article{chebolu2024optimal, 
  title={Optimal and sub-optimal temporal decisions can explain procrastination in a real-world task}, 
  author={Chebolu, Sahiti and Dayan, Peter}, 
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society}, 
  volume={46}, 
  year={2024} 
}
</code>

## Contact

For any questions or comments, please contact us at <sahiti.chebolu@tuebingen.mpg.de>

   


