# Paper plan and instructions
This project is for https://newinml.github.io/NewInML2026NeurIPS/ for now.

We want to build an organelle-segmentation working of the cellmap data https://github.com/janelia-cellmap/cellmap-segmentation-challenge.

We currently have implemented 4 different versions to test out how each model performs on the cellmap data.

All code will be deployed in a 2x H200 system. Hence paths must correspond to that machine under /mnt/graid/codebases_other/cellmap-segmentation-challenge/saleem/organellenet

Data currently resides here: /mnt/graid/codebases_other/cellmap-segmentation-challenge/data

## Goal:
Beat the currently leaderboard front runner on this segmentation task. Submit it to NewInML 2026 NeurIPS. Even if we cannot beat, we must design a `Novel and Principled` method that can be published at the conference.

## ASK:
- First check if the code for deployment of training and inference of these models sounds?
- Are we using train/val/test splits?
- Are we using early stopping?
- Are we using balanced batches?
- Are we using appropriate loss functions?
- Are we using appropriate metrics?
- Are we using right amount of data augmentation?
- I want to train all 4 models in parallel on my cards, can we do that?
- All ckpts, logs, plots for metrics being stored in their own self contained `runs/` folders. The folders should have memorable names like `baseline-unet-earlystopping_{otherparameters}`. The folders should have all the information about that run.
- Allow config driven execution of all models. So basically yaml files that can specify all the parameters for training and inference.
- Make changes and write a commands.md file to me to run the models in my gpu server.
- Also write a changelog for changes made to the existing code.
- Make the codebase modular and do not pile everything into the same folder. Rearrange where appropriate. Ensure you include sys.path commands in the files, such that we can import methods from other scripts.

## Next ASK:
- Plan a set of ablation tests.
- We might need to explore a `novel` direction that goes beyond trying models? Like checking if a vit could work? Check if we can add attention mechanisms to the unets?
- Write an implementation plan of the new methods in an implementation.md. Be concise, but tell why we could explore that direction, what it adds, and why is it considered novel. Do not copy paste existing ideas. 

