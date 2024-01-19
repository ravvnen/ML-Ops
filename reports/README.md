---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

`--- question 1 fill here ---`

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is _exhaustic_ which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

- [ ] Create a git repository
- [ ] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages
- [ ] Create the initial file structure using cookiecutter
- [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
- [ ] Add a model file and a training script and get that running
- [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (`pep8`) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
- [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

- [ ] Write unit tests related to the data part of your code
- [ ] Write unit tests related to model construction and or model training
- [ ] Calculate the coverage.
- [ ] Get some continuous integration running on the github repository
- [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
- [ ] Create a trigger workflow for automatically building your docker images
- [ ] Get your model training in GCP using either the Engine or Vertex AI
- [ ] Create a FastAPI application that can do inference using your model
- [ ] If applicable, consider deploying the model locally using torchserve
- [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

- [ ] Check how robust your model is towards data drifting
- [ ] Setup monitoring for the system telemetry of your deployed model
- [ ] Setup monitoring for the performance of your deployed model
- [ ] If applicable, play around with distributed data loading
- [ ] If applicable, play around with distributed model training
- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Uploaded all your code to github

## Group information

### Question 1

> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 98

### Question 2

> **Enter the study number for each member in the group**
>
> Example:
>
> _sXXXXXX, sXXXXXX, sXXXXXX_
>
> Answer:

s203211, s216410, s233478

### Question 3

> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> _We used the third-party framework ... in our project. We used functionality ... and functionality ... from the_ > _package to do ... and ... in our project_.
>
> Answer:

In this project, we utilized the [TIMM](https://timm.fast.ai/) framework, a deep-learning library offering a variety of pre-trained image processing models. Initially, we developed a custom deep-learning model comprising three convolutional layers, pooling, and fully connected layers. This model served as our baseline for evaluating suitable models. Digging into TIMM's model repository, we trained several models, assessing their performance against our baseline. The framework's flexibility in model selection and ease of integration greatly facilitated the comparative analysis and significantly contributed to the project's execution, enabling efficient experimentation.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go** > **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> _We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a_ > _complete copy of our development environment, one would have to run the following commands_
>
> Answer:

For managing dependencies in our project, we employed a combination of Makefile scripts and conda, an open-source package management and environment management system. This approach allowed us to automate the environment setup and ensure all dependencies are correctly installed, fostering a consistent development environment for all team members.

To set up an identical environment, new team members are required to perform a few simple steps. After obtaining the project's source code, they should run make `create_environment` in the terminal, which triggers the setup of a new conda environment with the necessary Python version and dependencies. Once the environment is created, they would activate it using `conda activate ml_art`. Subsequently, to install all required dependencies, they execute make requirements, which installs the exact versions of the packages specified in the project's `requirements.txt`. Finally, any additional configuration steps, if necessary, are performed as per the project documentation. This process ensures that the team member's environment mirrors the project's intended setup precisely.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your** > **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> _From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder_ > _because we did not use any ... in our project. We have added an ... folder that contains ... for running our_ > _experiments._
> Answer:

In our project, which was initialized using the [mlops_template](https://github.com/SkafteNicki/mlops_template) we tailored the repository structure to align with our project's workflow, that streamlines machine learning operations. The `data` directory is partitioned into `processed` for ready-to-use datasets and `raw` for original datasets, ensuring data integrity and facilitating reproducibility. The `models` folder contains models, predictions, and summaries, while `notebooks` serve for interactive analysis and prototyping.

Documentation is centralized in the `docs` folder, and the `reports` directory includes this generated analyses and figures. The `requirements.txt` and `requirements_dev.txt` files list the necessary libraries for running and developing the project, respectively.

Unit testing is handled within the `tests` directory, ensuring robustness and reliability for our code. The `ML-Art` directory is the core module, structured to hold data script, model code, and visualization scripts. Additionally, `train_model.py` and `predict_model.py` scripts are included at the root of the module for straightforward model training and prediction.

The `Makefile` provides convenience commands for us when deploying the model. `pyproject.toml` configures project settings, and the `LICENSE` file specifies the terms of use. This structure helps us to navigate the project with ease, promoting a good workflow from development to deployment.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these** > **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

In our project, we enforced code quality and format rules using pre-commit hooks with `Black` for consistent code formatting and `Flake8` for ensuring adherence to coding standards. These tools automate the process of maintaining a clean codebase, which is crucial in larger projects to ensure readability, reduce complexity, and facilitate collaboration among team members by standardizing the code style and catching errors early in the development cycle.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> _In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our_ > _application but also ... ._
>
> Answer:

Throughout our development process, we implemented a comprehensive suite of 6 unit tests focused on the core functionalities of our application. These tests cover data processing routines, model training/predicting stability and the integration of our machine learning pipeline components such as `omegaconf` and `logging`. By testing three critical .py scripts: `make_dataset`, `predict_model` and `train_model`, we ensure that data is correctly handled, models perform as expected, and the end-to-end workflow is robust against any future changes and additions to the code.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close** > **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our \*\*
> *code and even if we were then...\*
>
> Answer:

The current code coverage for our project stands at 60%, where all essential components, as aforementioned in question 7, only the visualization of the model performance is barely covered. Ideally, all integral components, including data handling, model creation, and training functions should be throughly tested. However, achieving 100% coverage does not guarantee an error-free codebase. While high coverage can significantly reduce the likelihood of undetected bugs by ensuring that more code paths are tested, it cannot account for every possible real-world scenario or data anomaly. Code coverage is a valuable metric for identifying untested parts of the code, but it should be complemented with other quality assurance practices such as integration testing, manual code review and such to enhance the code's reliability and robustness.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and** > **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> _We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in_ > _addition to the main branch. To merge code we ..._
>
> Answer:

Our workflow indeed incorporated the use of branches and pull requests, which are fundamental practices in collaborative version control. Each new feature or bug fixing,was developed on a separate branch, ensuring that the master branch remained stable. For instance, the `feature/unit_tests` branch, was likely used to work on unit tests specifically. Once the work on a branch was completed and tested, a pull request (PR) was opened. This triggered a code review process, allowing team members to discuss changes and request modifications before merging into the main codebase. PRs also kicked off automated tests which include `GitGuardian Security Checks` `DVC` `Unit tests` and the code formation test as previously mentioned, to ensure new changes didn't break existing functionality and maintain the coding consistance. By using this approach, we maintained a commit history with good manner and ensured that only reviewed and tested code was integrated, thereby improving the overall code quality and project maintainability.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version** > **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> _We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our_ > _pipeline_
>
> Answer:

We did not use Data Version Control (DVC) in our project, but one place (TODO). However, DVC can be incredibly beneficial in scenarios where data evolves over time, especially in projects where datasets are frequently updated with new information. For instance, if we were dealing with a continually improving dataset, where new samples were added periodically to enhance the model, DVC would allow us to track these changes efficiently. It would provide a systematic approach to record different dataset versions, making it easier to reproduce experiments, roll back to previous data states, and collaborate with team members by linking data changes to specific code updates. This ensures transparency and traceability in the development process, as every model's performance can be traced back to the exact data version as it was trained on, thereby simplifying debugging and improving reproducibility.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test** > **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of** > **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> _We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running_ > _... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>_
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would** > **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> _We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25_
>
> Answer:

In our experiments, we used Hydra library, allowing us to dynamically create and manage configurations for each run. It enabled us to define a structured configuration in YAML format, as seen in `config.yaml` under config folder inside ml_art. To run an experiment with different hyperparameters or settings, we can used Hydra's command-line overrides, like so:

```bash
python train_model.py +model=efficientnet batch_size=64 lr=0.01
```

Here, `model=efficientnet` specifies the model to use, and we override the batch size and learning rate directly from the command line without altering the `config.yaml` file. This method is powerful for running multiple experiments with varying configurations.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information** > **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> _We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment_ > _one would have to do ..._
>
> Answer:

Our approach to ensuring reproducibility hinged on controlled randomness and meticulous logging. By initializing all random number generators with a fixed seed, we guaranteed deterministic behavior in experiments. Configurations for each run were managed by Hydra, which, coupled with the `logging` package, ensured all settings were logged. The unique output directory generated by Hydra for each run contained the exact config file used, allowing for straightforward replication of experiments. For added robustness, we containerized our environment, thus preserving the experiment's context in its entirety.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking** > **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take** > **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are** > **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> _As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments._ > _As seen in the second image we are also tracking ... and ..._
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your** > **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> _For our project we developed several images: one for training, inference and deployment. For example to run the_ > _training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>_
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you** > **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> _Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling_ > _run of our main code at some point that showed ..._
>
> Answer:

When debugging, we used a multi-faceted strategy, combining the traditional print statements always easy to trace execution, `pdb` debugger for interactive inspection. Additionally, use of assertions and logging facilitated early detection of anomalies.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> _We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for..._
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs** > **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> _We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the_ > _using a custom container: ..._
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.** > **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.** > **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in** > **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and** > **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> _For deployment we wrapped our model into application using ... . We first tried locally serving the model, which_ > _worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call_ > _`curl -X POST -F "file=@file.json"<weburl>`_
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how** > **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> _We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could_ > _measure ... and ... that would inform us about this ... behaviour of our application._
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> _Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service_ > _costing the most was ... due to ..._
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.** > **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the** > **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> _The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code._ > _Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ..._
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these** > **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> _The biggest challenges in the project was using ... tool to do ... . The reason for this was ..._
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to** > **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> _Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the_ > _docker containers for training our applications._ > _Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards._ > _All members contributed to code by..._
>
> Answer:

--- question 27 fill here ---
