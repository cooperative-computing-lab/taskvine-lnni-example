# SETUP (do this in one terminal instance)

## Create an empty directory and cd to it

## Clone dev cctools (install git first if no git is available)

`git clone git@github.com:tphung3/cctools.git cctools-src`

## Select the dev branch

`cd cctools-src`

`git checkout -b add_context`

`git pull origin add_context`

## Create conda environment (install miniconda first if no conda is available)

`cd ..`

`./create_env.sh`

## Install pillow in conda environment

`conda activate lnni`

`pip install pillow`

## Compile and install cctools from source

`cd cctools-src`

`./configure --with-base-dir $CONDA_PREFIX --prefix $CONDA_PREFIX`

`make install`

## Create env.tar.gz, a software tarball

`poncho_package_create lnni env.tar.gz`

## Create a sandbox for taskvine worker

`mkdir sandbox`

# RUN

`python run.py local-p`: This command runs an inference task locally to make sure no software problem is happening.

`python run.py local-s`: This command simulates the context technique in a local manner.

For the following runs, we need to set up a TaskVine worker locally.

- To set up a TaskVine worker, on another terminal on the same machine, cd to the same directory and do these:

    - `conda activate lnni` 

    - `./run_worker.sh`
    
    Remember, every run with TaskVine will spawn a manager (by running python run.py ...) and tasks are deployed to a TaskVine worker. When the manager ends with all tasks completed, feel free to SIGINT this worker process.

`python run.py remote-p`: This command runs a TaskVine manager with regular PythonTasks. These tasks are independent and share no context. You need to run a worker process side by side.
