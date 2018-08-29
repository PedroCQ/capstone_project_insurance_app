FROM python:3.6

## Upgrade PIP (and six)
RUN pip3 install --no-cache-dir --upgrade pip six

## Install stuff
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libblas-dev \
        liblapack-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8888
EXPOSE 5000

RUN pip install -U jupyter
RUN pip install -U notebook

#set default password for jupyter notebook
RUN jupyter notebook --generate-config  --allow-root
RUN echo "c.NotebookApp.password='sha1:27d1751e4a24:df2f919d862d54860de4f7021be7d46a01e52fa5'">>/root/.jupyter/jupyter_notebook_config.py

#after changes run in this directory : docker build --rm -t ldssa_final_project .
