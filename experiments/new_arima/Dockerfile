FROM continuumio/miniconda3:24.4.0-0

ADD ./environment.yml ./environment.yml

RUN conda install -c conda-forge -c defaults -c anaconda python=3.12 pip numpy prophet=1.1.6 numba r-base=4.3.2 r-urca r-tidyverse r-future r-future.apply r-forecast r-furrr tabulate scikit-learn pmdarima=2.0.4 fire statsforecast=2.0.1 && \
    conda clean -afy && \
    pip install datasetsforecast jupyterlab && \
    R -e "install.packages(c('tsibble', 'fable'), repos='https://cran.rstudio.com/')"
