FROM python:3.7.0

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN jupyter contrib nbextension install --system

EXPOSE 8888
EXPOSE 8080
EXPOSE 80

CMD /bin/bash
CMD jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token='' > ./jupyter_std_out &
