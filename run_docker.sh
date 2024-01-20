docker run -it \
  -p 9009:9009 \
  -v $(pwd):/home/repo \
  jupyter/scipy-notebook jupyter lab \
  --port=9009 \
  --no-browser \
  --ip=0.0.0.0 \
  --allow-root \
  --NotebookApp.token='' \
  --notebook-dir=/home/repo
