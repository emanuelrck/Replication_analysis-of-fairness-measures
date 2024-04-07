docker_name="fairness_measures"
conflicting_docker=$(docker ps -a -q --filter "name=^${docker_name}$")
if [[ "${conflicting_docker}" != "" ]]
then
  docker stop "${conflicting_docker}"
  docker rm "${conflicting_docker}"
fi

docker run -it \
  --name ${docker_name} \
  --user root \
  -e NB_USER="js" \
  -e CHOWN_HOME=yes \
  -w "/home/js" \
  -p 9009:9009 \
  -v $(pwd):/home/js/repo \
  jupyter/scipy-notebook jupyter lab \
  --port=9009 \
  --no-browser \
  --ip=0.0.0.0 \
  --allow-root \
  --NotebookApp.token='' \
  --notebook-dir=/home/js/repo
