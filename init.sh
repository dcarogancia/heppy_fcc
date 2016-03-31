SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  THIS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$THIS_DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
THIS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

source /opt/rh/python27/enable

export HEPPY_FCC=$THIS_DIR
if [ -z ${PODIO+x} ]; then
    echo 'PODIO not set'
else
    export PYTHONPATH=$PODIO/python:$PYTHONPATH
fi
export PYTHONPATH=$THIS_DIR/..:$PYTHONPATH
export ROOT_INCLUDE_PATH=$PODIO/include:$ROOT_INCLUDE_PATH
