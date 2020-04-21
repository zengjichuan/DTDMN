PY_HOME=/research/lyu1/jczeng/anaconda2/bin/
echo "py_home=${PY_HOME}"
echo "gpu=$1"
echo "token=$2"
CUDA_VISIBLE_DEVICES=$1 ${PY_HOME}/python dtdmn_run.py --k 50 --y 10 --batch_size 16 --max_epoch=50 --token $2