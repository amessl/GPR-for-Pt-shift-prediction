#!/bin/bash

CLUSTER_USER="amessler"
CLUSTER_HOST="login"
REMOTE_DIR="/home/amessler/ML_models/Pt_NMR_new/mlruns"
LOCAL_DIR="/home/alex/Pt_NMR/mlruns_cluster"

echo "rsync from ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR} -> ${LOCAL_DIR}"
rsync -avz --progress --delete \
    ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/ \
    ${LOCAL_DIR}/

if [ "$1" == "--ui" ]; then
    mlflow ui --backend-store-uri file:${LOCAL_DIR}
fi
