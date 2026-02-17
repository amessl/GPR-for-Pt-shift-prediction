#!/bin/bash

CLUSTER_USER="amessler"
CLUSTER_HOST="login"
REMOTE_DIR="/home/amessler/ML_models/Pt_NMR_new/src"
LOCAL_DIR="/home/alex/Pt_NMR/src"

echo "rsync from ${LOCAL_DIR} -> ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}"
rsync -avz --progress --delete \
   ${LOCAL_DIR}/ ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/
