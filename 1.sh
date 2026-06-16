#!/usr/bin/env bash
# -*- encoding=utf-8 -*-
#read  -p "Enter release > " release
#read  -p "Enter namespace > " namespace
release="xdbmysql57"
namespace="vio-algo"
backup_dir=/data/backup/${release}/${release}_0000

if [ "${release}" == "det-tn" ]; then
    backup_dir=/data/backup/tn/tn_0000
fi
 
master_pod=`kubectl get po -l release=${release} --show-labels --no-headers -n $namespace | grep mysql | grep role=leader | awk '{print $1}'`
slave_pods=`kubectl get po -l release=${release} --show-labels --no-headers -n $namespace | grep mysql | grep -v role=leader | grep -v zookeeper | awk '{print $1}'`
 
# 备份主库
kubectl -n $namespace  exec ${master_pod} -- bash -c "cd /xagent/script/backup_lightxdb; \
    mkdir -p ${backup_dir}; \
    bash backup_lightxdb_snapshot.sh backup_lightxdb.conf backup"
backup_file=`kubectl -n $namespace exec ${master_pod} -- bash -c "ls -t ${backup_dir} | head -n 1"`
kubectl -n $namespace cp ${master_pod}:${backup_dir}/${backup_file} ./${backup_file}
 
# 恢复备库
master_ip=`kubectl -n $namespace exec ${master_pod} -- bash -c 'hostname -i'`
master_port=3306
for slave_pod in ${slave_pods[@]}
do
    kubectl -n $namespace exec ${slave_pod} -- bash -c "mkdir -p ${backup_dir}"
    kubectl -n $namespace cp ./${backup_file} ${slave_pod}:${backup_dir}/${backup_file}
    kubectl -n $namespace exec ${slave_pod} -- bash -c "cd /xagent/script/backup_lightxdb; \
        sed -i 's/# g_master_ip=xxxx #/g_master_ip=${master_ip} #/g' backup_lightxdb.conf; \
        sed -i 's/# g_master_port=xxxx #/g_master_port=${master_port} #/g' backup_lightxdb.conf; \
        bash /xagent/load.sh stop; \
        bash backup_lightxdb_snapshot.sh backup_lightxdb.conf recover; \
        bash /xagent/load.sh start; \
        sed -i 's/g_master_ip=${master_ip} #/# g_master_ip=xxxx #/g' backup_lightxdb.conf; \
        sed -i 's/g_master_port=${master_port} #/# g_master_port=xxxx #/g' backup_lightxdb.conf"
done
 
kubectl -n $namespace  exec ${master_pod} -- bash -c "/xagent/bin/script/exchange_script/main check '{}' "

rm ./${backup_file}
