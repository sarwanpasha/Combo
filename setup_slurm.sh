#!/bin/bash
# Optional: single-node Slurm so that jobs can be submitted with sbatch on a standalone GPU machine (run once, needs sudo).
# GPUs are exposed as a generic resource, so "sbatch --gres=gpu:1" pins each job to its own GPU.
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq slurm-wlm munge > /dev/null
HOST=$(hostname -s)
CPUS=$(nproc)
MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 8000 ))
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
sudo mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo chown slurm:slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
GRES=""; LIC=""
SHARE=${GPU_SHARE:-1}   # GPU_SHARE=k: k jobs per GPU, counted with a "gpu" license; jobs pick a device themselves
if [ "$NGPU" -gt 0 ]; then
  if [ "$SHARE" -gt 1 ]; then LIC="Licenses=gpu:$((NGPU*SHARE))"; sudo rm -f /etc/slurm/gres.conf
  else GRES="Gres=gpu:$NGPU"; echo "Name=gpu File=/dev/nvidia[0-$((NGPU-1))]" | sudo tee /etc/slurm/gres.conf > /dev/null; fi
fi
sudo tee /etc/slurm/slurm.conf > /dev/null << EOF
ClusterName=local
SlurmctldHost=$HOST
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld
SlurmUser=slurm
TaskPlugin=task/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
$LIC
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
NodeName=$HOST CPUs=$CPUS RealMemory=$MEM $GRES State=UNKNOWN
PartitionName=main Nodes=$HOST Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
EOF
sudo systemctl enable --now munge
sudo systemctl restart slurmctld
sudo systemctl restart slurmd
sleep 3
sudo scontrol update nodename=$HOST state=idle || true
sinfo
echo "SLURM READY: $CPUS cpus, ${MEM}MB, $NGPU GPUs"
