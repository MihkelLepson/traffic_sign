ROCKETI KÄSUD:
Esiteks tuleb ssh’da rocketisse - oma kodukataloogi. Jooksuta
```
$ module load python-3.7.1
$ ipython profile create MPI
$ echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >>/gpfs/space/home/${USER}/.ipython/profile_MPI/ipcluster_config.py

```

Siis loo fail jupyter.sbatch: (ainult esimesel korral)

```
touch jupyter.sbatch
```
Ja täida see järgneva infoga (kasuta nano või vim vms):
```

#!/bin/bash
#SBATCH -J par-jup
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --time 11000
#SBATCH --cpus-per-task 2
#SBATCH --mem 128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
module load python/3.6.3/virtenv
module load openmpi-2.1.0 #
profile=MPI

### Need to call only once in the beginning
# echo "Creating profile ${profile}"
# ipython profile create ${profile}
# echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >> /gpfs/hpchome/${USER}/.ipython/profile_${profile}/ipcluster_config.py
# cat /gpfs/hpchome/${USER}/.ipython/profile_${profile}/ipcluster_config.py

echo "Launching controller"
ipcontroller --ip="*" --profile=${profile} --log-to-file &
sleep 5

echo "Launching engines"
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file &

sleep 5
jupyter-start
sleep 5


```
Nüüd jooksuta käsud:

```
module load python/3.6.3/virtenv
virtualenv venv_example (seda ainult esimene kord)
source venv_example/bin/activate
```

Nüüd olete te nö oma keskonnas ja saate kõike läbi CLI allalaadida.

Selleks, et nüüd jupyter tööle panna jooksutage

```
sbatch jupyter.sbatch --time=11000 --mem=128G --cpus-per-task=8
```

Teile tekib vastava töö id’ga fail, mis loobib alguses erroreid, kuid siis tekib sinna URL, mis viib teid teie notebooki, urli saate kui jooksutate:

```
cat slurm-<id>.out
```
NB! Ole ka VPN eduroamiga sees.
VOILA!
