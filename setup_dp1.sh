source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/v29.2.1/loadLSST-ext.bash
setup lsst_distrib

export REPO=/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml

setup -r ./bps_parsl_sites -j
setup -r ./source_injection_dp1 -j
setup -r ./pipe_tasks_pl -j
setup -r ./obs_base -j
setup -r ./analysis_tools_dp1 -j

eups list 
