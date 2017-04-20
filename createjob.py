import sys

for i in range(18,28):
  f = open('xj_' + str(i) + '.sh', 'w')
  f.write('#$ -cwd\n#$ -pe openmp 4-32\n#$ -l h_rt=5:0:0\n#$ -l h_vmem=8G\nmodule purge\nmodule load gcc/4.8.3 java/8 python/2.7.10\n')
  f.write('python trajectoryNet.py conf/config' + str(i) + '.json')
  f.close()
