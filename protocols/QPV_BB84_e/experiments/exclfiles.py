import os

resultfiles = os.listdir('./results/adversaries_results_over_distance/')

with open('exclfile.txt', 'w') as filehandle:
    for filename in resultfiles:
        filehandle.write(f'{filename}\n')
