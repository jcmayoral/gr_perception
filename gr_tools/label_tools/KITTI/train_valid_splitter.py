import random
import statistics
import os


def read_files():
    d = dict()
    for root,folder, files in os.walk("/home/jose/datasets/KITTI/training/image_02"):
        if len(folder )>1:
            print (folder)
            for f in folder:
                d[f] = [0,0,0,0]
            continue

        dclass = root.split("/")[-1]

        for f in files:
            if "png" in f:
                continue
            txtfile = os.path.join(root, f)
            with open (txtfile, "r") as file:
                for line in file:
                    cl, x1, y1, w,h = line.rstrip().split(" ")
                    d[dclass][int(cl)]+=1


    for key, value in d.items():
        print ("Key {} Value{}".format(key,value))
    return d

def run_try(d):
    totals = [0, 0, 0, 0]
    runs_ids = list()
    nruns = random.randint(5, len(d.values()))
    for fsample in range(nruns):
        b = [i for i in d.values()]
        maxindex = len(b)
        index = random.randint(0, maxindex - 1)
        runs_ids.append(index)
        totals[0] += b[index][0]
        totals[1] += b[index][1]
        totals[2] += b[index][2]
        totals[3] += b[index][3]

    stats = statistics.stdev(totals) / nruns
    return runs_ids, totals, stats


nclass_runs = read_files()

ids = list()
minstd = 1000000
best_soln = []
best_totals = []

for i in range(500):
    sample, totals, stats = run_try(nclass_runs)
    if stats < minstd:
        best_soln = sample
        minstd = stats
        best_totals = totals

print ("best ", best_soln, " best totals ", best_totals, " result " , minstd)