import signac
import numpy as np
import matplotlib.pyplot as plt
import freud
import gsd
import hoomd
import gsd.pygsd
import gsd.hoomd
from matplotlib import interactive
interactive(True)


data_path = "/home/erjank_project/ptb7-project/workspace"

project = signac.get_project(data_path)

state_dict = {"molecule": "PTB7_10mer_smiles",
              "size":"medium",
              "process":"quench",
              "density": 0.9,
              "kT_reduced": 1.7}

job_list = project.find_jobs(state_dict)

for job in job_list:
    rdf_path = job.fn("trajectory.gsd")
    print(job)

def atom_type_pos(frame, atom_types): 
    positions = []
    for idx, type_id in enumerate(frame.particles.typeid):
        if frame.particles.types[type_id] in atom_types:
            positions.append(frame.particles.position[idx])
    return positions

def create_rdf(rdf_path,
               atom_types='all',
               r_max = None,
               r_min = 0.1,
               nbins = 50,
               start = 100):
    
    
    f = gsd.pygsd.GSDFile(open(rdf_path, "rb"))
    trajectory = gsd.hoomd.HOOMDTrajectory(f) 
    if r_max is None:
        r_max = max(trajectory[-1].configuration.box[:3]) * 0.45 
    freud_rdf = freud.density.RDF(bins=nbins, r_max=r_max, r_min=r_min)
    for frame in trajectory[start:]:
        if atom_types == 'all':
            freud_rdf.compute(system=frame, reset=False)
        else:
            query_points = atom_type_pos(frame, atom_types)
            box = frame.configuration.box
            freud_rdf.compute(system=(box, query_points), reset = False)
        
 
    x = freud_rdf.bin_centers
    y = freud_rdf.rdf
    filename = job.fn('{}-trajectory.txt').format(atom_types[0])
    np.savetxt(filename, np.transpose([x,y]), delimiter=',', header= "bin_centers, rdf")
    f.close()
    return freud_rdf, filename

def rdf_plot(job_list): 
    
    for job in job_list:
        rdf_path = job.fn("trajectory.gsd")
        freud_rdf, filename = create_rdf(rdf_path, atom_types=['f'], start = 95)
        txt_path = job.fn('f-trajectory.txt')

    line = np.genfromtxt(txt_path, names = True, delimiter = ",") 

    x = line["bin_centers"]
    y = line["rdf"]
    
    plt.plot(x, y, color = '#ffc08a')

    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.ylim(0, 1.6)
    plt.title(state_dict)
    
    plt.show()
    plt.savefig("plot.png")
    plt.show()
rdf_plot(job_list)


