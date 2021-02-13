#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
# include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/parallel.h"

#include "boson.h"
#include "entropy.h"
#include "MPSfunctions.h"

using namespace std;
using namespace itensor;

int main ( int argc, char *argv[] )
{
            //---------------------------------------------------------
    //----------------------initialize MPI---------------------
    Environment env(argc,argv);
    
    //---------------------------------------------------------
    //---------parse parameters from input file ---------------
    
    string folder = string(argv[1]);
    
    auto input = InputGroup(folder + "/input","input");
    
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",200);
    
    //system parameters
    auto N = input.getInt("N"); //
    auto Nx = input.getInt("Nx",N);
    auto Ny = input.getInt("Ny",N);
    auto nfock = input.getInt("nfock",6); //
    auto filling = input.getInt("filling", 1);
    auto periodic_x = input.getInt("periodic_x", 0) == 1;
    auto periodic_y = input.getInt("periodic_y", 0) == 1;
    
    // BH model parameters
    auto U = input.getReal("U",-1.);
    auto Jmax = input.getReal("Jmax",0.15);
    auto NJ = input.getInt("NJ", 20);
    
    //annealing parameters
    auto tmin = input.getReal("tmin", 1./abs(U));
    auto tmax = input.getReal("tmax", 200./abs(U));
    auto Nt = input.getInt("Nt", env.nnodes());
    
    auto dt = input.getReal("dt",0.01/abs(U));
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);
    
    //Setup Siteset
    int N_site = Nx*Ny;
    auto sites = Boson(N_site, args);

    
    //****************************************************************
    // OBSERVABLES
    //****************************************************************
    
    // densities
    vector<IQMPO> density;
    for (int i=1; i<=N_site; ++i){
        auto ampoxx = AutoMPO(sites);
        ampoxx += "N",i;
        IQMPO temp = IQMPO(ampoxx);
        density.push_back(temp);
    }
    
    // J values for rank
    double time_diff = (tmax-tmin)/(Nt-1);
    double t_ev = tmin + env.rank()*time_diff;
    string file_sites = folder + "/sites_ts_" + to_string(t_ev);
    
    vector<vector<double>> n;
    n.resize(NJ+1);
    
    // start loop
    double deltaJ_save = Jmax/NJ;
    
    for (int iJ = 0; iJ<= NJ; iJ++){
        
        double J = deltaJ_save*iJ;
        
        IQMPS psi;
        string file_MPS = folder + "/psi_ts_" + to_string(t_ev) + "_J_" + to_string(J);
        psi = readMPS(sites, file_sites, file_MPS);
        cout << "rank " + to_string(env.rank()) + ": state successfully loaded" << endl;
        
        for (int is = 0; is <  N_site; is++){
            // normalize
            //psi.position(is);
        
            // observable
            n[iJ].push_back(real(overlapC(psi, density[is], psi)));
        }

        
        // let all nodes finish
        env.barrier();
        if (env.rank()==0){
            cout << endl << "**************************************************************" <<endl;
            cout << "\tJ value " + to_string(iJ+1) + "/" + to_string(NJ+1) + " finished" << endl;
            cout << "**************************************************************" <<endl << endl;
        }
        
    }
    
    if (env.rank()==0) cout << "Saving data... ";
    // save data to file
    ofstream n_file;
    
    n_file.open(folder + "/n_ts_" + to_string(t_ev) + ".txt");
    
    for (int iJ = 0; iJ<= NJ; iJ++){
        for (int is = 0; is < N_site; is++){
            n_file << n[iJ][is] << " ";
        }
        n_file << endl;
    }
    
    env.barrier();
    
    if (env.rank()==0) cout << "DONE" << endl << endl;
    
   
    return 0;
}
    
    
    
    